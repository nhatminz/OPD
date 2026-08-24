from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from b200_experiment.checkpoint_evaluation import (
    _write_history_atomically,
    discover_evaluation_targets,
    main,
)
from b200_experiment.evaluation import BENCHMARK_ORDER


class CheckpointEvaluationTests(unittest.TestCase):
    @staticmethod
    def _make_run(root: Path, method: str) -> Path:
        base = root / "base"
        base.mkdir(exist_ok=True)
        (base / "config.json").write_text("{}\n", encoding="utf-8")
        teacher = root / "teacher"
        teacher.mkdir(exist_ok=True)
        (teacher / "config.json").write_text("{}\n", encoding="utf-8")
        output = root / method
        output.mkdir()
        config = {
            "experiment": {"method": method},
            "paths": {"storage_root": str(root)},
            "models": {"student_path": str(base), "teacher_path": str(teacher)},
            "data": {"path": str(root / "data")},
            "training_evaluation": {"output_subdir": "training_eval"},
            "evaluation": {
                "benchmarks": {
                    name: {"path": str(root / f"{name}.jsonl")}
                    for name in BENCHMARK_ORDER
                }
            },
        }
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
        for name in ("checkpoint-000050", "final"):
            checkpoint = output / name
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
        (output / "latest.json").write_text(
            json.dumps({"step": 100, "checkpoint": "final", "final": True}) + "\n",
            encoding="utf-8",
        )
        (output / "summary.json").write_text(
            json.dumps({"steps": 100}) + "\n", encoding="utf-8"
        )
        return output

    def test_discovers_base_numbered_and_final_checkpoints_in_step_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self._make_run(Path(temporary), "opd")

            _, _, targets, max_steps = discover_evaluation_targets(output, "opd", "OPD")

            self.assertEqual([target.step for target in targets], [0, 50, 100])
            self.assertEqual(targets[-1].model_path.name, "final")
            self.assertEqual(max_steps, 100)

    def test_unmatched_old_eval_directory_is_rejected_before_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self._make_run(Path(temporary), "ta")
            unmatched = output / "training_eval/step-000075"
            unmatched.mkdir(parents=True)

            with self.assertRaisesRegex(
                ValueError, "without a matching saved checkpoint"
            ):
                discover_evaluation_targets(output, "ta", "TA-OPD")

    def test_new_history_and_metrics_replace_old_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "eval_history.jsonl").write_text("old\n", encoding="utf-8")
            (output / "eval_metrics.csv").write_text("old\n", encoding="utf-8")
            history = [{"step": 50, "method": "rac"}]
            metrics = [
                {
                    "step": 50,
                    "method": "rac",
                    "backend": "vllm",
                    "benchmark": "MATH-500",
                    "correct": 1,
                    "total": 2,
                    "accuracy": 0.5,
                    "evaluation_time_sec": 3.0,
                }
            ]

            _write_history_atomically(output, history, metrics)

            rows = [
                json.loads(line)
                for line in (output / "eval_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            with (output / "eval_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(rows, history)
            self.assertEqual(metric_rows[0]["accuracy"], "0.5")
            self.assertFalse(any(output.glob(".*pre-reeval-*")))

    def test_dry_run_validates_all_three_methods_without_writing_eval_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {
                method: self._make_run(root, method) for method in ("opd", "ta", "rac")
            }

            return_code = main(
                [
                    "--opd-output",
                    str(outputs["opd"]),
                    "--ta-output",
                    str(outputs["ta"]),
                    "--rac-output",
                    str(outputs["rac"]),
                    "--temperature",
                    "1.0",
                    "--dry-run",
                ]
            )

            self.assertEqual(return_code, 0)
            for output in outputs.values():
                self.assertFalse((output / "eval_history.jsonl").exists())

    def test_dry_run_can_select_only_opd_without_other_output_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self._make_run(Path(temporary), "opd")

            return_code = main(
                [
                    "--methods",
                    "opd",
                    "--opd-output",
                    str(output),
                    "--temperature",
                    "1.0",
                    "--dry-run",
                ]
            )

            self.assertEqual(return_code, 0)
            self.assertFalse((output / "eval_history.jsonl").exists())

    def test_selected_method_requires_only_its_matching_output(self):
        with self.assertRaisesRegex(ValueError, "provide --opd-output"):
            main(["--methods", "opd", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
