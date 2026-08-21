from __future__ import annotations

import gzip
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from b200_experiment.resume import (
    resolve_resume_checkpoint,
    restore_optimizer,
    validate_append_history,
    validate_resume_config,
)


def _controlled_config(method: str = "ta") -> dict:
    return {
        "experiment": {"method": method, "seed": 1234},
        "models": {"student_path": "/base", "teacher_path": "/teacher"},
        "data": {"path": "/data", "split": "all"},
        "rollout": {
            "backend": "vllm",
            "batch_size": 64,
            "max_new_tokens": 256,
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": 42,
        },
        "selector": {
            "top_k": 16,
            "rac_gamma": 0.995,
            "rac_w_min": 0.1,
            "rac_beta": 2.0,
        },
        "token_budget": {"rho": 0.10},
        "training": {
            "learning_rate": 1e-5,
            "adam_betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "ppo_clip_low": 0.2,
            "ppo_clip_high": 0.28,
        },
    }


class ResumeTests(unittest.TestCase):
    def test_legacy_single_gpu_optimizer_checkpoint_restores_at_step_100(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-000100"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            source_model = torch.nn.Linear(3, 2)
            source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-5)
            source_model(torch.ones(1, 3)).sum().backward()
            source_optimizer.step()
            torch.save(
                {"step": 100, "optimizer": source_optimizer.state_dict()},
                checkpoint / "optimizer.pt",
            )

            target_model = torch.nn.Linear(3, 2)
            target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=9e-4)
            resolved = resolve_resume_checkpoint(checkpoint)
            state = restore_optimizer(target_optimizer, resolved, torch.device("cpu"))
            self.assertEqual(state.step, 100)
            self.assertEqual(target_optimizer.param_groups[0]["lr"], 1e-5)
            self.assertTrue(target_optimizer.state)

    def test_cuda_rng_restore_passes_a_cpu_byte_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-000650"
            checkpoint.mkdir()
            source_model = torch.nn.Linear(3, 2)
            source_optimizer = torch.optim.SGD(source_model.parameters(), lr=1e-3)
            saved_rng_state = torch.arange(32, dtype=torch.int64)
            payload = {
                "step": 650,
                "optimizer": source_optimizer.state_dict(),
                "cuda_rng_state_all": [saved_rng_state],
            }
            target_model = torch.nn.Linear(3, 2)
            target_optimizer = torch.optim.SGD(target_model.parameters(), lr=9e-4)

            with mock.patch(
                "b200_experiment.resume.torch.load", return_value=payload
            ) as load, mock.patch(
                "b200_experiment.resume.torch.cuda.set_rng_state"
            ) as set_rng_state:
                state = restore_optimizer(
                    target_optimizer, checkpoint, torch.device("cuda", 0)
                )

            self.assertEqual(state.step, 650)
            load.assert_called_once_with(
                checkpoint / "optimizer.pt",
                map_location=torch.device("cuda", 0),
                weights_only=True,
            )
            restored_rng_state = set_rng_state.call_args.args[0]
            self.assertEqual(restored_rng_state.device.type, "cpu")
            self.assertEqual(restored_rng_state.dtype, torch.uint8)
            self.assertTrue(
                torch.equal(restored_rng_state, saved_rng_state.to(torch.uint8))
            )
            self.assertEqual(
                set_rng_state.call_args.kwargs["device"], torch.device("cuda", 0)
            )

    def test_single_saved_cuda_rng_state_can_restore_another_logical_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-000650"
            checkpoint.mkdir()
            source_model = torch.nn.Linear(3, 2)
            source_optimizer = torch.optim.SGD(source_model.parameters(), lr=1e-3)
            saved_rng_state = torch.arange(16, dtype=torch.uint8)
            torch.save(
                {
                    "step": 650,
                    "optimizer": source_optimizer.state_dict(),
                    "cuda_rng_state_all": [saved_rng_state],
                },
                checkpoint / "optimizer.pt",
            )
            target_model = torch.nn.Linear(3, 2)
            target_optimizer = torch.optim.SGD(target_model.parameters(), lr=9e-4)

            with (
                mock.patch(
                    "b200_experiment.resume.torch.load",
                    return_value={
                        "step": 650,
                        "optimizer": source_optimizer.state_dict(),
                        "cuda_rng_state_all": [saved_rng_state],
                    },
                ),
                mock.patch(
                    "b200_experiment.resume.torch.cuda.set_rng_state"
                ) as set_rng_state,
            ):
                restore_optimizer(target_optimizer, checkpoint, torch.device("cuda", 3))

            restored_rng_state = set_rng_state.call_args.args[0]
            self.assertTrue(torch.equal(restored_rng_state, saved_rng_state))
            self.assertEqual(
                set_rng_state.call_args.kwargs["device"], torch.device("cuda", 3)
            )

    def test_resume_rewinds_logs_and_artifacts_ahead_of_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-000100"
            checkpoint.mkdir()
            (root / "metrics.jsonl").write_text(
                "".join(
                    json.dumps({"step": step}) + "\n" for step in (99, 100, 101, 127)
                ),
                encoding="utf-8",
            )
            (root / "eval_history.jsonl").write_text(
                "".join(json.dumps({"step": step}) + "\n" for step in (0, 100, 120)),
                encoding="utf-8",
            )
            for filename, steps in (
                ("train_metrics.csv", (99, 100, 101, 127)),
                ("eval_metrics.csv", (0, 100, 120)),
            ):
                with (root / filename).open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=("step", "value"))
                    writer.writeheader()
                    for step in steps:
                        writer.writerow({"step": step, "value": f"step-{step}"})

            stats = root / "token_score_stats"
            stats.mkdir()
            for step in (100, 127):
                (stats / f"step-{step:06d}.json").write_text(
                    json.dumps({"step": step}) + "\n", encoding="utf-8"
                )
            evaluations = root / "training_eval"
            for step in (100, 120):
                (evaluations / f"step-{step:06d}").mkdir(parents=True)
            (root / "checkpoint-000120").mkdir()
            (root / "final").mkdir()
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            (root / "latest.json").write_text(
                json.dumps(
                    {"step": 120, "checkpoint": "checkpoint-000120", "final": False}
                )
                + "\n",
                encoding="utf-8",
            )

            result = validate_append_history(root, 100, checkpoint)

            with (root / "metrics.jsonl").open(encoding="utf-8") as handle:
                self.assertEqual(
                    [json.loads(line)["step"] for line in handle], [99, 100]
                )
            with (root / "eval_history.jsonl").open(encoding="utf-8") as handle:
                self.assertEqual(
                    [json.loads(line)["step"] for line in handle], [0, 100]
                )
            for filename, expected in (
                ("train_metrics.csv", [99, 100]),
                ("eval_metrics.csv", [0, 100]),
            ):
                with (root / filename).open(newline="", encoding="utf-8") as handle:
                    self.assertEqual(
                        [int(row["step"]) for row in csv.DictReader(handle)], expected
                    )
            self.assertTrue((stats / "step-000100.json").is_file())
            self.assertFalse((stats / "step-000127.json").exists())
            self.assertTrue((evaluations / "step-000100").is_dir())
            self.assertFalse((evaluations / "step-000120").exists())
            self.assertFalse((root / "checkpoint-000120").exists())
            self.assertFalse((root / "final").exists())
            self.assertFalse((root / "summary.json").exists())
            latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["step"], 100)
            self.assertEqual(latest["checkpoint"], "checkpoint-000100")
            self.assertTrue(result["rewound"])
            self.assertEqual(result["metrics_last_step"], 100)
            self.assertEqual(result["evaluation_last_step"], 100)
            self.assertEqual(result["removed_rows"]["metrics.jsonl"], 2)

    def test_resume_accepts_history_and_selector_logs_through_step_100(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metrics.jsonl").write_text(
                json.dumps({"step": 99}) + "\n" + json.dumps({"step": 100}) + "\n",
                encoding="utf-8",
            )
            (root / "eval_history.jsonl").write_text(
                json.dumps({"step": 0}) + "\n" + json.dumps({"step": 100}) + "\n",
                encoding="utf-8",
            )
            selector_root = root / "selector_scores"
            selector_root.mkdir()
            with gzip.open(
                selector_root / "selected_steps_000051_000100.jsonl.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps({"training_step": 100}) + "\n")
            result = validate_append_history(root, 100)
            self.assertEqual(result["metrics_last_step"], 100)
            self.assertEqual(result["evaluation_last_step"], 100)
            self.assertFalse(result["rewound"])

    def test_resume_rewrites_mixed_selector_chunk_and_deletes_later_chunk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selector_root = root / "selector_scores"
            selector_root.mkdir()
            mixed = selector_root / "selected_steps_000051_000150.jsonl.gz"
            with gzip.open(mixed, "wt", encoding="utf-8") as handle:
                for step in (99, 100, 101):
                    handle.write(json.dumps({"training_step": step}) + "\n")
            later = selector_root / "selected_steps_000151_000200.jsonl.gz"
            with gzip.open(later, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"training_step": 151}) + "\n")

            result = validate_append_history(root, 100)

            with gzip.open(mixed, "rt", encoding="utf-8") as handle:
                self.assertEqual(
                    [json.loads(line)["training_step"] for line in handle], [99, 100]
                )
            self.assertFalse(later.exists())
            self.assertEqual(result["selector_rows_removed"], 2)

    def test_resume_rewind_does_not_modify_logs_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_metrics = json.dumps({"step": 101}) + "\n"
            (root / "metrics.jsonl").write_text(original_metrics, encoding="utf-8")
            (root / "eval_history.jsonl").write_text(
                "not-valid-json\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "malformed"):
                validate_append_history(root, 100)

            self.assertEqual(
                (root / "metrics.jsonl").read_text(encoding="utf-8"),
                original_metrics,
            )
            self.assertFalse(list(root.glob(".*.resume-rewind-*.tmp")))

    def test_resume_config_prevents_switching_ta_checkpoint_to_rac(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = output / "checkpoint-000100"
            checkpoint.mkdir()
            (output / "resolved_config.yaml").write_text(
                yaml.safe_dump(_controlled_config("ta")), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "experiment.method"):
                validate_resume_config(checkpoint, _controlled_config("rac"))

    def test_auto_resume_uses_latest_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (50, 100):
                checkpoint = root / f"checkpoint-{step:06d}"
                checkpoint.mkdir()
                (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
                torch.save({"step": step, "optimizer": {}}, checkpoint / "optimizer.pt")
            resolved = resolve_resume_checkpoint("auto", root)
            self.assertEqual(resolved.name, "checkpoint-000100")


if __name__ == "__main__":
    unittest.main()
