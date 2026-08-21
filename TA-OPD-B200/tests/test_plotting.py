from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from b200_experiment.plotting import plot_results, plot_training_progress


class PlottingTests(unittest.TestCase):
    @staticmethod
    def _write_training_output(output: Path, accuracy: float) -> None:
        output.mkdir()
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for step in range(1, 4):
                handle.write(json.dumps({"step": step, "loss": 1.0 / step}) + "\n")
        with (output / "eval_history.jsonl").open("w", encoding="utf-8") as handle:
            for step in (0, 100):
                value = 0.1 if step == 0 else accuracy
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "benchmarks": {
                                name: {"accuracy": value}
                                for name in ("MATH-500", "AIME24", "AIME25")
                            },
                        }
                    )
                    + "\n"
                )

    def test_accuracy_and_loss_plots_are_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            with (results / "comparison.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("Method", "MATH-500", "AIME24", "AIME25")
                )
                writer.writeheader()
                for index, method in enumerate(("Base", "TA-OPD", "RAC")):
                    writer.writerow(
                        {
                            "Method": method,
                            "MATH-500": 0.1 + index / 10,
                            "AIME24": 0.2,
                            "AIME25": 0.3,
                        }
                    )
            outputs = []
            for method, accuracy in (("ta", 0.2), ("rac", 0.3)):
                output = root / method
                self._write_training_output(output, accuracy)
                outputs.append(output)
            paths = plot_results(results, outputs[0], outputs[1], smoothing_window=2)
            self.assertTrue(Path(paths["accuracy"]).is_file())
            self.assertTrue(Path(paths["loss"]).is_file())
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())
            self.assertTrue(Path(paths["history_csv"]).is_file())

    def test_single_method_plot_contains_all_benchmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            ta_output = root / "ta"
            self._write_training_output(ta_output, 0.25)

            paths = plot_training_progress(
                results,
                ta_output=ta_output,
                method="ta",
                plot_name="ta_report",
            )

            self.assertEqual(paths["method"], "TA-OPD")
            self.assertEqual(
                Path(paths["accuracy_over_steps"]).name,
                "ta_opd_accuracy_over_steps.png",
            )
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())
            self.assertTrue(
                Path(paths["accuracy_over_steps"]).with_suffix(".pdf").is_file()
            )
            with Path(paths["history_csv"]).open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                tuple(rows[0]),
                ("Method", "Step", "MATH-500", "AIME24", "AIME25"),
            )
            self.assertEqual({row["Method"] for row in rows}, {"Base", "TA-OPD"})

    def test_single_rac_plot_does_not_require_ta_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rac_output = root / "rac"
            self._write_training_output(rac_output, 0.35)

            paths = plot_training_progress(
                root / "results", rac_output=rac_output, method="bellman-rac"
            )

            self.assertEqual(paths["method"], "Bellman-RAC")
            self.assertEqual(
                Path(paths["accuracy_over_steps"]).name,
                "rac_accuracy_over_steps.png",
            )

    def test_selected_method_requires_its_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "--rac-output is required"):
                plot_training_progress(Path(temporary), method="rac")


if __name__ == "__main__":
    unittest.main()
