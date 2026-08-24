from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from b200_experiment.plotting import plot_results, plot_training_progress


class PlottingTests(unittest.TestCase):
    def test_single_response_history_is_labeled_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "opd"
            self._write_training_output(output, 0.25)
            history_path = output / "eval_history.jsonl"
            rows = [json.loads(line) for line in history_path.read_text().splitlines()]
            for row in rows:
                row["parameters"] = {"metric": "accuracy", "num_responses": 1}
                for result in row["benchmarks"].values():
                    result["samples_per_problem"] = 1
            history_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            paths = plot_training_progress(
                root / "results", opd_output=output, methods=["opd"]
            )

            self.assertEqual(paths["metric"], "accuracy")
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())

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

    def test_final_plot_accepts_base_and_all_three_trained_methods(self):
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
                for index, method in enumerate(("Base", "OPD", "TA-OPD", "RAC")):
                    writer.writerow(
                        {
                            "Method": method,
                            "MATH-500": 0.1 + index / 10,
                            "AIME24": 0.2 + index / 10,
                            "AIME25": 0.3 + index / 10,
                        }
                    )
            outputs = {}
            for method, accuracy in (("opd", 0.2), ("ta", 0.3), ("rac", 0.4)):
                outputs[method] = root / method
                self._write_training_output(outputs[method], accuracy)

            paths = plot_results(
                results,
                outputs["ta"],
                outputs["rac"],
                smoothing_window=2,
                opd_output=outputs["opd"],
            )

            self.assertEqual(paths["methods"], ["OPD", "TA-OPD", "Bellman-RAC"])
            self.assertTrue(Path(paths["accuracy"]).is_file())
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())

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

    def test_new_history_adds_competition_math_to_plot_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "opd"
            self._write_training_output(output, 0.25)
            history_path = output / "eval_history.jsonl"
            rows = [json.loads(line) for line in history_path.read_text().splitlines()]
            for row in rows:
                row["benchmarks"] = {
                    "Competition-MATH": {"accuracy": 0.4},
                    **row["benchmarks"],
                }
            history_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            paths = plot_training_progress(
                root / "results", opd_output=output, methods=["opd"]
            )

            with Path(paths["history_csv"]).open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(
                tuple(csv_rows[0]),
                ("Method", "Step", "Competition-MATH", "MATH-500", "AIME24", "AIME25"),
            )
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())

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

    def test_single_opd_plot_does_not_require_ta_or_rac_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            opd_output = root / "opd"
            self._write_training_output(opd_output, 0.20)

            paths = plot_training_progress(
                root / "results", opd_output=opd_output, methods=["opd"]
            )

            self.assertEqual(paths["method"], "OPD")
            self.assertEqual(
                Path(paths["accuracy_over_steps"]).name,
                "opd_accuracy_over_steps.png",
            )

    def test_any_two_or_all_three_methods_can_be_compared(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {}
            for method, accuracy in (("opd", 0.2), ("ta", 0.3), ("rac", 0.4)):
                outputs[method] = root / method
                self._write_training_output(outputs[method], accuracy)

            pair = plot_training_progress(
                root / "pair-results",
                opd_output=outputs["opd"],
                rac_output=outputs["rac"],
                methods=["opd", "rac"],
                plot_name="opd_vs_rac",
            )
            all_methods = plot_training_progress(
                root / "all-results",
                opd_output=outputs["opd"],
                ta_output=outputs["ta"],
                rac_output=outputs["rac"],
                methods=["opd", "ta", "rac"],
                plot_name="all_three",
            )

            self.assertEqual(pair["methods"], ["OPD", "Bellman-RAC"])
            self.assertEqual(all_methods["methods"], ["OPD", "TA-OPD", "Bellman-RAC"])
            self.assertTrue(Path(pair["accuracy_over_steps"]).is_file())
            self.assertTrue(Path(all_methods["accuracy_over_steps"]).is_file())
            with Path(pair["history_csv"]).open(newline="", encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["Method"] for row in pair_rows},
                {"Base", "OPD", "Bellman-RAC"},
            )

    def test_selected_method_requires_its_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "--rac-output is required"):
                plot_training_progress(Path(temporary), method="rac")

    def test_all_methods_require_opd_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "--opd-output is required"):
                plot_training_progress(Path(temporary), method="all")


if __name__ == "__main__":
    unittest.main()
