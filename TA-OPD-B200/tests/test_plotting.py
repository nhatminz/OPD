from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from b200_experiment.plotting import plot_results


class PlottingTests(unittest.TestCase):
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
            for method in ("ta", "rac"):
                output = root / method
                output.mkdir()
                with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                    for step in range(1, 4):
                        handle.write(
                            json.dumps({"step": step, "loss": 1.0 / step}) + "\n"
                        )
                outputs.append(output)
            paths = plot_results(results, outputs[0], outputs[1], smoothing_window=2)
            self.assertTrue(Path(paths["accuracy"]).is_file())
            self.assertTrue(Path(paths["loss"]).is_file())


if __name__ == "__main__":
    unittest.main()
