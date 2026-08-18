from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import BENCHMARK_ORDER, MODEL_ORDER


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _moving_average(values: list[float], window: int) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    if window <= 1:
        return values_array
    result = np.empty_like(values_array)
    for index in range(len(values_array)):
        result[index] = values_array[max(0, index - window + 1) : index + 1].mean()
    return result


def plot_results(
    results_dir: str | Path,
    ta_output: str | Path,
    rac_output: str | Path,
    smoothing_window: int = 10,
):
    results_dir = Path(results_dir).resolve()
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = results_dir / "comparison.csv"
    with comparison_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {row["Method"]: row for row in rows}
    if tuple(by_method) != MODEL_ORDER:
        raise ValueError(f"comparison.csv must contain exactly {MODEL_ORDER}")

    x = np.arange(len(BENCHMARK_ORDER))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for offset, method in enumerate(MODEL_ORDER):
        values = [float(by_method[method][benchmark]) for benchmark in BENCHMARK_ORDER]
        bars = axis.bar(x + (offset - 1) * width, values, width, label=method)
        axis.bar_label(
            bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9
        )
    axis.set_xticks(x, BENCHMARK_ORDER)
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Accuracy")
    axis.set_title("Qwen3-1.7B: Base vs TA-OPD vs RAC")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    accuracy_path = plots_dir / "accuracy_comparison.png"
    fig.savefig(accuracy_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    for label, output, color in (
        ("TA-OPD", ta_output, "tab:blue"),
        ("RAC", rac_output, "tab:orange"),
    ):
        metrics = _read_jsonl(Path(output).resolve() / "metrics.jsonl")
        steps = [int(row["step"]) for row in metrics]
        losses = [float(row["loss"]) for row in metrics]
        axis.plot(
            steps, losses, color=color, alpha=0.25, linewidth=1, label=f"{label} raw"
        )
        axis.plot(
            steps,
            _moving_average(losses, smoothing_window),
            color=color,
            linewidth=2,
            label=f"{label} MA({smoothing_window})",
        )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("OPD loss")
    axis.set_title("TA-OPD vs RAC training loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    loss_path = plots_dir / "loss_comparison.png"
    fig.savefig(loss_path, dpi=180)
    plt.close(fig)
    return {"accuracy": str(accuracy_path), "loss": str(loss_path)}
