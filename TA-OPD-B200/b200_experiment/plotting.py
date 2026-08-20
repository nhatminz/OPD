from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import BENCHMARK_ORDER, MODEL_ORDER


def _plot_directory(results_dir: Path, plot_name: str | None) -> Path:
    name = plot_name or f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not name.replace("-", "").replace(".", "").replace("_", "").isalnum():
        raise ValueError(f"Invalid plot output name: {name!r}")
    path = results_dir / "plots" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


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


def _plot_loss_comparison(
    plots_dir: Path,
    ta_output: str | Path,
    rac_output: str | Path,
    smoothing_window: int,
) -> Path:
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for label, output, color in (
        ("TA-OPD", ta_output, "tab:blue"),
        ("Bellman-RAC", rac_output, "tab:orange"),
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
        if any("unweighted_opd_loss" in row for row in metrics):
            unweighted = [float(row.get("unweighted_opd_loss", row["loss"])) for row in metrics]
            axis.plot(
                steps,
                _moving_average(unweighted, smoothing_window),
                color=color,
                linewidth=1.2,
                linestyle=":",
                label=f"{label} unweighted",
            )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("OPD loss")
    axis.set_title("TA-OPD vs RAC training loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = plots_dir / "loss_comparison.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _read_token_stats(output: Path) -> list[dict]:
    root = output / "token_score_stats"
    if not root.is_dir():
        return []
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("step-*.json"))
    ]
    return sorted(rows, key=lambda row: int(row["step"]))


def _snapshot_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    indices = sorted({0, len(rows) // 2, len(rows) - 1})
    return [rows[index] for index in indices]


def _plot_token_score_distributions(
    plots_dir: Path, ta_output: Path, rac_output: Path
) -> dict[str, str]:
    result: dict[str, str] = {}
    ta_rows, rac_rows = _read_token_stats(ta_output), _read_token_stats(rac_output)
    if ta_rows:
        fig, axis = plt.subplots(figsize=(8.5, 5.2))
        for row in _snapshot_rows(ta_rows):
            histogram = row["scores"]["s_TA"]["histogram"]
            edges = np.asarray(histogram["edges"], dtype=float)
            counts = np.asarray(histogram["counts"], dtype=float)
            counts /= max(counts.sum(), 1.0)
            axis.stairs(counts, edges, linewidth=1.8, label=f"step {row['step']}")
        axis.set(xlabel="TA local teachability s_teach", ylabel="Token fraction")
        axis.set_title("TA-OPD token-score distribution")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        path = plots_dir / "ta_token_score_distribution.png"
        _save_figure(fig, path)
        plt.close(fig)
        result["ta_token_scores"] = str(path)
    if rac_rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for axis, key, title in zip(
            axes,
            ("g", "V", "w"),
            ("Local g", "Bellman V", "Soft weight w"),
        ):
            for row in _snapshot_rows(rac_rows):
                histogram = row["scores"][key]["histogram"]
                edges = np.asarray(histogram["edges"], dtype=float)
                counts = np.asarray(histogram["counts"], dtype=float)
                counts /= max(counts.sum(), 1.0)
                axis.stairs(
                    counts, edges, linewidth=1.7, label=f"step {row['step']}"
                )
            axis.set_title(title)
            axis.set_xlabel("Score")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Token fraction")
        axes[-1].legend()
        fig.suptitle("Bellman-RAC token-score distributions")
        fig.tight_layout()
        path = plots_dir / "rac_token_score_distributions.png"
        _save_figure(fig, path)
        plt.close(fig)
        result["rac_token_scores"] = str(path)

        fig, axis = plt.subplots(figsize=(8.5, 5.2))
        steps = [int(row["step"]) for row in rac_rows]
        for key, label in (
            ("alignment", "mean alignment"),
            ("V", "mean V"),
            ("w", "mean weight"),
        ):
            axis.plot(
                steps,
                [float(row["scores"][key]["mean"]) for row in rac_rows],
                marker="o",
                linewidth=1.8,
                label=label,
            )
        axis.set(xlabel="Optimizer step", ylabel="Mean over valid tokens")
        axis.set_ylim(0.0, 1.05)
        axis.set_title("Bellman-RAC score means")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        path = plots_dir / "rac_score_means.png"
        _save_figure(fig, path)
        plt.close(fig)
        result["rac_score_means"] = str(path)
    return result


def plot_training_progress(
    results_dir: str | Path,
    ta_output: str | Path,
    rac_output: str | Path,
    smoothing_window: int = 10,
    plot_name: str | None = None,
    _plots_dir: Path | None = None,
):
    """Plot checkpoint accuracy trajectories and the constant step-0 base."""
    results_dir = Path(results_dir).resolve()
    plots_dir = _plots_dir or _plot_directory(results_dir, plot_name)
    histories = {
        "TA-OPD": _read_jsonl(Path(ta_output).resolve() / "eval_history.jsonl"),
        "Bellman-RAC": _read_jsonl(Path(rac_output).resolve() / "eval_history.jsonl"),
    }
    for method, rows in histories.items():
        if not rows:
            raise ValueError(f"{method} evaluation history is empty")
        rows.sort(key=lambda row: int(row["step"]))

    base_accuracy: dict[str, float] = {}
    for benchmark in BENCHMARK_ORDER:
        candidates = [
            float(rows[0]["benchmarks"][benchmark]["accuracy"])
            for rows in histories.values()
            if int(rows[0]["step"]) == 0
        ]
        if candidates:
            if max(candidates) - min(candidates) > 1e-12:
                raise ValueError(
                    f"Step-0 base accuracy differs between TA/RAC for {benchmark}: {candidates}"
                )
            base_accuracy[benchmark] = candidates[0]

    fig, axes = plt.subplots(1, len(BENCHMARK_ORDER), figsize=(15, 4.8), sharey=True)
    colors = {"TA-OPD": "tab:blue", "Bellman-RAC": "tab:orange"}
    for axis, benchmark in zip(axes, BENCHMARK_ORDER):
        maximum_step = 0
        for method, rows in histories.items():
            steps = [int(row["step"]) for row in rows]
            values = [float(row["benchmarks"][benchmark]["accuracy"]) for row in rows]
            maximum_step = max(maximum_step, max(steps))
            axis.plot(
                steps,
                values,
                color=colors[method],
                marker="o",
                linewidth=2,
                label=method,
            )
            for step, value in zip(steps, values):
                if step == 0:
                    continue
                axis.annotate(
                    f"{value:.3f}",
                    (step, value),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                    color=colors[method],
                )
        if benchmark in base_accuracy:
            axis.hlines(
                base_accuracy[benchmark],
                0,
                maximum_step,
                colors="black",
                linestyles="--",
                linewidth=1.5,
                label="Base Qwen3-1.7B",
            )
        axis.set_title(benchmark)
        axis.set_xlabel("Optimizer step")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Evaluation accuracy during TA-OPD and RAC training", y=1.02)
    fig.tight_layout()
    progress_path = plots_dir / "accuracy_over_steps.png"
    _save_figure(fig, progress_path)
    plt.close(fig)

    combined_rows = []
    if all(benchmark in base_accuracy for benchmark in BENCHMARK_ORDER):
        combined_rows.append(
            {
                "Method": "Base",
                "Step": 0,
                **{
                    benchmark: base_accuracy[benchmark] for benchmark in BENCHMARK_ORDER
                },
            }
        )
    for method, rows in histories.items():
        for row in rows:
            combined_rows.append(
                {
                    "Method": method,
                    "Step": int(row["step"]),
                    **{
                        benchmark: float(row["benchmarks"][benchmark]["accuracy"])
                        for benchmark in BENCHMARK_ORDER
                    },
                }
            )
    with (results_dir / "training_eval_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("Method", "Step", *BENCHMARK_ORDER))
        writer.writeheader()
        writer.writerows(combined_rows)
    with (results_dir / "training_eval_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"base_accuracy": base_accuracy, "histories": histories},
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    loss_path = _plot_loss_comparison(
        plots_dir, ta_output, rac_output, smoothing_window
    )
    token_plots = _plot_token_score_distributions(
        plots_dir, Path(ta_output).resolve(), Path(rac_output).resolve()
    )
    return {
        "accuracy_over_steps": str(progress_path),
        "loss": str(loss_path),
        "history_csv": str(results_dir / "training_eval_history.csv"),
        "history_json": str(results_dir / "training_eval_history.json"),
        **token_plots,
    }


def plot_results(
    results_dir: str | Path,
    ta_output: str | Path,
    rac_output: str | Path,
    smoothing_window: int = 10,
    plot_name: str | None = None,
):
    results_dir = Path(results_dir).resolve()
    plots_dir = _plot_directory(results_dir, plot_name)
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
    _save_figure(fig, accuracy_path)
    plt.close(fig)

    loss_path = _plot_loss_comparison(
        plots_dir, ta_output, rac_output, smoothing_window
    )
    result = {"accuracy": str(accuracy_path), "loss": str(loss_path)}
    ta_history = Path(ta_output).resolve() / "eval_history.jsonl"
    rac_history = Path(rac_output).resolve() / "eval_history.jsonl"
    if ta_history.is_file() and rac_history.is_file():
        result.update(
            plot_training_progress(
                results_dir,
                ta_output,
                rac_output,
                smoothing_window,
                plot_name=plot_name,
                _plots_dir=plots_dir,
            )
        )
    return result
