from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import BENCHMARK_ORDER, MODEL_ORDER, evaluation_metric_name


_PROGRESS_METHOD_ALIASES = {
    "all": "all",
    "three": "all",
    "both": "both",
    "opd": "opd",
    "pure-opd": "opd",
    "ta": "ta",
    "ta-opd": "ta",
    "rac": "rac",
    "bellman-rac": "rac",
}
_PROGRESS_METHODS = {
    "opd": {
        "label": "OPD",
        "slug": "opd",
        "color": "tab:green",
        "output_argument": "--opd-output",
    },
    "ta": {
        "label": "TA-OPD",
        "slug": "ta_opd",
        "color": "tab:blue",
        "output_argument": "--ta-output",
    },
    "rac": {
        "label": "Bellman-RAC",
        "slug": "rac",
        "color": "tab:orange",
        "output_argument": "--rac-output",
    },
}


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


def _normalize_progress_method(method: str) -> str:
    normalized = method.strip().lower()
    if normalized not in _PROGRESS_METHOD_ALIASES:
        choices = ", ".join(_PROGRESS_METHOD_ALIASES)
        raise ValueError(f"Unknown plot method {method!r}; expected one of: {choices}")
    return _PROGRESS_METHOD_ALIASES[normalized]


def _normalize_progress_methods(
    method: str = "both", methods: list[str] | tuple[str, ...] | None = None
) -> tuple[str, ...]:
    requested = list(methods) if methods is not None else [method]
    selected = []
    for item in requested:
        normalized = _normalize_progress_method(item)
        expanded = {
            "all": ("opd", "ta", "rac"),
            "both": ("ta", "rac"),
        }.get(normalized, (normalized,))
        for candidate in expanded:
            if candidate not in selected:
                selected.append(candidate)
    if not selected:
        raise ValueError("At least one training method must be selected for plotting")
    return tuple(selected)


def _read_eval_history(output: str | Path | None, method: str) -> list[dict]:
    spec = _PROGRESS_METHODS[method]
    if output is None:
        raise ValueError(
            f"{spec['output_argument']} is required when plotting {spec['label']}"
        )
    history_path = Path(output).resolve() / "eval_history.jsonl"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Missing {spec['label']} evaluation history: {history_path}"
        )
    rows = _read_jsonl(history_path)
    if not rows:
        raise ValueError(f"{spec['label']} evaluation history is empty: {history_path}")
    rows.sort(key=lambda row: int(row["step"]))
    for row in rows:
        missing = [
            benchmark
            for benchmark in BENCHMARK_ORDER
            if benchmark not in row.get("benchmarks", {})
            or "accuracy" not in row["benchmarks"][benchmark]
        ]
        if missing:
            raise ValueError(
                f"{spec['label']} evaluation at step {row.get('step')} is missing "
                f"accuracy for: {', '.join(missing)}"
            )
    return rows


def _history_metric_name(histories: dict[str, list[dict]]) -> str:
    """Require one evaluation protocol and return its display metric."""
    metrics = set()
    for rows in histories.values():
        for row in rows:
            configured = row.get("parameters", {}).get("metric")
            if configured:
                metrics.add(str(configured))
                continue
            samples = {
                int(result.get("samples_per_problem", 16))
                for result in row["benchmarks"].values()
            }
            if len(samples) != 1:
                raise ValueError(
                    f"Evaluation step {row.get('step')} mixes response counts: {samples}"
                )
            metrics.add(evaluation_metric_name(samples.pop()))
    if len(metrics) != 1:
        raise ValueError(
            "Cannot compare histories generated with different evaluation metrics: "
            f"{sorted(metrics)}"
        )
    return metrics.pop()


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
    outputs: dict[str, str | Path],
    smoothing_window: int,
) -> Path:
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for method, output in outputs.items():
        spec = _PROGRESS_METHODS[method]
        label, color = spec["label"], spec["color"]
        metrics = _read_jsonl(Path(output).resolve() / "metrics.jsonl")
        if not metrics:
            raise ValueError(f"{label} training metrics are empty")
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
        if method != "opd" and any("unweighted_opd_loss" in row for row in metrics):
            unweighted = [
                float(row.get("unweighted_opd_loss", row["loss"])) for row in metrics
            ]
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
    axis.set_title(" vs ".join(_PROGRESS_METHODS[item]["label"] for item in outputs))
    axis.set_title(axis.get_title() + " training loss")
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
    plots_dir: Path, outputs: dict[str, str | Path]
) -> dict[str, str]:
    result: dict[str, str] = {}
    ta_rows = (
        _read_token_stats(Path(outputs["ta"]).resolve()) if "ta" in outputs else []
    )
    rac_rows = (
        _read_token_stats(Path(outputs["rac"]).resolve()) if "rac" in outputs else []
    )
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
                axis.stairs(counts, edges, linewidth=1.7, label=f"step {row['step']}")
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


def _plot_single_method_accuracy(
    plots_dir: Path, method: str, rows: list[dict], metric_name: str
) -> Path:
    spec = _PROGRESS_METHODS[method]
    colors = {
        "MATH-500": "tab:blue",
        "AIME24": "tab:orange",
        "AIME25": "tab:green",
    }
    line_styles = {
        "MATH-500": "-",
        "AIME24": "--",
        "AIME25": "-.",
    }
    steps = [int(row["step"]) for row in rows]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for benchmark in BENCHMARK_ORDER:
        values = [float(row["benchmarks"][benchmark]["accuracy"]) for row in rows]
        axis.plot(
            steps,
            values,
            color=colors[benchmark],
            linestyle=line_styles[benchmark],
            marker="o",
            markersize=4.5,
            linewidth=2,
            label=benchmark,
        )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel(metric_name)
    axis.set_ylim(0.0, 1.05)
    axis.set_title(f"{spec['label']} evaluation {metric_name} during training")
    axis.grid(alpha=0.25)
    axis.legend(title="Dataset")
    fig.tight_layout()
    path = plots_dir / f"{spec['slug']}_accuracy_over_steps.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _write_training_history(
    results_dir: Path,
    histories: dict[str, list[dict]],
    base_accuracy: dict[str, float],
    filename_prefix: str = "",
) -> tuple[Path, Path]:
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
    csv_path = results_dir / f"{filename_prefix}training_eval_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Method", "Step", *BENCHMARK_ORDER))
        writer.writeheader()
        writer.writerows(combined_rows)
    json_path = results_dir / f"{filename_prefix}training_eval_history.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"base_accuracy": base_accuracy, "histories": histories},
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    return csv_path, json_path


def plot_training_progress(
    results_dir: str | Path,
    ta_output: str | Path | None = None,
    rac_output: str | Path | None = None,
    smoothing_window: int = 10,
    plot_name: str | None = None,
    _plots_dir: Path | None = None,
    method: str = "both",
    opd_output: str | Path | None = None,
    methods: list[str] | tuple[str, ...] | None = None,
):
    """Plot the configured evaluation metric for one, two, or three methods."""
    results_dir = Path(results_dir).resolve()
    plots_dir = _plots_dir or _plot_directory(results_dir, plot_name)
    selected_methods = _normalize_progress_methods(method, methods)
    outputs = {"opd": opd_output, "ta": ta_output, "rac": rac_output}
    histories = {
        _PROGRESS_METHODS[item]["label"]: _read_eval_history(outputs[item], item)
        for item in selected_methods
    }
    metric_name = _history_metric_name(histories)

    base_accuracy: dict[str, float] = {}
    for benchmark in BENCHMARK_ORDER:
        candidates = [
            float(rows[0]["benchmarks"][benchmark]["accuracy"])
            for rows in histories.values()
            if int(rows[0]["step"]) == 0
        ]
        if candidates:
            if len(candidates) > 1 and max(candidates) - min(candidates) > 1e-12:
                raise ValueError(
                    f"Step-0 base accuracy differs between methods for "
                    f"{benchmark}: {candidates}"
                )
            base_accuracy[benchmark] = candidates[0]

    if len(selected_methods) == 1:
        selected_method = selected_methods[0]
        rows = next(iter(histories.values()))
        progress_path = _plot_single_method_accuracy(
            plots_dir, selected_method, rows, metric_name
        )
        prefix = f"{_PROGRESS_METHODS[selected_method]['slug']}_"
        history_csv, history_json = _write_training_history(
            results_dir, histories, base_accuracy, filename_prefix=prefix
        )
        return {
            "method": _PROGRESS_METHODS[selected_method]["label"],
            "metric": metric_name,
            "accuracy_over_steps": str(progress_path),
            "history_csv": str(history_csv),
            "history_json": str(history_json),
        }

    fig, axes = plt.subplots(1, len(BENCHMARK_ORDER), figsize=(15, 4.8), sharey=True)
    colors = {
        spec["label"]: spec["color"]
        for item, spec in _PROGRESS_METHODS.items()
        if item in selected_methods
    }
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
    axes[0].set_ylabel(metric_name)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(selected_methods) + 1,
        frameon=False,
    )
    compared_names = ", ".join(
        _PROGRESS_METHODS[item]["label"] for item in selected_methods
    )
    fig.suptitle(
        f"Evaluation {metric_name} during {compared_names} training", y=1.02
    )
    fig.tight_layout()
    progress_path = plots_dir / "accuracy_over_steps.png"
    _save_figure(fig, progress_path)
    plt.close(fig)

    history_csv, history_json = _write_training_history(
        results_dir, histories, base_accuracy
    )
    selected_outputs = {
        item: outputs[item] for item in selected_methods if outputs[item] is not None
    }
    loss_path = _plot_loss_comparison(plots_dir, selected_outputs, smoothing_window)
    token_plots = _plot_token_score_distributions(plots_dir, selected_outputs)
    return {
        "methods": [_PROGRESS_METHODS[item]["label"] for item in selected_methods],
        "metric": metric_name,
        "accuracy_over_steps": str(progress_path),
        "loss": str(loss_path),
        "history_csv": str(history_csv),
        "history_json": str(history_json),
        **token_plots,
    }


def plot_results(
    results_dir: str | Path,
    ta_output: str | Path,
    rac_output: str | Path,
    smoothing_window: int = 10,
    plot_name: str | None = None,
    opd_output: str | Path | None = None,
):
    results_dir = Path(results_dir).resolve()
    plots_dir = _plot_directory(results_dir, plot_name)
    comparison_path = results_dir / "comparison.csv"
    with comparison_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {row["Method"]: row for row in rows}
    plotted_models = tuple(by_method)
    expected_models = tuple(item for item in MODEL_ORDER if item in by_method)
    if (
        not plotted_models
        or plotted_models[0] != "Base"
        or plotted_models != expected_models
    ):
        raise ValueError(
            f"comparison.csv must contain Base followed by an ordered subset of "
            f"{MODEL_ORDER[1:]}; got {plotted_models}"
        )

    metric_name = "avg@16"
    comparison_json = results_dir / "comparison.json"
    if comparison_json.is_file():
        comparison_payload = json.loads(comparison_json.read_text(encoding="utf-8"))
        configured_metrics = {
            str(detail.get("parameters", {}).get("metric"))
            for detail in comparison_payload.get("details", {}).values()
            if detail.get("parameters", {}).get("metric")
        }
        if len(configured_metrics) > 1:
            raise ValueError(
                "Cannot plot final evaluations with different metrics: "
                f"{sorted(configured_metrics)}"
            )
        if configured_metrics:
            metric_name = configured_metrics.pop()

    x = np.arange(len(BENCHMARK_ORDER))
    width = min(0.8 / len(plotted_models), 0.24)
    fig, axis = plt.subplots(figsize=(9, 5.5))
    center = (len(plotted_models) - 1) / 2
    for offset, method in enumerate(plotted_models):
        values = [float(by_method[method][benchmark]) for benchmark in BENCHMARK_ORDER]
        bars = axis.bar(x + (offset - center) * width, values, width, label=method)
        axis.bar_label(
            bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9
        )
    axis.set_xticks(x, BENCHMARK_ORDER)
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel(metric_name)
    axis.set_title("Qwen3-1.7B: " + " vs ".join(plotted_models))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    accuracy_path = plots_dir / "accuracy_comparison.png"
    _save_figure(fig, accuracy_path)
    plt.close(fig)

    training_outputs: dict[str, str | Path] = {
        "ta": ta_output,
        "rac": rac_output,
    }
    if opd_output is not None:
        training_outputs = {"opd": opd_output, **training_outputs}
    loss_path = _plot_loss_comparison(plots_dir, training_outputs, smoothing_window)
    result = {
        "metric": metric_name,
        "accuracy": str(accuracy_path),
        "loss": str(loss_path),
    }
    opd_history = (
        Path(opd_output).resolve() / "eval_history.jsonl"
        if opd_output is not None
        else None
    )
    ta_history = Path(ta_output).resolve() / "eval_history.jsonl"
    rac_history = Path(rac_output).resolve() / "eval_history.jsonl"
    if ta_history.is_file() and rac_history.is_file():
        progress_methods = ["ta", "rac"]
        if opd_history is not None and opd_history.is_file():
            progress_methods.insert(0, "opd")
        result.update(
            plot_training_progress(
                results_dir,
                ta_output,
                rac_output,
                smoothing_window,
                plot_name=plot_name,
                _plots_dir=plots_dir,
                opd_output=opd_output,
                methods=progress_methods,
            )
        )
    return result
