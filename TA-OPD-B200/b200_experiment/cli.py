from __future__ import annotations

import argparse
import json
import sys

from .autotune import run_batch_autotune
from .config import apply_overrides, load_with_overlays, resolve_runtime_paths
from .evaluation import aggregate_evaluations, evaluate_suite
from .plotting import plot_results, plot_training_progress
from .preflight import run_preflight
from .trainer import run_training


def _configured(args):
    return resolve_runtime_paths(
        apply_overrides(
            load_with_overlays(args.config, getattr(args, "overlay", [])),
            getattr(args, "overrides", []),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone B200 TA-OPD versus RAC experiment"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--overlay", action="append", default=[])
    train.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--overlay", action="append", default=[])
    preflight.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )
    preflight.add_argument("--output", required=True)

    tune = commands.add_parser("autotune-batch")
    tune.add_argument("--ta-config", required=True)
    tune.add_argument("--rac-config", required=True)
    tune.add_argument("--output", required=True)
    tune.add_argument("--generated-config", required=True)
    tune.add_argument("--candidates", nargs="+", type=int)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--overlay", action="append", default=[])
    evaluate.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )
    evaluate.add_argument("--name", required=True, choices=("Base", "TA-OPD", "RAC"))
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--output", required=True)

    aggregate = commands.add_parser("aggregate-eval")
    aggregate.add_argument("--base-dir", required=True)
    aggregate.add_argument("--ta-dir", required=True)
    aggregate.add_argument("--rac-dir", required=True)
    aggregate.add_argument("--output", required=True)

    plot = commands.add_parser("plot")
    plot.add_argument("--results", required=True)
    plot.add_argument("--ta-output", required=True)
    plot.add_argument("--rac-output", required=True)
    plot.add_argument("--smoothing-window", type=int, default=10)
    plot.add_argument("--plot-name")

    progress_plot = commands.add_parser("plot-training-progress")
    progress_plot.add_argument("--results", required=True)
    progress_plot.add_argument("--ta-output", required=True)
    progress_plot.add_argument("--rac-output", required=True)
    progress_plot.add_argument("--smoothing-window", type=int, default=10)
    progress_plot.add_argument("--plot-name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        run_training(_configured(args), command_line=sys.argv)
        return 0
    if args.command == "preflight":
        result = run_preflight(_configured(args), args.output)
    elif args.command == "autotune-batch":
        result = run_batch_autotune(
            args.ta_config,
            args.rac_config,
            args.output,
            args.generated_config,
            args.candidates,
        )
    elif args.command == "evaluate":
        result = evaluate_suite(args.name, args.model, _configured(args), args.output)
    elif args.command == "aggregate-eval":
        result = aggregate_evaluations(
            {"Base": args.base_dir, "TA-OPD": args.ta_dir, "RAC": args.rac_dir},
            args.output,
        )
    elif args.command == "plot":
        result = plot_results(
            args.results,
            args.ta_output,
            args.rac_output,
            args.smoothing_window,
            args.plot_name,
        )
    elif args.command == "plot-training-progress":
        result = plot_training_progress(
            args.results,
            args.ta_output,
            args.rac_output,
            args.smoothing_window,
            args.plot_name,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
