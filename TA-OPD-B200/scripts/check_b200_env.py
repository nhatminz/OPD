#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import torch


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def main() -> int:
    storage = Path(os.environ.get("STORAGE_ROOT", "/workspace/storage-shared"))

    def asset_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else storage / path

    dataset = os.environ.get("TRAIN_DATASET", "competition_math").lower().replace(
        "-", "_"
    )
    if dataset in {"competition_math", "math"}:
        default_train = (
            "nlp/minhpn19/data/competition_math/data/"
            "train-00000-of-00001.parquet"
        )
    elif dataset in {"dapo_math", "dapo"}:
        default_train = "nlp/minhpn19/data/DAPO-Math-17k-Processed"
    elif dataset == "custom":
        default_train = ""
    else:
        raise ValueError(
            f"Unknown TRAIN_DATASET={dataset!r}; use competition_math, dapo_math, or custom"
        )
    train_value = os.environ.get("TRAIN_DATA_PATH", default_train)
    if not train_value:
        raise ValueError("TRAIN_DATASET=custom requires TRAIN_DATA_PATH")

    paths = {
        "teacher": asset_path(
            os.environ.get(
                "TEACHER_MODEL_PATH",
                os.environ.get("TEACHER_PATH", "models/Qwen3-8B"),
            )
        ),
        "student": asset_path(
            os.environ.get(
                "STUDENT_MODEL_PATH",
                os.environ.get(
                    "STUDENT_PATH",
                    "nlp/tungdd11/stable-on-policy-distillation/OPD/model/"
                    "Qwen3-1.7B-Base",
                ),
            )
        ),
        "training_data": asset_path(train_value),
        "competition_math_test": asset_path(
            os.environ.get(
                "COMPETITION_MATH_TEST_PATH",
                "nlp/minhpn19/data/competition_math/data/"
                "test-00000-of-00001.parquet",
            )
        ),
        "math500": storage / "nlp/minhpn19/data/eval/math500",
        "aime24": storage / "nlp/minhpn19/data/eval/aime24",
        "aime25": storage / "nlp/minhpn19/data/eval/aime25",
    }
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "storage_root": str(storage),
        "training_dataset_preset": dataset,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_gib": round(
                    torch.cuda.get_device_properties(index).total_memory / 2**30, 2
                ),
            }
            for index in range(torch.cuda.device_count())
        ],
        "packages": {
            name: _version(name)
            for name in (
                "transformers",
                "accelerate",
                "vllm",
                "pandas",
                "pyarrow",
                "matplotlib",
                "math-verify",
                "tensorboard",
            )
        },
        "paths": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in paths.items()
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    missing = [name for name, detail in report["paths"].items() if not detail["exists"]]
    if (
        not report["cuda_available"]
        or report["packages"]["vllm"] == "missing"
        or report["packages"]["tensorboard"] == "missing"
        or missing
    ):
        print(
            "WARNING: environment is not ready for the full B200 run; inspect the report above.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
