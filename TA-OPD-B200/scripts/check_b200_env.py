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
    paths = {
        "teacher": storage / "models/Qwen3-4B",
        "student": storage / "models/Qwen3-1.7B",
        "training_data": storage / "nlp/minhpn19/data/DAPO-Math-17k-Processed",
        "math500": storage / "nlp/minhpn19/data/eval/math500",
        "aime24": storage / "nlp/minhpn19/data/eval/aime24",
        "aime25": storage / "nlp/minhpn19/data/eval/aime25",
    }
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "storage_root": str(storage),
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
