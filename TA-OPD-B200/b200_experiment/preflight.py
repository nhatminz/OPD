from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from packaging.version import Version

from .data import read_records, record_messages, stable_sample_id
from .evaluation import inspect_evaluation_data
from .metadata import gpu_inventory
from .models import inspect_model_assets


def run_preflight(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is visible; set CUDA_VISIBLE_DEVICES before B200 preflight"
        )
    gpus = gpu_inventory()
    if bool(config["experiment"].get("require_b200", True)):
        non_b200 = [gpu for gpu in gpus if "B200" not in gpu["name"].upper()]
        if non_b200:
            detected = ", ".join(
                f"cuda:{gpu['visible_index']}={gpu['name']}" for gpu in non_b200
            )
            raise RuntimeError(
                "Every visible training GPU must be an NVIDIA B200; detected "
                + detected
            )
    records, files = read_records(
        config["data"]["path"], split=config["data"].get("split")
    )
    if not records:
        raise ValueError("Competition-MATH training dataset is empty")
    prompt_key = config["data"].get("prompt_key", "prompt")
    examples = []
    for index, row in enumerate(records[:3]):
        messages = record_messages(
            row,
            prompt_key=prompt_key,
            prefer_source_prompt=bool(
                config["data"].get("prefer_source_prompt", False)
            ),
        )
        examples.append(
            {
                "dataset_index": index,
                "sample_id": stable_sample_id(row, index),
                "messages": messages,
            }
        )
    requested_backends = {
        str(config[section].get("backend", "hf")).lower()
        for section in ("rollout", "training_evaluation", "evaluation")
        if section in config
    }
    software = {"torch": torch.__version__}
    if bool(config.get("logging", {}).get("tensorboard", {}).get("enabled", True)):
        try:
            software["tensorboard"] = version("tensorboard")
        except PackageNotFoundError as error:
            raise RuntimeError(
                "TensorBoard logging is enabled but tensorboard is not installed; "
                "run python -m pip install -r requirements.txt"
            ) from error
    if "vllm" in requested_backends:
        try:
            vllm_version = version("vllm")
        except PackageNotFoundError as error:
            raise RuntimeError(
                "vLLM rollout/evaluation is enabled but vllm is not installed; run "
                "python -m pip install -r requirements.txt inside the active venv"
            ) from error
        if not Version("0.17.1") <= Version(vllm_version) < Version("0.18"):
            raise RuntimeError(
                f"This project requires vLLM >=0.17.1,<0.18 for CUDA-IPC "
                f"rollout; found {vllm_version}. Reinstall requirements.txt"
            )
        software["vllm"] = vllm_version
    report = {
        "status": "passed",
        "gpu": gpus,
        "software": software,
        "models": inspect_model_assets(config),
        "training_data": {
            "path": str(Path(config["data"]["path"]).resolve()),
            "files": [str(path) for path in files],
            "rows": len(records),
            "split": config["data"].get("split"),
            "columns": sorted(records[0]),
            "full_dataset": True,
            "examples": examples,
        },
        "evaluation_data": inspect_evaluation_data(config),
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report
