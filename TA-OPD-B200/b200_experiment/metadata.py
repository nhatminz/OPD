from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def _command(args, cwd: Path) -> str:
    try:
        return subprocess.run(
            args, cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception as error:
        return f"unavailable: {type(error).__name__}: {error}"


def gpu_inventory() -> list[dict]:
    inventory = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        inventory.append(
            {
                "visible_index": index,
                "name": props.name,
                "memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return inventory


def collect_metadata(repo_root, command_line, model_metadata, data_path, data_files):
    root = Path(repo_root).resolve()
    packages = {}
    for name in (
        "transformers",
        "accelerate",
        "peft",
        "pandas",
        "pyarrow",
        "matplotlib",
        "math_verify",
        "tqdm",
    ):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            packages[name] = f"unavailable: {type(error).__name__}"
    hashes = {}
    for path in root.rglob("*"):
        if not path.is_file() or any(
            part in {"outputs", "results", "__pycache__", ".git"} for part in path.parts
        ):
            continue
        if path.suffix in {".py", ".sh", ".yaml", ".md", ".txt"}:
            hashes[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return {
        "project": "standalone TA-OPD-B200",
        "git_commit": _command(["git", "rev-parse", "HEAD"], root),
        "git_status": _command(["git", "status", "--short"], root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": packages,
        "gpu": gpu_inventory(),
        "command_line": command_line,
        "models": model_metadata,
        "data_path": str(Path(data_path).resolve()),
        "data_files": [str(Path(item).resolve()) for item in data_files],
        "workspace_code_sha256": hashes,
    }


def save_metadata(
    metadata: dict, output_dir: str | Path, filename: str = "run_metadata.json"
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / filename).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
