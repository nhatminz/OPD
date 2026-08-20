from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import load_config
from .metadata import gpu_inventory


def _last_metric(path: Path) -> dict[str, Any]:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise ValueError(f"No metrics were written to {path}")
    return json.loads(lines[-1])


def _write_generated_config(
    path: Path, selected: int, report_path: Path, peak_ratio: float
) -> None:
    payload = {
        "autotune": {
            "validated": True,
            "selected_batch_size": selected,
            "peak_reserved_fraction": peak_ratio,
            "report": str(report_path),
        },
        "rollout": {"batch_size": selected},
        "training": {"micro_batch_size": 1},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated on the target B200 by scripts/smoke_test_b200.sh.\n")
        yaml.safe_dump(payload, handle, sort_keys=False)


def _update_readme(
    readme: Path, selected: int, peak_ratio: float, gpu_name: str
) -> None:
    start = "<!-- B200_AUTOTUNE_RESULT_START -->"
    end = "<!-- B200_AUTOTUNE_RESULT_END -->"
    content = readme.read_text(encoding="utf-8")
    if start not in content or end not in content:
        raise ValueError(f"Autotune result markers are missing from {readme}")
    before, remainder = content.split(start, 1)
    _, after = remainder.split(end, 1)
    block = (
        f"{start}\n"
        f"**Measured target result:** `{gpu_name}`, global batch/micro-batch **{selected}/1**, "
        f"worst peak-reserved fraction `{peak_ratio:.3f}`.\n"
        f"{end}"
    )
    readme.write_text(before + block + after, encoding="utf-8")


def run_batch_autotune(
    ta_config: str | Path,
    rac_config: str | Path,
    output_root: str | Path,
    generated_config: str | Path,
    candidates: list[int] | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Batch autotune requires CUDA")
    base = load_config(ta_config)
    candidates = candidates or [
        int(item) for item in base["autotune"]["batch_candidates"]
    ]
    candidates = sorted(set(item for item in candidates if item > 0))
    if not candidates:
        raise ValueError("At least one positive batch candidate is required")
    max_reserved_fraction = float(base["autotune"].get("max_reserved_fraction", 0.90))
    total_memory = torch.cuda.get_device_properties(0).total_memory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(output_root).resolve() / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    attempts, selected, selected_peak_ratio = [], None, None
    repo_root = Path(__file__).resolve().parents[1]

    # Monotonic memory use lets us binary-search the bounded candidate list.
    # This limits the default smoke to at most three TA and three RAC steps.
    low, high = 0, len(candidates) - 1
    while low <= high:
        candidate_index = (low + high) // 2
        batch_size = candidates[candidate_index]
        pair: dict[str, Any] = {"batch_size": batch_size, "methods": {}}
        pair_ok = True
        for method, config_path in (("ta", ta_config), ("rac", rac_config)):
            output_dir = run_root / f"{method}_batch_{batch_size}"
            command = [
                sys.executable,
                "-m",
                "b200_experiment.cli",
                "train",
                "--config",
                str(Path(config_path).resolve()),
                "--set",
                f"experiment.output_dir={output_dir}",
                "--set",
                "training.max_steps=1",
                "--set",
                "training.save_checkpoints=false",
                "--set",
                "training.micro_batch_size=1",
                "--set",
                f"rollout.batch_size={batch_size}",
                "--set",
                "logging.selected_tokens_enabled=false",
                "--set",
                "training_evaluation.enabled=false",
                "--set",
                "rollout.backend=hf",
            ]
            completed = subprocess.run(command, cwd=repo_root, check=False)
            method_result: dict[str, Any] = {
                "returncode": completed.returncode,
                "command": command,
            }
            if completed.returncode == 0:
                metric = _last_metric(output_dir / "metrics.jsonl")
                method_result["metrics"] = metric
                method_result["peak_reserved_fraction"] = (
                    metric["peak_gpu_reserved_bytes"] / total_memory
                )
                if not math.isfinite(float(metric["loss"])):
                    pair_ok = False
            else:
                pair_ok = False
            pair["methods"][method] = method_result
            if not pair_ok:
                break
        if pair_ok:
            ta_metric = pair["methods"]["ta"]["metrics"]
            rac_metric = pair["methods"]["rac"]["metrics"]
            ta_budget_ok = ta_metric["selector"]["selected_tokens"] == math.ceil(
                float(base["token_budget"]["rho"])
                * ta_metric["selector"]["valid_tokens"]
            )
            rac_all_tokens = (
                rac_metric["selector"]["selected_tokens"]
                == rac_metric["selector"]["valid_tokens"]
            )
            same_rollout = (
                ta_metric["rollout_token_sha256"] == rac_metric["rollout_token_sha256"]
            )
            peak_ratio = max(
                pair["methods"]["ta"]["peak_reserved_fraction"],
                pair["methods"]["rac"]["peak_reserved_fraction"],
            )
            pair.update(
                ta_hard_budget_valid=ta_budget_ok,
                rac_all_tokens_supervised=rac_all_tokens,
                identical_initial_rollout=same_rollout,
                peak_reserved_fraction=peak_ratio,
            )
            pair_ok = (
                ta_budget_ok
                and rac_all_tokens
                and same_rollout
                and peak_ratio <= max_reserved_fraction
            )
        pair["accepted"] = pair_ok
        attempts.append(pair)
        if pair_ok:
            selected, selected_peak_ratio = (
                batch_size,
                float(pair["peak_reserved_fraction"]),
            )
            low = candidate_index + 1
        else:
            high = candidate_index - 1

    if selected is None:
        raise RuntimeError(
            f"No batch candidate passed TA+RAC validation; see {run_root}"
        )
    report = {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_inventory(),
        "candidate_policy": {
            "candidates": candidates,
            "max_reserved_fraction": max_reserved_fraction,
            "selection": "largest tested TA+Bellman-RAC batch passing finite-loss, method-allocation, identical-rollout and memory-headroom checks",
        },
        "selected_batch_size": selected,
        "selected_micro_batch_size": 1,
        "selected_peak_reserved_fraction": selected_peak_ratio,
        "attempts": attempts,
    }
    report_path = Path(output_root).resolve() / "batch_autotune.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
    generated_path = Path(generated_config).resolve()
    _write_generated_config(
        generated_path, selected, report_path, float(selected_peak_ratio)
    )
    validation_md = repo_root / "B200_VALIDATION.md"
    validation_md.write_text(
        "# B200 validation result\n\n"
        f"- GPU: `{report['gpu'][0]['name']}` ({report['gpu'][0]['memory_bytes'] / 2**30:.1f} GiB)\n"
        f"- Selected training batch: **{selected}**\n"
        "- Global micro-batch: **1** (accumulated within each rollout)\n"
        f"- Peak reserved fraction (worst of TA/RAC): `{selected_peak_ratio:.3f}`\n"
        f"- Full report: `{report_path}`\n"
        "- Validation: both methods had finite loss and identical rollout hash; TA used its exact hard budget and Bellman-RAC supervised every valid token. No checkpoint/full run was started.\n",
        encoding="utf-8",
    )
    _update_readme(
        repo_root / "README.md",
        selected,
        float(selected_peak_ratio),
        report["gpu"][0]["name"],
    )
    return report
