from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
_SELECTOR_CHUNK_PATTERN = re.compile(
    r"^selected_steps_(\d+)_(\d+)(?:_rank-\d+)?\.jsonl\.gz$"
)


@dataclass(frozen=True)
class ResumeState:
    checkpoint: Path
    optimizer_path: Path
    step: int


def resolve_resume_checkpoint(
    value: str | Path | None, output_dir: str | Path | None = None
) -> Path | None:
    if value is None or not str(value).strip():
        return None
    if str(value).strip().lower() == "auto":
        if output_dir is None:
            raise ValueError("RESUME=auto requires an existing run output directory")
        root = Path(output_dir).expanduser().resolve()
        latest = root / "latest.json"
        checkpoint = None
        if latest.is_file():
            payload = json.loads(latest.read_text(encoding="utf-8"))
            candidate = Path(payload["checkpoint"])
            checkpoint = candidate if candidate.is_absolute() else root / candidate
        if checkpoint is None or not checkpoint.is_dir():
            candidates = sorted(
                (
                    path
                    for path in root.glob("checkpoint-*")
                    if path.is_dir() and (path / "optimizer.pt").is_file()
                ),
                key=lambda path: int(path.name.rsplit("-", 1)[-1]),
            )
            if (root / "final/optimizer.pt").is_file():
                candidates.append(root / "final")
            if not candidates:
                raise FileNotFoundError(
                    f"RESUME=auto found no complete checkpoint under {root}"
                )
            checkpoint = candidates[-1]
        checkpoint = checkpoint.resolve()
    else:
        checkpoint = Path(value).expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(
            f"Resume checkpoint is missing config.json: {checkpoint}"
        )
    if not (checkpoint / "optimizer.pt").is_file():
        raise FileNotFoundError(
            f"True resume requires optimizer.pt, but it is missing from {checkpoint}"
        )
    return checkpoint


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _cpu_byte_rng_state(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"Invalid {name}: expected a tensor")
    state = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if state.ndim != 1:
        raise ValueError(f"Invalid {name}: expected a one-dimensional tensor")
    return state


def _restore_rng_states(payload: dict[str, Any], device: torch.device) -> None:
    if "torch_rng_state" in payload:
        torch.set_rng_state(
            _cpu_byte_rng_state(payload["torch_rng_state"], "torch_rng_state")
        )
    if device.type != "cuda" or "cuda_rng_state_all" not in payload:
        return

    saved = payload["cuda_rng_state_all"]
    # Accept the list produced by torch.cuda.get_rng_state_all() as well as a
    # tensor from older single-GPU checkpoints.
    if torch.is_tensor(saved):
        saved_states = [saved]
    elif isinstance(saved, (list, tuple)):
        saved_states = list(saved)
    else:
        raise ValueError(
            "Invalid cuda_rng_state_all: expected a tensor or a list of tensors"
        )
    if not saved_states:
        raise ValueError("Invalid cuda_rng_state_all: no CUDA RNG states were saved")

    target_index = device.index if device.index is not None else 0
    # A checkpoint can be resumed with fewer/more visible GPUs. Restore the
    # matching logical device when present, otherwise use the sole/main state.
    source_index = target_index if target_index < len(saved_states) else 0
    state = _cpu_byte_rng_state(
        saved_states[source_index], f"cuda_rng_state_all[{source_index}]"
    )
    torch.cuda.set_rng_state(state, device=device)


def restore_optimizer(
    optimizer, checkpoint: str | Path, device: torch.device
) -> ResumeState:
    checkpoint = Path(checkpoint).resolve()
    optimizer_path = checkpoint / "optimizer.pt"
    payload = _torch_load(optimizer_path, device)
    if (
        not isinstance(payload, dict)
        or "step" not in payload
        or "optimizer" not in payload
    ):
        raise ValueError(
            f"Invalid optimizer checkpoint {optimizer_path}; expected step and optimizer"
        )
    step = int(payload["step"])
    if step < 0:
        raise ValueError(f"Resume step must be non-negative, got {step}")
    match = _CHECKPOINT_PATTERN.match(checkpoint.name)
    if match is not None and int(match.group(1)) != step:
        raise ValueError(
            f"Checkpoint directory says step {int(match.group(1))}, but "
            f"optimizer.pt says step {step}"
        )
    optimizer.load_state_dict(payload["optimizer"])
    # map_location normally handles this. The explicit walk also supports
    # optimizer states saved by older PyTorch versions on cuda:0.
    for state in optimizer.state.values():
        for key, item in state.items():
            if torch.is_tensor(item):
                state[key] = item.to(device)
    _restore_rng_states(payload, device)
    return ResumeState(checkpoint, optimizer_path, step)


def _last_jsonl_step(path: Path) -> int | None:
    if not path.is_file():
        return None
    last = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                last = int(row["step"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Cannot resume safely: malformed {path} line {line_number}"
                ) from error
    return last


def _selector_steps_after(root: Path, resume_step: int) -> list[tuple[Path, int]]:
    offending: list[tuple[Path, int]] = []
    if not root.is_dir():
        return offending
    for path in sorted(root.glob("selected_steps_*.jsonl.gz")):
        match = _SELECTOR_CHUNK_PATTERN.match(path.name)
        if match is not None and int(match.group(2)) <= resume_step:
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    step = int(json.loads(line)["training_step"])
                    if step > resume_step:
                        offending.append((path, step))
                        break
        except (OSError, EOFError, json.JSONDecodeError, KeyError, ValueError) as error:
            raise ValueError(
                f"Cannot resume safely: unreadable selector log {path}"
            ) from error
    return offending


def validate_append_history(output_dir: str | Path, resume_step: int) -> dict[str, Any]:
    """Refuse to append behind already-logged or partially-logged later work."""
    output_dir = Path(output_dir).resolve()
    metrics_step = _last_jsonl_step(output_dir / "metrics.jsonl")
    evaluation_step = _last_jsonl_step(output_dir / "eval_history.jsonl")
    later_selector = _selector_steps_after(output_dir / "selector_scores", resume_step)
    later = {
        "metrics.jsonl": metrics_step,
        "eval_history.jsonl": evaluation_step,
    }
    later = {
        name: step
        for name, step in later.items()
        if step is not None and step > resume_step
    }
    if later or later_selector:
        details = [f"{name}: step {step}" for name, step in later.items()]
        details.extend(f"{path}: step {step}" for path, step in later_selector)
        raise ValueError(
            "Cannot append this resume without duplicating/rewinding logs beyond "
            f"checkpoint step {resume_step}: " + "; ".join(details)
        )
    return {
        "metrics_last_step": metrics_step,
        "evaluation_last_step": evaluation_step,
        "selector_logs_checked": True,
    }


def _get(config: dict[str, Any], dotted: str):
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_resume_config(
    checkpoint: str | Path,
    current: dict[str, Any],
    *,
    allow_mismatch: bool = False,
) -> dict[str, Any]:
    """Check scientific settings while allowing GPU count/eval/save changes."""
    source_path = Path(checkpoint).resolve().parent / "resolved_config.yaml"
    if not source_path.is_file():
        return {"source_config": None, "checked": False, "mismatches": {}}
    source = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    keys = (
        "experiment.method",
        "experiment.seed",
        "models.student_path",
        "models.teacher_path",
        "data.path",
        "data.split",
        "rollout.backend",
        "rollout.batch_size",
        "rollout.max_new_tokens",
        "rollout.temperature",
        "rollout.top_p",
        "rollout.seed",
        "selector.top_k",
        "selector.rac_gamma",
        "selector.rac_w_min",
        "selector.rac_beta",
        "token_budget.rho",
        "training.learning_rate",
        "training.adam_betas",
        "training.weight_decay",
        "training.ppo_clip_low",
        "training.ppo_clip_high",
    )
    mismatches = {
        key: {"checkpoint": _get(source, key), "current": _get(current, key)}
        for key in keys
        if _get(source, key) != _get(current, key)
    }
    if mismatches and not allow_mismatch:
        formatted = ", ".join(
            f"{key}={values['checkpoint']!r}->{values['current']!r}"
            for key, values in mismatches.items()
        )
        raise ValueError(
            "Resume would change controlled training settings: "
            + formatted
            + ". Set RESUME_ALLOW_CONFIG_MISMATCH=true only if intentional."
        )
    return {
        "source_config": str(source_path),
        "checked": True,
        "mismatches": mismatches,
    }
