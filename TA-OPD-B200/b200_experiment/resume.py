from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
_SELECTOR_CHUNK_PATTERN = re.compile(
    r"^selected_steps_(\d+)_(\d+)(?:_rank-\d+)?\.jsonl\.gz$"
)
_STEP_JSON_PATTERN = re.compile(r"^step-(\d+)\.json$")
_STEP_DIRECTORY_PATTERN = re.compile(r"^step-(\d+)$")


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


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.resume-rewind-{uuid.uuid4().hex}.tmp")


def _stage_jsonl_rewind(
    path: Path, resume_step: int, step_field: str
) -> tuple[Path | None, int | None, int]:
    if not path.is_file():
        return None, None, 0
    temporary = _temporary_sibling(path)
    retained_last_step = None
    removed_rows = 0
    try:
        with path.open(encoding="utf-8") as source, temporary.open(
            "x", encoding="utf-8"
        ) as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    target.write(line)
                    continue
                try:
                    step = int(json.loads(line)[step_field])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Cannot resume safely: malformed {path} line {line_number}"
                    ) from error
                if step <= resume_step:
                    target.write(line)
                    if not line.endswith("\n"):
                        target.write("\n")
                    retained_last_step = (
                        step
                        if retained_last_step is None
                        else max(retained_last_step, step)
                    )
                else:
                    removed_rows += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if removed_rows == 0:
        temporary.unlink()
        return None, retained_last_step, 0
    return temporary, retained_last_step, removed_rows


def _stage_csv_rewind(
    path: Path, resume_step: int
) -> tuple[Path | None, int | None, int]:
    if not path.is_file():
        return None, None, 0
    temporary = _temporary_sibling(path)
    retained_last_step = None
    removed_rows = 0
    try:
        with path.open(newline="", encoding="utf-8") as source, temporary.open(
            "x", newline="", encoding="utf-8"
        ) as target:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "step" not in reader.fieldnames:
                raise ValueError(f"Cannot resume safely: {path} has no step column")
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for line_number, row in enumerate(reader, start=2):
                try:
                    step = int(row["step"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Cannot resume safely: malformed {path} line {line_number}"
                    ) from error
                if step <= resume_step:
                    writer.writerow(row)
                    retained_last_step = (
                        step
                        if retained_last_step is None
                        else max(retained_last_step, step)
                    )
                else:
                    removed_rows += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if removed_rows == 0:
        temporary.unlink()
        return None, retained_last_step, 0
    return temporary, retained_last_step, removed_rows


def _stage_selector_rewind(
    path: Path, resume_step: int
) -> tuple[Path | None, bool, int]:
    temporary = _temporary_sibling(path)
    retained_rows = 0
    removed_rows = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source, gzip.open(
            temporary, "xt", encoding="utf-8", compresslevel=6
        ) as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    step = int(json.loads(line)["training_step"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Cannot resume safely: malformed {path} line {line_number}"
                    ) from error
                if step <= resume_step:
                    target.write(line)
                    if not line.endswith("\n"):
                        target.write("\n")
                    retained_rows += 1
                else:
                    removed_rows += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if removed_rows == 0:
        temporary.unlink()
        return None, False, 0
    if retained_rows == 0:
        temporary.unlink()
        return None, True, removed_rows
    return temporary, False, removed_rows


def _step_paths_after(root: Path, pattern: re.Pattern, resume_step: int) -> list[Path]:
    if not root.is_dir():
        return []
    stale = []
    for path in root.iterdir():
        match = pattern.match(path.name)
        if match is not None and int(match.group(1)) > resume_step:
            stale.append(path)
    return sorted(stale)


def validate_append_history(
    output_dir: str | Path,
    resume_step: int,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically rewind append-only outputs to the selected checkpoint step."""
    if resume_step < 0:
        raise ValueError(f"Resume step must be non-negative, got {resume_step}")
    output_dir = Path(output_dir).resolve()
    replacements: list[tuple[Path, Path]] = []
    deletions: set[Path] = set()
    removed_rows: dict[str, int] = {}
    metrics_step = None
    evaluation_step = None
    selector_removed_rows = 0

    try:
        for filename, kind in (
            ("metrics.jsonl", "jsonl"),
            ("eval_history.jsonl", "jsonl"),
            ("train_metrics.csv", "csv"),
            ("eval_metrics.csv", "csv"),
        ):
            path = output_dir / filename
            if kind == "jsonl":
                temporary, retained_step, removed = _stage_jsonl_rewind(
                    path, resume_step, "step"
                )
            else:
                temporary, retained_step, removed = _stage_csv_rewind(path, resume_step)
            if filename == "metrics.jsonl":
                metrics_step = retained_step
            elif filename == "eval_history.jsonl":
                evaluation_step = retained_step
            if temporary is not None:
                replacements.append((temporary, path))
            if removed:
                removed_rows[filename] = removed

        selector_root = output_dir / "selector_scores"
        if selector_root.is_dir():
            for path in sorted(selector_root.glob("selected_steps_*.jsonl.gz")):
                if _SELECTOR_CHUNK_PATTERN.match(path.name) is None:
                    continue
                temporary, delete_path, removed = _stage_selector_rewind(
                    path, resume_step
                )
                if temporary is not None:
                    replacements.append((temporary, path))
                if delete_path:
                    deletions.add(path)
                selector_removed_rows += removed

        deletions.update(
            _step_paths_after(
                output_dir / "token_score_stats", _STEP_JSON_PATTERN, resume_step
            )
        )
        deletions.update(
            _step_paths_after(
                output_dir / "training_eval", _STEP_DIRECTORY_PATTERN, resume_step
            )
        )
        deletions.update(
            _step_paths_after(output_dir, _CHECKPOINT_PATTERN, resume_step)
        )

        checkpoint = (
            Path(resume_checkpoint).expanduser().resolve()
            if resume_checkpoint is not None
            else None
        )
        final_checkpoint = output_dir / "final"
        if final_checkpoint.is_dir() and checkpoint != final_checkpoint.resolve():
            deletions.add(final_checkpoint)
        summary = output_dir / "summary.json"
        if summary.is_file():
            deletions.add(summary)

        if checkpoint is not None:
            try:
                checkpoint_value = str(checkpoint.relative_to(output_dir))
            except ValueError:
                checkpoint_value = str(checkpoint)
            latest = output_dir / "latest.json"
            latest_temporary = _temporary_sibling(latest)
            latest_temporary.write_text(
                json.dumps(
                    {
                        "step": resume_step,
                        "checkpoint": checkpoint_value,
                        "final": checkpoint.name == "final",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            replacements.append((latest_temporary, latest))

        for temporary, destination in replacements:
            temporary.replace(destination)
        for path in sorted(deletions, key=lambda item: len(item.parts), reverse=True):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    except BaseException:
        for temporary, _ in replacements:
            temporary.unlink(missing_ok=True)
        raise

    removed_paths = [str(path.relative_to(output_dir)) for path in sorted(deletions)]
    return {
        "metrics_last_step": metrics_step,
        "evaluation_last_step": evaluation_step,
        "selector_logs_checked": True,
        "rewound": bool(removed_rows or selector_removed_rows or removed_paths),
        "resume_step": resume_step,
        "removed_rows": removed_rows,
        "selector_rows_removed": selector_removed_rows,
        "removed_paths": removed_paths,
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
        "rollout.num_responses",
        "rollout.max_new_tokens",
        "rollout.temperature",
        "rollout.top_p",
        "rollout.seed",
        "selector.top_k",
        "opd.adv_estimator",
        "opd.top_k_strategy",
        "opd.reward_weight_mode",
        "opd.loss_agg_mode",
        "opd.teacher_temperature",
        "selector.rac_gamma",
        "selector.rac_w_min",
        "selector.rac_beta",
        "token_budget.rho",
        "training.learning_rate",
        "training.adam_betas",
        "training.weight_decay",
        "training.ppo_clip_low",
        "training.ppo_clip_high",
        "training.ppo_dual_clip",
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
