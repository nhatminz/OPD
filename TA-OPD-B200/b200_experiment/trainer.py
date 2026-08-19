from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from .config import save_config
from .data import epoch_batch_indices, read_records, stable_sample_id, tokenize_prompts
from .diagnostics import correlations, finite_or_raise, selector_summary
from .distributed import (
    DistributedContext,
    contiguous_partition,
    initialize_distributed,
    unique_free_port,
    unwrap_model,
)
from .evaluation import BENCHMARK_ORDER, evaluate_loaded_suite
from .eval_schedule import (
    should_run_training_evaluation,
    training_evaluation_steps,
)
from .metadata import collect_metadata, save_metadata
from .models import load_models
from .resume import (
    resolve_resume_checkpoint,
    restore_optimizer,
    validate_append_history,
    validate_resume_config,
)
from .scoring import (
    HFBranchProber,
    cuda_sync,
    generate_on_policy,
    score_original_rollout,
)
from .selector_logging import SelectedTokenLogger
from .selectors import RACSelector, TASelector, top_budget_mask
from .selectors.base import SelectorOutput, robust_quantile_normalize, scatter_valid
from .vllm_rollout import VLLMRolloutEngine


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _timed(device: torch.device, function, *args, **kwargs):
    cuda_sync(device)
    started = time.perf_counter()
    result = function(*args, **kwargs)
    cuda_sync(device)
    return result, time.perf_counter() - started


def _globalize_ta_output(
    local: SelectorOutput,
    valid_mask: torch.Tensor,
    selector: TASelector,
    distributed: DistributedContext,
) -> tuple[SelectorOutput, dict[str, torch.Tensor], int, int]:
    """Apply TA's quantile normalization over the true global rollout batch."""
    global_d, start, end, lengths = distributed.all_gather_variable_1d(
        local.diagnostics["D"][valid_mask]
    )
    global_c, c_start, c_end, c_lengths = distributed.all_gather_variable_1d(
        local.diagnostics["C"][valid_mask]
    )
    if (start, end, lengths) != (c_start, c_end, c_lengths):
        raise AssertionError("Distributed TA D/C layouts differ")
    global_d_norm = robust_quantile_normalize(
        global_d, selector.q_low, selector.q_high, selector.eps
    )
    global_c_norm = robust_quantile_normalize(
        global_c, selector.q_low, selector.q_high, selector.eps
    )
    global_score = global_d_norm * global_c_norm
    diagnostics = dict(local.diagnostics)
    diagnostics.update(
        D_norm=scatter_valid(global_d_norm[start:end], valid_mask),
        C_norm=scatter_valid(global_c_norm[start:end], valid_mask),
        s_TA=scatter_valid(global_score[start:end], valid_mask),
    )
    global_diagnostics = {
        "D": global_d,
        "C": global_c,
        "D_norm": global_d_norm,
        "C_norm": global_c_norm,
        "s_TA": global_score,
    }
    return (
        SelectorOutput(diagnostics["s_TA"], diagnostics),
        global_diagnostics,
        start,
        end,
    )


def _gather_selector_diagnostics(
    diagnostics: dict[str, Any],
    valid_mask: torch.Tensor,
    keys: tuple[str, ...],
    distributed: DistributedContext,
) -> tuple[dict[str, torch.Tensor], int, int]:
    gathered: dict[str, torch.Tensor] = {}
    layout: tuple[int, int, tuple[int, ...]] | None = None
    for key in keys:
        value = diagnostics.get(key)
        if not torch.is_tensor(value) or value.shape != valid_mask.shape:
            continue
        combined, start, end, lengths = distributed.all_gather_variable_1d(
            value[valid_mask]
        )
        current_layout = (start, end, lengths)
        if layout is None:
            layout = current_layout
        elif layout != current_layout:
            raise AssertionError(f"Distributed selector layout differs for {key}")
        gathered[key] = combined
    if layout is None:
        raise ValueError("No token-shaped selector diagnostics were available")
    return gathered, layout[0], layout[1]


def _local_mask_from_global_budget(
    global_scores: torch.Tensor,
    local_valid_mask: torch.Tensor,
    start: int,
    end: int,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_valid = torch.ones_like(global_scores, dtype=torch.bool)
    global_selected = top_budget_mask(global_scores, global_valid, rho)
    local_selected = torch.zeros_like(local_valid_mask, dtype=torch.bool)
    local_selected[local_valid_mask] = global_selected[start:end]
    return local_selected, global_selected


def _rollout_hash(
    response_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    distributed: DistributedContext,
) -> str:
    serialized = []
    for row_ids, row_valid in zip(response_ids, valid_mask):
        tokens = row_ids[row_valid].long()
        serialized.append(
            torch.cat(
                (
                    torch.tensor([tokens.numel()], device=tokens.device),
                    tokens,
                )
            )
        )
    local = (
        torch.cat(serialized)
        if serialized
        else torch.empty(0, dtype=torch.long, device=distributed.device)
    )
    combined, _, _, _ = distributed.all_gather_variable_1d(local)
    return hashlib.sha256(combined.detach().cpu().numpy().tobytes()).hexdigest()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=True) + "\n")


def _save_inference_snapshot(model, tokenizer, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    previous_use_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        model.save_pretrained(destination, safe_serialization=True)
    finally:
        model.config.use_cache = previous_use_cache
    tokenizer.save_pretrained(destination)


def _evaluate_vllm_subprocess(
    model,
    tokenizer,
    model_name: str,
    model_path: Path | None,
    step: int,
    config: dict[str, Any],
    resolved_config_path: Path,
    output_dir: Path,
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    if hasattr(model, "peft_config"):
        raise RuntimeError(
            "vLLM periodic evaluation currently requires full-parameter training; "
            "set training_evaluation.backend=hf when training.use_lora=true"
        )

    temporary_snapshot: tempfile.TemporaryDirectory[str] | None = None
    if model_path is None:
        snapshot_root = output_dir.parent / ".snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        temporary_snapshot = tempfile.TemporaryDirectory(
            prefix=f"step-{step:06d}-", dir=snapshot_root
        )
        model_path = Path(temporary_snapshot.name)
        _save_inference_snapshot(model, tokenizer, model_path)

    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["VLLM_LOGGING_LEVEL"] = environment.get("VLLM_LOGGING_LEVEL", "WARNING")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repo_root), environment.get("PYTHONPATH", "")))
    )
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
    ):
        environment.pop(name, None)
    command = [
        sys.executable,
        "-m",
        "b200_experiment.vllm_evaluation",
        "--config",
        str(resolved_config_path.resolve()),
        "--model",
        str(model_path.resolve()),
        "--name",
        model_name,
        "--output",
        str(output_dir.resolve()),
        "--settings-json",
        json.dumps(runtime_settings),
    ]
    torch.cuda.empty_cache()
    try:
        subprocess.run(command, cwd=repo_root, env=environment, check=True)
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    finally:
        if temporary_snapshot is not None:
            temporary_snapshot.cleanup()


def _run_training_evaluation(
    model,
    tokenizer,
    method: str,
    step: int,
    max_steps: int,
    config: dict[str, Any],
    output_dir: Path,
    resolved_config_path: Path,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    settings = config.get("training_evaluation", {})
    eval_root = output_dir / str(settings.get("output_subdir", "training_eval"))
    step_dir = eval_root / f"step-{step:06d}"
    runtime_settings = {
        "backend": str(settings.get("backend", "vllm")).lower(),
        "batch_size": int(
            settings.get("batch_size", config["evaluation"].get("batch_size", 16))
        ),
        "max_new_tokens": int(
            settings.get(
                "max_new_tokens", config["evaluation"].get("max_new_tokens", 2048)
            )
        ),
        "limit": settings.get("limit", config["evaluation"].get("limit")),
        "benchmark_names": settings.get("benchmark_names", list(BENCHMARK_ORDER)),
        "vllm": settings.get("vllm", {}),
    }
    started = time.perf_counter()
    model_name = "Base student" if step == 0 else f"{method.upper()}-OPD step {step}"
    backend = runtime_settings["backend"]
    if backend == "vllm":
        source_path = (
            Path(config["models"]["student_path"]).resolve()
            if step == 0
            else checkpoint
        )
        suite = _evaluate_vllm_subprocess(
            model,
            tokenizer,
            model_name,
            source_path,
            step,
            config,
            resolved_config_path,
            step_dir,
            runtime_settings,
        )
    elif backend == "hf":
        suite = evaluate_loaded_suite(
            model,
            tokenizer,
            model_name,
            config,
            step_dir,
            runtime_settings=runtime_settings,
        )
    else:
        raise ValueError("training_evaluation.backend must be 'vllm' or 'hf'")
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    history_entry = {
        "step": step,
        "max_steps": max_steps,
        "method": method,
        "model_role": "base_student" if step == 0 else method,
        "backend": backend,
        "evaluation_time": elapsed,
        "benchmarks": {
            name: {
                "correct": result["correct"],
                "total": result["total"],
                "accuracy": result["accuracy"],
            }
            for name, result in suite["benchmarks"].items()
        },
        "parameters": suite["parameters"],
        "details": str((step_dir / "summary.json").resolve()),
    }
    _append_jsonl(output_dir / "eval_history.jsonl", history_entry)
    return history_entry


def _make_optimizer(parameters, training: dict[str, Any]):
    kwargs = dict(
        lr=float(training.get("learning_rate", 1e-5)),
        betas=tuple(training.get("adam_betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    if bool(training.get("fused_optimizer", True)):
        try:
            return torch.optim.AdamW(parameters, fused=True, **kwargs), True
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(parameters, **kwargs), False


def _opd_train_step(
    model,
    optimizer,
    rollout,
    selected_mask,
    old_student_log_probs,
    teacher_log_probs,
    config,
    device,
    distributed: DistributedContext,
    global_batch_size: int,
):
    training = config["training"]
    global_micro_batch = max(
        1, int(training.get("micro_batch_size", global_batch_size))
    )
    micro_batch = max(1, math.ceil(global_micro_batch / distributed.world_size))
    local_batch_size = rollout.input_ids.shape[0]
    if local_batch_size <= 0 or global_batch_size <= 0:
        raise ValueError("OPD train step received an empty local/global batch")
    eps_low, eps_high = (
        float(training.get("ppo_clip_low", 0.2)),
        float(training.get("ppo_clip_high", 0.28)),
    )
    advantage = (teacher_log_probs - old_student_log_probs).detach()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    unwrap_model(model).config.use_cache = False
    forward_seconds = backward_seconds = loss_value = 0.0
    begins = list(range(0, local_batch_size, micro_batch))
    for chunk_index, begin in enumerate(begins):
        end = min(begin + micro_batch, local_batch_size)
        synchronize = chunk_index == len(begins) - 1
        sync_context = (
            nullcontext()
            if synchronize or not isinstance(model, DistributedDataParallel)
            else model.no_sync()
        )
        with sync_context:
            cuda_sync(device)
            started = time.perf_counter()
            output = model(
                input_ids=rollout.input_ids[begin:end],
                attention_mask=rollout.attention_mask[begin:end],
                use_cache=False,
                return_dict=True,
            )
            width, start = rollout.response_ids.shape[1], rollout.prompt_width - 1
            logits = output.logits[:, start : start + width]
            labels = rollout.response_ids[begin:end]
            current = (
                -F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                )
                .view_as(labels)
                .float()
            )
            old, adv = old_student_log_probs[begin:end], advantage[begin:end]
            ratio = torch.exp((current - old).clamp(-20.0, 20.0))
            elementwise = torch.maximum(
                -ratio * adv, -ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * adv
            )
            masks = selected_mask[begin:end].float()
            # DDP averages gradients across ranks. Multiplying each local sum
            # by world/global_batch therefore reproduces the original global
            # per-sample mean even for an uneven final batch.
            loss = (
                (elementwise * masks).sum(-1) / masks.sum(-1).clamp_min(1.0)
            ).sum() * (distributed.world_size / global_batch_size)
            cuda_sync(device)
            forward_seconds += time.perf_counter() - started
            finite_or_raise("OPD loss", loss.detach().reshape(1))
            cuda_sync(device)
            started = time.perf_counter()
            loss.backward()
            cuda_sync(device)
            backward_seconds += time.perf_counter() - started
            loss_value += float(loss.detach().item())
            del output, logits, current, ratio, elementwise, loss
    cuda_sync(device)
    started = time.perf_counter()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(training.get("max_grad_norm", 1.0)),
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    cuda_sync(device)
    backward_seconds += time.perf_counter() - started
    global_loss = distributed.sum_float(loss_value) / distributed.world_size
    return {
        "loss": global_loss,
        "gradient_norm": float(gradient_norm.detach().item()),
        "training_forward_time": forward_seconds,
        "backward_update_time": backward_seconds,
    }


def _save_checkpoint(
    model,
    tokenizer,
    optimizer,
    output_dir: Path,
    step: int,
    final: bool,
    save_optimizer: bool,
):
    checkpoint = output_dir / ("final" if final else f"checkpoint-{step:06d}")
    _save_inference_snapshot(model, tokenizer, checkpoint)
    if save_optimizer:
        torch.save(
            {"step": step, "optimizer": optimizer.state_dict()},
            checkpoint / "optimizer.pt",
        )
    return checkpoint


def run_training(
    config: dict[str, Any], command_line: list[str] | None = None
) -> dict[str, Any]:
    distributed_cfg = config.get("distributed", {})
    distributed = initialize_distributed(distributed_cfg.get("backend", "nccl"))
    device = distributed.device
    if (
        bool(config["experiment"].get("require_b200", True))
        and "B200" not in torch.cuda.get_device_name(device).upper()
    ):
        raise RuntimeError(
            f"This config requires NVIDIA B200; detected {torch.cuda.get_device_name(device)!r}"
        )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    experiment, training = config["experiment"], config["training"]
    method = str(experiment["method"]).lower()
    if method not in {"ta", "rac"}:
        raise ValueError(f"Training method must be ta or rac, got {method!r}")
    seed = int(experiment.get("seed", 1234))
    seed_everything(seed)
    resume_checkpoint = resolve_resume_checkpoint(
        training.get("resume_from_checkpoint")
    )
    resume_config_validation = (
        validate_resume_config(
            resume_checkpoint,
            config,
            allow_mismatch=bool(training.get("resume_allow_config_mismatch", False)),
        )
        if resume_checkpoint is not None
        else None
    )
    output_dir = Path(experiment["output_dir"]).resolve()
    metrics_path = output_dir / "metrics.jsonl"
    if (
        resume_checkpoint is None
        and metrics_path.exists()
        and not bool(experiment.get("allow_existing_output", False))
    ):
        raise FileExistsError(f"Refusing to append to existing run: {metrics_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / (
        "resolved_config.yaml"
        if resume_checkpoint is None
        else f"resolved_config.resume-{resume_checkpoint.name}.yaml"
    )
    if distributed.is_main:
        save_config(config, resolved_config_path)
        canonical_config = output_dir / "resolved_config.yaml"
        if resume_checkpoint is not None and not canonical_config.exists():
            save_config(config, canonical_config)
    distributed.barrier()

    rollout_backend = str(config["rollout"].get("backend", "vllm")).lower()
    if rollout_backend not in {"vllm", "hf"}:
        raise ValueError("rollout.backend must be 'vllm' or 'hf'")
    if rollout_backend == "vllm" and bool(training.get("use_lora", False)):
        raise RuntimeError(
            "vLLM CUDA-IPC rollout requires full-parameter training; "
            "set training.use_lora=false or rollout.backend=hf"
        )
    rollout_engine: VLLMRolloutEngine | None = None

    setup_progress = tqdm(
        total=4,
        desc=f"Setup {method.upper()}-OPD",
        unit="stage",
        dynamic_ncols=True,
        leave=False,
        disable=not distributed.is_main,
    )
    setup_progress.set_postfix_str("stage=load-data", refresh=True)
    records, data_files = read_records(
        config["data"]["path"], split=config["data"].get("split")
    )
    if not records:
        raise ValueError("Full DAPO dataset is empty")
    setup_progress.update(1)
    setup_progress.set_postfix_str(
        f"stage=start-rollout-{rollout_backend}", refresh=True
    )
    if rollout_backend == "vllm":
        rollout_engine = VLLMRolloutEngine(
            config,
            output_dir,
            local_rank=distributed.local_rank,
            world_size=distributed.world_size,
            port=unique_free_port(distributed),
        )
        rollout_engine.start()
    setup_progress.update(1)
    setup_progress.set_postfix_str("stage=load-models", refresh=True)
    model_load_config = config
    if resume_checkpoint is not None:
        model_load_config = copy.deepcopy(config)
        model_load_config["models"]["student_path"] = str(resume_checkpoint)
    student, teacher, tokenizer, model_metadata = load_models(model_load_config, device)
    training_student = student
    if distributed.enabled:
        training_student = DistributedDataParallel(
            student,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=bool(
                distributed_cfg.get("find_unused_parameters", False)
            ),
            gradient_as_bucket_view=bool(
                distributed_cfg.get("gradient_as_bucket_view", True)
            ),
            static_graph=bool(distributed_cfg.get("static_graph", True)),
            bucket_cap_mb=float(distributed_cfg.get("bucket_cap_mb", 100)),
        )
    setup_progress.update(1)
    setup_progress.set_postfix_str("stage=optimizer", refresh=True)
    optimizer, fused_optimizer = _make_optimizer(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        training,
    )
    resume_state = (
        restore_optimizer(optimizer, resume_checkpoint, device)
        if resume_checkpoint is not None
        else None
    )
    resume_step = resume_state.step if resume_state is not None else 0
    resume_history = (
        validate_append_history(output_dir, resume_step)
        if resume_state is not None
        else None
    )
    if distributed.is_main:
        metadata = collect_metadata(
            Path(__file__).resolve().parents[1],
            command_line or sys.argv,
            model_metadata,
            config["data"]["path"],
            data_files,
        )
        metadata["data_schema"] = {
            "rows": len(records),
            "columns": sorted(records[0]),
            "files": [str(path) for path in data_files],
            "split": config["data"].get("split"),
            "full_dataset": True,
        }
        metadata["distributed"] = {
            "strategy": "ddp" if distributed.enabled else "single_process",
            "world_size": distributed.world_size,
            "global_batch_preserved": True,
            "global_ta_normalization": True,
            "global_token_budget": True,
        }
        metadata["resume"] = (
            {
                "checkpoint": str(resume_state.checkpoint),
                "optimizer_path": str(resume_state.optimizer_path),
                "step": resume_state.step,
                "history": resume_history,
                "config_validation": resume_config_validation,
                "source_world_size": "unknown_for_legacy_checkpoint",
            }
            if resume_state is not None
            else None
        )
        metadata_filename = (
            "run_metadata.json"
            if resume_state is None
            else f"run_metadata.resume-step-{resume_step:06d}.json"
        )
        save_metadata(metadata, output_dir, filename=metadata_filename)
    setup_progress.update(1)
    setup_progress.close()
    selector_cfg = config["selector"]
    ta_selector = TASelector(
        int(selector_cfg.get("top_k", 16)),
        float(selector_cfg.get("q_low", 0.05)),
        float(selector_cfg.get("q_high", 0.95)),
    )
    rac_selector = RACSelector(
        int(selector_cfg.get("top_k", 16)),
        int(selector_cfg.get("branch_m", 2)),
        float(selector_cfg.get("eps", 1e-8)),
        selector_cfg.get("rac_delta_mode", "full_vocab"),
    )
    batch_size = int(config["rollout"]["batch_size"])
    steps_per_epoch = math.ceil(len(records) / batch_size)
    configured_max_steps = training.get("max_steps")
    max_steps = (
        int(configured_max_steps)
        if configured_max_steps is not None
        else int(training.get("epochs", 1)) * steps_per_epoch
    )
    if resume_step >= max_steps:
        raise ValueError(
            f"Checkpoint is already at step {resume_step}, but configured total "
            f"max_steps is {max_steps}. Set MAX_STEPS above {resume_step} or "
            "increase EPOCHS. MAX_STEPS is the total target, not extra steps."
        )
    rho = float(config["token_budget"]["rho"])
    token_logger = SelectedTokenLogger(
        output_dir,
        tokenizer,
        method,
        chunk_steps=int(config.get("logging", {}).get("selector_chunk_steps", 50)),
        enabled=bool(config.get("logging", {}).get("selected_tokens_enabled", True)),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    training_eval_settings = config.get("training_evaluation", {})
    evaluation_steps = training_evaluation_steps(max_steps, training_eval_settings)
    if evaluation_steps and distributed.is_main:
        tqdm.write(
            f"Periodic evaluation ({training_eval_settings.get('backend', 'vllm')}): "
            + ", ".join(map(str, evaluation_steps))
        )
    if resume_state is not None and distributed.is_main:
        tqdm.write(
            f"Resuming {method.upper()}-OPD from optimizer step {resume_step}: "
            f"{resume_state.checkpoint}"
        )
    initial_evaluation = None
    if resume_step == 0 and should_run_training_evaluation(
        0, max_steps, training_eval_settings
    ):
        distributed.barrier()
        if distributed.is_main:
            tqdm.write("Evaluating the untouched base student at optimizer step 0...")
            initial_evaluation = _run_training_evaluation(
                student,
                tokenizer,
                method,
                0,
                max_steps,
                config,
                output_dir,
                resolved_config_path,
            )
        distributed.barrier()
    final_metrics: dict[str, Any] = {}
    progress = tqdm(
        range(resume_step, max_steps),
        desc=f"{method.upper()}-OPD B200",
        unit="step",
        dynamic_ncols=True,
        leave=True,
        disable=not distributed.is_main,
        mininterval=0.5,
        initial=resume_step,
        total=max_steps,
    )
    for step_index in progress:
        step, step_started = step_index + 1, time.perf_counter()
        if distributed.is_main:
            progress.set_postfix_str(f"stage=rollout-{rollout_backend}", refresh=True)
        torch.cuda.reset_peak_memory_stats(device)
        global_indices = epoch_batch_indices(len(records), batch_size, step_index, seed)
        if len(global_indices) < distributed.world_size:
            raise ValueError(
                f"Global batch has {len(global_indices)} samples but WORLD_SIZE is "
                f"{distributed.world_size}; every DDP worker needs at least one sample"
            )
        local_start, local_end = contiguous_partition(
            len(global_indices), distributed.rank, distributed.world_size
        )
        indices = global_indices[local_start:local_end]
        batch_records = [records[index] for index in indices]
        encoded, _ = tokenize_prompts(batch_records, tokenizer, config["data"], device)
        rollout_function = (
            rollout_engine.generate
            if rollout_engine is not None
            else generate_on_policy
        )
        rollout, rollout_time = _timed(
            device,
            rollout_function,
            student,
            encoded["input_ids"],
            encoded["attention_mask"],
            max_new_tokens=int(config["rollout"].get("max_new_tokens", 256)),
            temperature=float(config["rollout"].get("temperature", 1.0)),
            top_p=float(config["rollout"].get("top_p", 1.0)),
            eos_token_ids=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            seed=int(config["rollout"].get("seed", seed)) + step_index,
            sample_seed_offset=local_start,
        )
        rollout_backend_metrics = (
            dict(rollout_engine.last_metrics) if rollout_engine is not None else {}
        )
        for field in (
            "input_ids",
            "attention_mask",
            "response_ids",
            "valid_mask",
            "rollout_log_probs",
        ):
            setattr(rollout, field, getattr(rollout, field).clone())
        original_rollout = rollout.input_ids.clone()
        rollout_hash = _rollout_hash(
            rollout.response_ids, rollout.valid_mask, distributed
        )
        if distributed.is_main:
            progress.set_postfix_str("stage=score-student", refresh=True)
        student_scores, student_base_time = _timed(
            device, score_original_rollout, student, rollout, True
        )
        if distributed.is_main:
            progress.set_postfix_str("stage=score-teacher", refresh=True)
        teacher_scores, teacher_base_time = _timed(
            device, score_original_rollout, teacher, rollout, True
        )
        valid = rollout.valid_mask
        finite_or_raise("student probabilities", student_scores.probabilities[valid])
        finite_or_raise("teacher probabilities", teacher_scores.probabilities[valid])
        ta_raw, ta_raw_time = _timed(
            device,
            ta_selector.compute_scores,
            student_scores.probabilities,
            teacher_scores.probabilities,
            valid,
            normalize=False,
        )
        ta_globalized, ta_normalization_time = _timed(
            device,
            _globalize_ta_output,
            ta_raw,
            valid,
            ta_selector,
            distributed,
        )
        ta_output, global_ta_diagnostics, ta_start, ta_end = ta_globalized
        ta_time = ta_raw_time + ta_normalization_time
        cross_diagnostics, prober = {}, None
        if method == "rac":
            if distributed.is_main:
                progress.set_postfix_str("stage=selector-RAC", refresh=True)
            prober = HFBranchProber(
                student,
                teacher,
                rollout,
                student_scores.cache,
                teacher_scores.cache,
                top_k=int(selector_cfg.get("top_k", 16)),
                mode=selector_cfg.get("rac_probe_mode", "kv_cache"),
                chunk_size=int(selector_cfg.get("rac_branch_chunk_size", 256)),
            )
            if step_index == 0 and bool(selector_cfg.get("validate_kv_cache", True)):
                cache_error = prober.validate_kv_equivalence()
                cache_error = distributed.max_float(cache_error)
                tolerance = float(selector_cfg.get("kv_cache_tolerance", 0.04))
                if cache_error > tolerance:
                    raise AssertionError(
                        f"KV/full-prefix C+ mismatch {cache_error} > {tolerance}"
                    )
                cross_diagnostics["kv_cache_cplus_max_abs_error"] = cache_error
            primary, selector_time = _timed(
                device,
                rac_selector.compute_scores,
                student_scores.probabilities,
                teacher_scores.probabilities,
                valid,
                prober,
                student_topk=(
                    ta_output.diagnostics["_student_top_values"],
                    ta_output.diagnostics["_student_top_ids"],
                ),
            )
            rac_keys = (
                "s_RAC",
                "Delta",
                "A",
                "F",
                "B",
                "positive_reachable_candidates",
                "evaluated_branches",
                "mean_Cplus",
            )
            global_primary_diagnostics, primary_start, primary_end = (
                _gather_selector_diagnostics(
                    primary.diagnostics, valid, rac_keys, distributed
                )
            )
            if (primary_start, primary_end) != (ta_start, ta_end):
                raise AssertionError("TA/RAC distributed token layouts differ")
            global_valid_mask = torch.ones_like(
                global_ta_diagnostics["s_TA"], dtype=torch.bool
            )
            cross_diagnostics.update(
                correlations(
                    global_ta_diagnostics["s_TA"],
                    global_primary_diagnostics,
                    global_valid_mask,
                )
            )
            rac_probe_time, rac_branches = (
                prober.counterfactual_seconds,
                prober.evaluated_branches,
            )
        else:
            if distributed.is_main:
                progress.set_postfix_str("stage=selector-TA", refresh=True)
            primary, selector_time, rac_probe_time, rac_branches = (
                ta_output,
                ta_time,
                0.0,
                0,
            )
            global_primary_diagnostics = global_ta_diagnostics
            primary_start, primary_end = ta_start, ta_end
        finite_or_raise(f"{method} selector", primary.scores[valid])
        score_key = "s_RAC" if method == "rac" else "s_TA"
        selected, global_selected = _local_mask_from_global_budget(
            global_primary_diagnostics[score_key],
            valid,
            primary_start,
            primary_end,
            rho,
        )
        expected = math.ceil(rho * global_primary_diagnostics[score_key].numel())
        if distributed.sum_int(int(selected.sum().item())) != expected:
            raise AssertionError("TA/RAC shared budget count is incorrect")
        if not torch.equal(original_rollout, rollout.input_ids):
            raise AssertionError("Selector changed the original rollout")
        sample_ids = [
            stable_sample_id(record, index)
            for record, index in zip(batch_records, indices)
        ]
        logged_selected = token_logger.write(
            step=step,
            dataset_indices=indices,
            sample_ids=sample_ids,
            response_ids=rollout.response_ids,
            selected_mask=selected,
            diagnostics=primary.diagnostics,
            batch_index_offset=local_start,
        )
        global_logged_selected = distributed.sum_int(logged_selected)
        if (
            bool(config.get("logging", {}).get("selected_tokens_enabled", True))
            and global_logged_selected != expected
        ):
            raise AssertionError(
                f"Detailed selector logs wrote {global_logged_selected}, "
                f"expected {expected}"
            )

        old_student = student_scores.sampled_log_probs.detach().clone()
        teacher_log_probs = teacher_scores.sampled_log_probs.detach().clone()
        del student_scores, teacher_scores, prober
        # Let PyTorch reuse the released probability/cache blocks for backward.
        # Emptying the CUDA allocator every step is materially slower on B200.
        if bool(training.get("empty_cuda_cache_each_step", False)):
            torch.cuda.empty_cache()
        if distributed.is_main:
            progress.set_postfix_str("stage=train", refresh=True)
        train_metrics = _opd_train_step(
            training_student,
            optimizer,
            rollout,
            selected,
            old_student,
            teacher_log_probs,
            config,
            device,
            distributed,
            len(global_indices),
        )
        checkpoint = None
        save_checkpoints = bool(training.get("save_checkpoints", True))
        save_interval = int(training.get("save_interval", 100))
        if (
            distributed.is_main
            and save_checkpoints
            and (step == max_steps or (save_interval > 0 and step % save_interval == 0))
        ):
            progress.set_postfix_str("stage=checkpoint", refresh=True)
            checkpoint = _save_checkpoint(
                student,
                tokenizer,
                optimizer,
                output_dir,
                step,
                step == max_steps,
                bool(training.get("save_optimizer", True)),
            )
        distributed.barrier()
        cuda_sync(device)
        local_wall_time = time.perf_counter() - step_started
        local_valid_tokens = int(valid.sum().item())
        local_peak_allocated = torch.cuda.max_memory_allocated(device)
        local_peak_reserved = torch.cuda.max_memory_reserved(device)
        wall_time = distributed.max_float(local_wall_time)
        valid_tokens = distributed.sum_int(local_valid_tokens)
        peak_allocated = distributed.max_int(local_peak_allocated)
        peak_reserved = distributed.max_int(local_peak_reserved)
        peak_allocated_total = distributed.sum_int(local_peak_allocated)
        peak_reserved_total = distributed.sum_int(local_peak_reserved)
        aggregated_train_metrics = {
            "loss": train_metrics["loss"],
            "gradient_norm": distributed.max_float(train_metrics["gradient_norm"]),
            "training_forward_time": distributed.max_float(
                train_metrics["training_forward_time"]
            ),
            "backward_update_time": distributed.max_float(
                train_metrics["backward_update_time"]
            ),
        }
        final_metrics = {
            "step": step,
            "epoch": step_index // steps_per_epoch,
            "step_in_epoch": step_index % steps_per_epoch,
            "method": method,
            "resumed_from": (
                str(resume_state.checkpoint) if resume_state is not None else None
            ),
            "resume_step": resume_step,
            "batch_size": len(global_indices),
            "local_batch_size_rank0": len(global_indices) // distributed.world_size
            + int(0 < len(global_indices) % distributed.world_size),
            "configured_batch_size": batch_size,
            "micro_batch_size": int(training.get("micro_batch_size", batch_size)),
            "local_micro_batch_size": math.ceil(
                int(training.get("micro_batch_size", batch_size))
                / distributed.world_size
            ),
            "distributed_world_size": distributed.world_size,
            "distributed_strategy": (
                "ddp" if distributed.enabled else "single_process"
            ),
            "global_batch_preserved": True,
            "global_ta_normalization": True,
            "global_token_budget": True,
            "rac_reused_ta_student_topk": method == "rac",
            "fused_optimizer": fused_optimizer,
            **aggregated_train_metrics,
            "rollout_backend": rollout_backend,
            "rollout_time": distributed.max_float(rollout_time),
            "vllm_weight_sync_time": distributed.max_float(
                rollout_backend_metrics.get("weight_sync_time", 0.0)
            ),
            "vllm_generation_time": distributed.max_float(
                rollout_backend_metrics.get("generation_time", 0.0)
            ),
            "vllm_sleep_time": distributed.max_float(
                rollout_backend_metrics.get("sleep_time", 0.0)
            ),
            "vllm_torch_cache_released": distributed.any(
                bool(rollout_backend_metrics.get("torch_cache_released", 0.0))
            ),
            "vllm_torch_cache_release_time": distributed.max_float(
                rollout_backend_metrics.get("torch_cache_release_time", 0.0)
            ),
            "student_base_scoring_time": distributed.max_float(student_base_time),
            "teacher_base_scoring_time": distributed.max_float(teacher_base_time),
            "ta_diagnostic_time": distributed.max_float(ta_time),
            "selector_time": distributed.max_float(selector_time),
            "rac_counterfactual_time": distributed.max_float(rac_probe_time),
            "rac_evaluated_branches": distributed.sum_int(rac_branches),
            "wall_clock_step_time": wall_time,
            "tokens_per_second": valid_tokens / max(wall_time, 1e-12),
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "peak_gpu_allocated_bytes_all_workers": peak_allocated_total,
            "peak_gpu_reserved_bytes_all_workers": peak_reserved_total,
            "selector": selector_summary(
                method,
                global_primary_diagnostics,
                torch.ones_like(
                    global_primary_diagnostics[score_key], dtype=torch.bool
                ),
                global_selected,
            ),
            "cross_selector": cross_diagnostics,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "rollout_token_sha256": rollout_hash,
        }
        # The rollout server is already sleeping; release tensors before a
        # possible periodic-evaluation subprocess reserves its KV cache.
        del (
            encoded,
            rollout,
            original_rollout,
            old_student,
            teacher_log_probs,
            selected,
            primary,
            ta_raw,
            ta_output,
            global_primary_diagnostics,
            global_ta_diagnostics,
            global_selected,
            valid,
            rollout_backend_metrics,
        )
        if should_run_training_evaluation(step, max_steps, training_eval_settings):
            distributed.barrier()
            if distributed.is_main:
                progress.set_postfix_str("stage=evaluation", refresh=True)
                periodic_evaluation = _run_training_evaluation(
                    student,
                    tokenizer,
                    method,
                    step,
                    max_steps,
                    config,
                    output_dir,
                    resolved_config_path,
                    checkpoint=checkpoint,
                )
                final_metrics["periodic_evaluation"] = {
                    "evaluation_time": periodic_evaluation["evaluation_time"],
                    "benchmarks": periodic_evaluation["benchmarks"],
                    "details": periodic_evaluation["details"],
                }
                final_metrics["wall_clock_step_plus_eval_time"] = (
                    wall_time + periodic_evaluation["evaluation_time"]
                )
            distributed.barrier()
        if distributed.is_main:
            _append_jsonl(metrics_path, final_metrics)
            progress.set_postfix(
                loss=f"{train_metrics['loss']:.4f}",
                selected=f"{expected}/{valid_tokens}",
                selector=f"{final_metrics['selector_time']:.2f}s",
                gpu=f"{final_metrics['peak_gpu_allocated_bytes'] / 2**30:.1f}GiB",
                refresh=True,
            )
        if distributed.is_main and bool(experiment.get("verbose_metrics", False)):
            tqdm.write(
                json.dumps(final_metrics, indent=2, ensure_ascii=False, allow_nan=True)
            )
    if rollout_engine is not None:
        rollout_engine.close()
    distributed.barrier()
    summary = {
        "status": "ok",
        "method": method,
        "steps": max_steps,
        "epochs": training.get("epochs"),
        "dataset_rows": len(records),
        "full_dataset": True,
        "rollout_backend": rollout_backend,
        "resumed_from": (
            str(resume_state.checkpoint) if resume_state is not None else None
        ),
        "resume_step": resume_step,
        "distributed_world_size": distributed.world_size,
        "global_batch_preserved": True,
        "vllm_rollout_server_log": str(
            (output_dir / "vllm_rollout_server.log").resolve()
        )
        if rollout_backend == "vllm" and distributed.world_size == 1
        else None,
        "vllm_rollout_server_logs": [
            str(
                (
                    output_dir
                    / (
                        f"vllm_rollout_server.rank-{rank:05d}.log"
                        if distributed.world_size > 1
                        else "vllm_rollout_server.log"
                    )
                ).resolve()
            )
            for rank in range(distributed.world_size)
        ]
        if rollout_backend == "vllm"
        else None,
        "last": final_metrics,
        "student_path": model_metadata["student_path"],
        "teacher_path": model_metadata["teacher_path"],
        "selector_score_dir": str((output_dir / "selector_scores").resolve()),
        "evaluation_history": str((output_dir / "eval_history.jsonl").resolve())
        if bool(training_eval_settings.get("enabled", False))
        else None,
        "initial_evaluation": initial_evaluation,
    }
    if distributed.is_main:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
            handle.write("\n")
    result = (
        summary
        if distributed.is_main
        else {"status": "worker_ok", "rank": distributed.rank}
    )
    distributed.close()
    return result
