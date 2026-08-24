from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from .config import save_config
from .data import (
    epoch_batch_indices,
    expand_prompt_batch,
    read_records,
    stable_sample_id,
    tokenize_prompts,
)
from .diagnostics import correlations, finite_or_raise, selector_summary
from .distributed import (
    BatchLayout,
    DistributedContext,
    batch_layout,
    contiguous_partition,
    initialize_distributed,
    isolate_distributed_subprocess_environment,
    padded_local_indices,
    unique_free_port,
    unwrap_model,
)
from .evaluation import (
    configured_benchmark_names,
    evaluate_loaded_suite,
    evaluation_metric_name,
)
from .evaluation_cache import evaluate_or_reuse_base
from .eval_schedule import (
    should_run_training_evaluation,
    training_evaluation_steps,
)
from .metadata import collect_metadata, save_metadata
from .models import load_models, validate_shared_tokenizer_protocol
from .fsdp import (
    clip_grad_norm,
    distributed_strategy,
    full_model_state_dict,
    full_optimizer_state_dict,
    is_fsdp_model,
    wrap_fsdp_model,
)
from .opd_core import (
    UPSTREAM_ADV_ESTIMATOR,
    UPSTREAM_LOSS_AGG_MODE,
    UPSTREAM_OPD_COMMIT,
    UPSTREAM_REWARD_WEIGHT_MODE,
    UPSTREAM_TOP_K_STRATEGY,
    TopKOPDReference,
    build_topk_opd_reference,
    gather_candidate_log_probs,
    topk_candidate_ppo_loss,
    topk_overlap_fraction,
    weighted_token_sums,
)
from .resume import (
    resolve_resume_checkpoint,
    restore_optimizer,
    validate_append_history,
    validate_resume_config,
)
from .scoring import (
    cuda_sync,
    generate_on_policy,
    score_original_rollout,
    score_student_teacher_rollout,
    supports_response_only_logits,
)
from .selector_logging import SelectedTokenLogger, TokenScoreStatsLogger
from .selectors import OPDSelector, RACSelector, TASelector, top_budget_mask
from .selectors.base import SelectorOutput, robust_quantile_normalize, scatter_valid
from .tensorboard_logging import TensorBoardLogger
from .vllm_rollout import VLLMRolloutEngine


METHOD_DISPLAY_NAMES = {
    "opd": "OPD",
    "ta": "TA-OPD",
    "rac": "Bellman-RAC",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _micro_batch_size_per_gpu(training: dict[str, Any], world_size: int) -> int:
    if "micro_batch_size_per_gpu" in training:
        value = int(training["micro_batch_size_per_gpu"])
    elif "micro_batch_size" in training:
        if world_size > 1:
            raise ValueError(
                "training.micro_batch_size is a legacy ambiguous global value. "
                "Set training.micro_batch_size_per_gpu explicitly for distributed "
                "training."
            )
        value = int(training["micro_batch_size"])
    else:
        raise ValueError("training.micro_batch_size_per_gpu is required")
    if value <= 0:
        raise ValueError("training.micro_batch_size_per_gpu must be positive")
    return value


def _format_batch_layout(
    layout: BatchLayout,
    strategy: str,
    config: dict[str, Any],
) -> str:
    fsdp = config.get("distributed", {}).get("fsdp", {})
    training = config["training"]
    return "\n".join(
        (
            f"distributed strategy          = {strategy}",
            f"world_size                    = {layout.world_size}",
            f"global_prompt_batch_size      = {layout.global_prompt_batch_size}",
            f"local_prompt_batch_size       = {layout.local_prompt_batch_size}",
            f"num_responses                 = {layout.num_responses}",
            f"global_trajectory_batch_size  = {layout.global_trajectory_batch_size}",
            f"local_trajectory_batch_size   = {layout.local_trajectory_batch_size}",
            f"micro_batch_size_per_gpu      = {layout.micro_batch_size_per_gpu}",
            f"micro_batches_per_gpu         = {layout.micro_batches_per_gpu}",
            f"learning_rate                 = {float(training['learning_rate']):.8g}",
            f"max_prompt_tokens             = {int(config['data']['max_prompt_tokens'])}",
            f"max_response_tokens           = {int(config['rollout']['max_new_tokens'])}",
            "student_sharding_strategy     = "
            + ("FULL_SHARD" if strategy == "fsdp" else "none"),
            "teacher_sharding              = "
            + ("FULL_SHARD" if strategy == "fsdp" else "replicated"),
            "teacher_cpu_offload           = "
            + str(bool(fsdp.get("teacher_cpu_offload", False))).lower(),
            "gradient_checkpointing        = "
            + str(bool(training.get("gradient_checkpointing", False))).lower(),
        )
    )


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


def _globalize_opd_output(
    local: SelectorOutput,
    valid_mask: torch.Tensor,
    distributed: DistributedContext,
) -> tuple[SelectorOutput, dict[str, torch.Tensor], int, int]:
    """Gather uniform pure-OPD weights for global metrics and auditing."""
    gathered, start, end = _gather_selector_diagnostics(
        local.diagnostics, valid_mask, ("w",), distributed
    )
    return local, gathered, start, end


def _globalize_rac_output(
    local: SelectorOutput,
    valid_mask: torch.Tensor,
    selector: RACSelector,
    distributed: DistributedContext,
) -> tuple[SelectorOutput, dict[str, torch.Tensor], int, int]:
    """Normalize Bellman V over the true global rollout and build soft weights."""
    keys = ("g", "alignment", "R", "M", "V")
    gathered, start, end = _gather_selector_diagnostics(
        local.diagnostics, valid_mask, keys, distributed
    )
    global_z = robust_quantile_normalize(
        gathered["V"], selector.q_low, selector.q_high, selector.eps
    )
    global_weights = selector.w_min + (1.0 - selector.w_min) * global_z.pow(
        selector.beta
    )
    diagnostics = dict(local.diagnostics)
    diagnostics.update(
        z=scatter_valid(global_z[start:end], valid_mask),
        w=scatter_valid(global_weights[start:end], valid_mask),
    )
    gathered.update(z=global_z, w=global_weights)
    return SelectorOutput(diagnostics["w"], diagnostics), gathered, start, end


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
        if tokens.numel() == 0:
            continue
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


def _append_csv_row(
    path: Path, row: dict[str, Any], fieldnames: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _upsert_jsonl_row(
    path: Path, payload: dict[str, Any], key_fields: tuple[str, ...]
) -> None:
    """Atomically insert/replace a row so a pre-train retry cannot duplicate step 0."""
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

    def matches(row: dict[str, Any]) -> bool:
        return all(row.get(key) == payload.get(key) for key in key_fields)

    output: list[dict[str, Any]] = []
    replaced = False
    for row in rows:
        if matches(row):
            if not replaced:
                output.append(payload)
                replaced = True
        else:
            output.append(row)
    if not replaced:
        output.append(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    os.replace(temporary, path)


def _upsert_csv_row(
    path: Path,
    row: dict[str, Any],
    fieldnames: tuple[str, ...],
    key_fields: tuple[str, ...],
) -> None:
    rows: list[dict[str, Any]] = []
    if path.is_file() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    def matches(item: dict[str, Any]) -> bool:
        return all(str(item.get(key)) == str(row.get(key)) for key in key_fields)

    output: list[dict[str, Any]] = []
    replaced = False
    for existing in rows:
        if matches(existing):
            if not replaced:
                output.append(row)
                replaced = True
        else:
            output.append(existing)
    if not replaced:
        output.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    os.replace(temporary, path)


def _append_train_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    selector = metrics.get("selector", {})
    row = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, (dict, list, tuple))
    }
    for score in ("D", "C", "s_TA", "g", "alignment", "R", "M", "V", "z", "w"):
        for statistic, value in selector.get(score, {}).items():
            row[f"{score}_{statistic}"] = value
    for key in (
        "selected_tokens",
        "selected_fraction",
        "selection_threshold",
        "effective_token_weight_mass",
        "effective_sample_size",
    ):
        row[key] = selector.get(key)
    common = (
        "step",
        "epoch",
        "method",
        "lr",
        "train_loss",
        "weighted_final_loss",
        "base_topk_opd_loss",
        "unweighted_opd_loss",
        "grad_norm",
        "num_valid_tokens",
        "mean_response_length",
        "min_response_length",
        "max_response_length",
        "response_clip_ratio",
        "student_teacher_topk_overlap_ratio",
        "student_teacher_topk_divergence",
        "throughput_tokens_per_sec",
        "rollout_time",
        "total_scoring_time_sec",
        "joint_student_teacher_scoring_time",
        "teacher_score_time_sec",
        "student_cross_topk_scoring_time",
        "ta_local_score_time_sec",
        "bellman_scan_time_sec",
        "forward_backward_time_sec",
        "optimizer_time_sec",
        "peak_gpu_allocated_gb",
        "peak_gpu_reserved_gb",
        "selected_tokens",
        "selected_fraction",
        "selection_threshold",
        "effective_token_weight_mass",
        "effective_sample_size",
    )
    statistics = tuple(
        f"{score}_{statistic}"
        for score in ("D", "C", "s_TA", "g", "alignment", "R", "M", "V", "z", "w")
        for statistic in ("mean", "min", "max", "q05", "q25", "q50", "q75", "q95")
    )
    _append_csv_row(path, row, common + statistics)


def _save_inference_snapshot(
    model,
    tokenizer,
    destination: Path,
    distributed: DistributedContext | None = None,
) -> None:
    """Export a normal HF checkpoint; every FSDP rank enters collectives."""
    if is_fsdp_model(model) and distributed is None:
        raise RuntimeError("FSDP snapshot export requires distributed context")
    raw_model = unwrap_model(model)
    previous_use_cache = raw_model.config.use_cache
    raw_model.config.use_cache = True
    try:
        state_dict = full_model_state_dict(model) if is_fsdp_model(model) else None
        is_main = distributed is None or distributed.is_main
        if is_main:
            destination.mkdir(parents=True, exist_ok=True)
            save_kwargs = {"state_dict": state_dict} if state_dict is not None else {}
            raw_model.save_pretrained(
                destination,
                safe_serialization=True,
                **save_kwargs,
            )
            tokenizer.save_pretrained(destination)
    finally:
        raw_model.config.use_cache = previous_use_cache
    if distributed is not None:
        distributed.barrier()


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
        if is_fsdp_model(model):
            raise RuntimeError(
                "FSDP periodic vLLM evaluation requires a collectively exported "
                "checkpoint/snapshot path"
            )
        snapshot_root = output_dir.parent / ".snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        temporary_snapshot = tempfile.TemporaryDirectory(
            prefix=f"step-{step:06d}-", dir=snapshot_root
        )
        model_path = Path(temporary_snapshot.name)
        _save_inference_snapshot(model, tokenizer, model_path)

    repo_root = Path(__file__).resolve().parents[1]
    environment = isolate_distributed_subprocess_environment()
    environment["VLLM_LOGGING_LEVEL"] = environment.get("VLLM_LOGGING_LEVEL", "WARNING")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repo_root), environment.get("PYTHONPATH", "")))
    )
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
        "temperature": float(
            settings.get("temperature", config["evaluation"].get("temperature", 0.7))
        ),
        "top_p": float(settings.get("top_p", config["evaluation"].get("top_p", 0.95))),
        "num_responses": int(
            settings.get("num_responses", config["evaluation"].get("num_responses", 16))
        ),
        "batch_size": int(
            settings.get("batch_size", config["evaluation"].get("batch_size", 16))
        ),
        "max_new_tokens": int(
            settings.get(
                "max_new_tokens", config["evaluation"].get("max_new_tokens", 2048)
            )
        ),
        "limit": settings.get("limit", config["evaluation"].get("limit")),
        "benchmark_names": list(
            configured_benchmark_names(config, settings.get("benchmark_names"))
        ),
        "vllm": settings.get("vllm", {}),
    }
    started = time.perf_counter()
    method_name = METHOD_DISPLAY_NAMES[method]
    model_name = "Base student" if step == 0 else f"{method_name} step {step}"
    backend = runtime_settings["backend"]
    cache_status = "disabled"
    if backend == "vllm":
        source_path = (
            Path(config["models"]["student_path"]).resolve()
            if step == 0
            else checkpoint
        )

        def evaluate_vllm() -> dict[str, Any]:
            return _evaluate_vllm_subprocess(
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

        if step == 0 and bool(settings.get("reuse_base_evaluation", True)):
            suite, cache_status = evaluate_or_reuse_base(
                config=config,
                runtime_settings=runtime_settings,
                model_path=source_path,
                model_name=model_name,
                destination=step_dir,
                evaluator=evaluate_vllm,
                cache_dir=settings.get("base_cache_dir"),
                reuse_destination=True,
            )
            if cache_status != "generated":
                skipped_responses = sum(
                    int(result["total"]) for result in suite["benchmarks"].values()
                )
                tqdm.write(
                    "Reused untouched-base avg@16 evaluation "
                    f"({cache_status} cache); skipped {skipped_responses:,} generations."
                )
        else:
            suite = evaluate_vllm()
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
    samples_per_problem = int(runtime_settings["num_responses"])
    metric_name = str(
        suite.get("parameters", {}).get(
            "metric", evaluation_metric_name(samples_per_problem)
        )
    )
    history_entry = {
        "step": step,
        "max_steps": max_steps,
        "method": method,
        "model_role": "base_student" if step == 0 else method,
        "backend": backend,
        "evaluation_time": elapsed,
        "base_cache_status": cache_status if step == 0 else None,
        "benchmarks": {
            name: {
                "correct": result["correct"],
                "total": result["total"],
                "accuracy": result["accuracy"],
                "avg_at_n": result.get("avg_at_n", result["accuracy"]),
                **({"avg_at_16": result["avg_at_16"]} if "avg_at_16" in result else {}),
                "problems": result.get("problems"),
                "samples_per_problem": result.get(
                    "samples_per_problem", samples_per_problem
                ),
                "metric": metric_name,
            }
            for name, result in suite["benchmarks"].items()
        },
        "parameters": suite["parameters"],
        "details": str((step_dir / "summary.json").resolve()),
    }
    _upsert_jsonl_row(
        output_dir / "eval_history.jsonl", history_entry, ("step", "method")
    )
    eval_metric_fields = (
        "step",
        "method",
        "backend",
        "benchmark",
        "correct",
        "total",
        "accuracy",
        "avg_at_n",
        "avg_at_16",
        "problems",
        "samples_per_problem",
        "metric",
        "evaluation_time_sec",
    )
    for benchmark, result in history_entry["benchmarks"].items():
        _upsert_csv_row(
            output_dir / "eval_metrics.csv",
            {
                "step": step,
                "method": method,
                "backend": backend,
                "benchmark": benchmark,
                **result,
                "evaluation_time_sec": elapsed,
            },
            eval_metric_fields,
            ("step", "method", "benchmark"),
        )
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
    position_weights,
    opd_reference: TopKOPDReference,
    config,
    device,
    distributed: DistributedContext,
    objective_valid_mask: torch.Tensor | None = None,
):
    training = config["training"]
    micro_batch = _micro_batch_size_per_gpu(training, distributed.world_size)
    local_batch_size = rollout.input_ids.shape[0]
    if local_batch_size <= 0:
        raise ValueError("OPD train step received an empty local/global batch")
    eps_low, eps_high = (
        float(training.get("ppo_clip_low", 0.2)),
        float(training.get("ppo_clip_high", 0.28)),
    )
    dual_clip = float(training.get("ppo_dual_clip", 3.0))
    objective_valid = (
        rollout.valid_mask
        if objective_valid_mask is None
        else objective_valid_mask.to(device=rollout.valid_mask.device, dtype=torch.bool)
    )
    if objective_valid.shape != rollout.valid_mask.shape:
        raise ValueError("objective_valid_mask must align with rollout.valid_mask")
    if bool((objective_valid & ~rollout.valid_mask.bool()).any()):
        raise ValueError("Objective-valid tokens must also be valid rollout tokens")
    local_weight_mass = float(
        position_weights[objective_valid].detach().float().sum().item()
    )
    global_weight_mass = distributed.sum_float(local_weight_mass)
    if global_weight_mass <= 0.0:
        raise ValueError("Position weights must have positive global mass")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    unwrap_model(model).config.use_cache = False
    forward_seconds = backward_seconds = optimizer_seconds = loss_value = 0.0
    base_loss_sum = 0.0
    local_clipped_candidates = local_candidate_count = 0
    response_lengths = rollout.valid_mask.long().sum(dim=-1)
    order = torch.arange(local_batch_size, device=rollout.input_ids.device)
    if bool(training.get("length_bucketed_micro_batches", True)) and micro_batch > 1:
        order = torch.argsort(response_lengths, descending=True, stable=True)
    micro_batches = list(order.split(micro_batch))
    for chunk_index, indices in enumerate(micro_batches):
        synchronize = chunk_index == len(micro_batches) - 1
        fsdp_no_sync = bool(
            config.get("distributed", {}).get("fsdp", {}).get("use_no_sync", False)
        )
        may_skip_sync = isinstance(model, DistributedDataParallel) or (
            is_fsdp_model(model) and fsdp_no_sync
        )
        sync_context = (
            model.no_sync() if not synchronize and may_skip_sync else nullcontext()
        )
        with sync_context:
            cuda_sync(device)
            started = time.perf_counter()
            local_width = int(response_lengths.index_select(0, indices).max().item())
            start = rollout.prompt_width - 1
            input_stop = start + local_width
            input_ids = rollout.input_ids.index_select(0, indices)[:, :input_stop]
            attention_mask = rollout.attention_mask.index_select(0, indices)[
                :, :input_stop
            ]
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
                "return_dict": True,
            }
            response_only_logits = supports_response_only_logits(model)
            if response_only_logits:
                forward_kwargs["logits_to_keep"] = local_width
            output = model(**forward_kwargs)
            logits = (
                output.logits[:, -local_width:]
                if response_only_logits
                else output.logits[:, start : start + local_width]
            )
            candidate_ids = opd_reference.candidate_ids.index_select(0, indices)[
                :, :local_width
            ]
            current = gather_candidate_log_probs(
                logits,
                candidate_ids,
                temperature=float(config["rollout"].get("temperature", 1.0)),
                chunk_steps=int(config["selector"].get("score_chunk_steps", 128)),
            )
            chunk_reference = TopKOPDReference(
                candidate_ids=candidate_ids,
                old_student_log_probs=opd_reference.old_student_log_probs.index_select(
                    0, indices
                )[:, :local_width],
                teacher_log_probs=opd_reference.teacher_log_probs.index_select(
                    0, indices
                )[:, :local_width],
                student_weights=opd_reference.student_weights.index_select(0, indices)[
                    :, :local_width
                ],
                advantages=opd_reference.advantages.index_select(0, indices)[
                    :, :local_width
                ],
            )
            per_position_loss = topk_candidate_ppo_loss(
                current,
                chunk_reference,
                clip_low=eps_low,
                clip_high=eps_high,
                dual_clip=dual_clip,
            )
            valid_chunk = objective_valid.index_select(0, indices)[:, :local_width]
            chunk_weights = position_weights.index_select(0, indices)[:, :local_width]
            local_numerator, _ = weighted_token_sums(
                per_position_loss,
                chunk_weights,
                valid_chunk,
            )
            base_loss_sum += float(
                per_position_loss.detach()[valid_chunk].float().sum().item()
            )
            with torch.no_grad():
                ratio = torch.exp(
                    (current.detach() - chunk_reference.old_student_log_probs).clamp(
                        min=-20.0, max=20.0
                    )
                )
                candidate_valid = valid_chunk.unsqueeze(-1).expand_as(ratio)
                clipped = ratio.lt(1.0 - eps_low) | ratio.gt(1.0 + eps_high)
                local_clipped_candidates += int((clipped & candidate_valid).sum())
                local_candidate_count += int(candidate_valid.sum())
            # DDP and FSDP data-parallel reductions average synchronized rank
            # gradients. The world-size factor therefore yields exactly:
            # global sum(w_t*l_t) / global sum(w_t), including accumulation.
            loss = local_numerator * (
                distributed.world_size / float(global_weight_mass)
            )
            cuda_sync(device)
            forward_seconds += time.perf_counter() - started
            finite_or_raise("OPD loss", loss.detach().reshape(1))
            cuda_sync(device)
            started = time.perf_counter()
            loss.backward()
            cuda_sync(device)
            backward_seconds += time.perf_counter() - started
            loss_value += float(loss.detach().item())
            del (
                output,
                logits,
                current,
                per_position_loss,
                local_numerator,
                loss,
                input_ids,
                attention_mask,
                candidate_ids,
                chunk_weights,
            )
    gradient_norm = clip_grad_norm(
        model,
        float(training.get("max_grad_norm", 1.0)),
    )
    cuda_sync(device)
    started = time.perf_counter()
    optimizer.step()
    cuda_sync(device)
    optimizer_seconds = time.perf_counter() - started
    optimizer.zero_grad(set_to_none=True)
    global_loss = distributed.sum_float(loss_value) / distributed.world_size
    global_base_loss_sum = distributed.sum_float(base_loss_sum)
    global_valid_tokens = distributed.sum_int(int(objective_valid.sum().item()))
    global_clipped_candidates = distributed.sum_int(local_clipped_candidates)
    global_candidate_count = distributed.sum_int(local_candidate_count)
    return {
        "loss": global_loss,
        "weighted_final_loss": global_loss,
        "base_topk_opd_loss": global_base_loss_sum / max(global_valid_tokens, 1),
        "unweighted_opd_loss": global_base_loss_sum / max(global_valid_tokens, 1),
        "gradient_norm": float(gradient_norm.detach().item()),
        "training_forward_time": forward_seconds,
        "backward_time": backward_seconds,
        "optimizer_time": optimizer_seconds,
        "global_weight_mass": global_weight_mass,
        "clip_fraction": global_clipped_candidates / max(global_candidate_count, 1),
    }


def _save_checkpoint(
    model,
    tokenizer,
    optimizer,
    output_dir: Path,
    step: int,
    final: bool,
    save_optimizer: bool,
    distributed: DistributedContext,
):
    checkpoint = output_dir / ("final" if final else f"checkpoint-{step:06d}")
    temporary = output_dir / f".{checkpoint.name}.incomplete"
    if checkpoint.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint path: {checkpoint}")
    try:
        if distributed.is_main:
            temporary.mkdir(parents=True)
        distributed.barrier()
        _save_inference_snapshot(model, tokenizer, temporary, distributed)
        if save_optimizer:
            optimizer_state = (
                full_optimizer_state_dict(model, optimizer)
                if is_fsdp_model(model) or distributed.is_main
                else None
            )
            local_rng = {
                "torch_rng_state": torch.get_rng_state().cpu(),
                "cuda_rng_state": torch.cuda.get_rng_state(distributed.device).cpu(),
            }
            rng_states = distributed.all_gather_objects(local_rng)
            if distributed.is_main:
                torch.save(
                    {
                        "step": step,
                        "optimizer": optimizer_state,
                        "optimizer_format": (
                            "fsdp_full_v1" if is_fsdp_model(model) else "standard"
                        ),
                        "world_size": distributed.world_size,
                        "rng_states": rng_states,
                        # Retain legacy fields for older single-process loaders.
                        "torch_rng_state": rng_states[0]["torch_rng_state"],
                        "cuda_rng_state_all": [
                            state["cuda_rng_state"] for state in rng_states
                        ],
                    },
                    temporary / "optimizer.pt",
                )
        distributed.barrier()
        if distributed.is_main:
            os.replace(temporary, checkpoint)
    except BaseException:
        if distributed.is_main and temporary.exists():
            shutil.rmtree(temporary)
        raise
    distributed.barrier()
    if distributed.is_main:
        latest = output_dir / "latest.json"
        latest_temporary = output_dir / ".latest.json.tmp"
        latest_temporary.write_text(
            json.dumps(
                {"step": step, "checkpoint": checkpoint.name, "final": bool(final)},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(latest_temporary, latest)
    distributed.barrier()
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
    strategy = distributed_strategy(config, distributed)
    method = str(experiment["method"]).lower()
    if method not in {"opd", "ta", "rac"}:
        raise ValueError(f"Training method must be opd, ta, or rac, got {method!r}")
    # Fail before starting vLLM or loading either model if a launch override
    # accidentally enables thinking or proposes a teacher tokenizer path.
    validate_shared_tokenizer_protocol(config)
    global_prompt_batch_size = int(config["rollout"]["batch_size"])
    num_responses = int(config["rollout"].get("num_responses", 1))
    micro_batch_size_per_gpu = _micro_batch_size_per_gpu(
        training, distributed.world_size
    )
    layout = batch_layout(
        global_prompt_batch_size,
        num_responses,
        distributed.world_size,
        micro_batch_size_per_gpu,
    )
    configured_accumulation = training.get("grad_accum_steps", "auto")
    if configured_accumulation not in (None, "auto"):
        raise ValueError(
            "training.grad_accum_steps is derived automatically from each local "
            "trajectory batch. Configure training.micro_batch_size_per_gpu instead."
        )
    if distributed.is_main:
        tqdm.write(_format_batch_layout(layout, strategy, config))
    if strategy == "fsdp" and str(
        config["models"].get("dtype", "bfloat16")
    ).lower() not in {
        "bfloat16",
        "bf16",
    }:
        raise ValueError("FSDP production training requires models.dtype=bfloat16")
    opd_config = config.get("opd", {})
    top_k_strategy = str(opd_config.get("top_k_strategy", "only_stu"))
    reward_weight_mode = str(opd_config.get("reward_weight_mode", "student_p"))
    advantage_estimator = str(opd_config.get("adv_estimator", "token_reward_direct"))
    loss_aggregation = str(opd_config.get("loss_agg_mode", "token-mean"))
    controlled_settings = {
        "upstream_commit": (
            str(opd_config.get("upstream_commit", UPSTREAM_OPD_COMMIT)),
            UPSTREAM_OPD_COMMIT,
        ),
        "top_k_strategy": (top_k_strategy, UPSTREAM_TOP_K_STRATEGY),
        "reward_weight_mode": (reward_weight_mode, UPSTREAM_REWARD_WEIGHT_MODE),
        "adv_estimator": (advantage_estimator, UPSTREAM_ADV_ESTIMATOR),
        "loss_agg_mode": (loss_aggregation, UPSTREAM_LOSS_AGG_MODE),
    }
    incompatible = {
        key: actual
        for key, (actual, expected) in controlled_settings.items()
        if actual != expected
    }
    if incompatible:
        raise ValueError(
            "This controlled experiment supports only the pinned thunlp/OPD "
            f"recipe; incompatible settings: {incompatible}"
        )
    seed = int(experiment.get("seed", 1234))
    seed_everything(seed)
    resume_checkpoint = resolve_resume_checkpoint(
        training.get("resume_from_checkpoint"), experiment.get("output_dir")
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
    if strategy == "fsdp" and rollout_backend != "vllm":
        raise RuntimeError(
            "FSDP training requires the per-rank vLLM rollout backend. The HF "
            "autoregressive fallback can finish at different times on each rank "
            "and is not collective-safe."
        )
    if (
        strategy == "fsdp"
        and bool(config.get("training_evaluation", {}).get("enabled", False))
        and str(config["training_evaluation"].get("backend", "vllm")).lower() != "vllm"
    ):
        raise RuntimeError(
            "FSDP periodic evaluation must use backend=vllm so rank 0 evaluates "
            "a collectively exported HF snapshot."
        )
    rollout_engine: VLLMRolloutEngine | None = None

    setup_progress = tqdm(
        total=4,
        desc=f"Setup {METHOD_DISPLAY_NAMES[method]}",
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
        raise ValueError("Configured training dataset is empty")
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
    scoring_teacher = teacher
    if strategy == "fsdp":
        training_student = wrap_fsdp_model(
            student,
            config,
            distributed,
            role="student",
        )
        scoring_teacher = wrap_fsdp_model(
            teacher,
            config,
            distributed,
            role="teacher",
        )
    elif strategy == "ddp":
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
        [
            parameter
            for parameter in training_student.parameters()
            if parameter.requires_grad
        ],
        training,
    )
    resume_state = (
        restore_optimizer(
            optimizer,
            resume_checkpoint,
            device,
            model=training_student,
            distributed=distributed,
        )
        if resume_checkpoint is not None
        else None
    )
    resume_step = resume_state.step if resume_state is not None else 0
    resume_history = None
    if resume_state is not None:
        if distributed.is_main:
            resume_history = validate_append_history(
                output_dir, resume_step, resume_state.checkpoint
            )
        distributed.barrier()
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
            "strategy": strategy,
            "world_size": distributed.world_size,
            "global_batch_preserved": True,
            "global_ta_normalization": method in {"ta", "rac"},
            "global_token_budget": method == "ta",
            "global_rac_weight_normalization": method == "rac",
            "uniform_full_response_mask": method == "opd",
            "student_sharding_strategy": ("FULL_SHARD" if strategy == "fsdp" else None),
            "teacher_sharding_strategy": (
                "FULL_SHARD" if strategy == "fsdp" else "replicated"
            ),
            "teacher_cpu_offload": bool(
                distributed_cfg.get("fsdp", {}).get("teacher_cpu_offload", False)
            ),
        }
        metadata["opd_upstream"] = {
            "repository": "https://github.com/thunlp/OPD",
            "commit": UPSTREAM_OPD_COMMIT,
            "adv_estimator": advantage_estimator,
            "top_k_strategy": top_k_strategy,
            "reward_weight_mode": reward_weight_mode,
            "loss_agg_mode": loss_aggregation,
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
    top_k = int(selector_cfg.get("top_k", 16))
    if top_k <= 0:
        raise ValueError("selector.top_k must be positive")
    opd_selector = OPDSelector()
    ta_selector = TASelector(
        top_k,
        float(selector_cfg.get("q_low", 0.05)),
        float(selector_cfg.get("q_high", 0.95)),
        float(selector_cfg.get("eps", 1e-8)),
    )
    rac_selector = RACSelector(
        gamma=float(selector_cfg.get("rac_gamma", 0.995)),
        w_min=float(selector_cfg.get("rac_w_min", 0.10)),
        beta=float(selector_cfg.get("rac_beta", 2.0)),
        q_low=float(selector_cfg.get("q_low", 0.05)),
        q_high=float(selector_cfg.get("q_high", 0.95)),
        eps=float(selector_cfg.get("eps", 1e-8)),
        scan_backend=str(selector_cfg.get("rac_scan_backend", "parallel")),
    )
    batch_size = global_prompt_batch_size
    if batch_size <= 0 or num_responses <= 0:
        raise ValueError("Prompt batch size and rollout.num_responses must be positive")
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
        enabled=method == "ta"
        and bool(config.get("logging", {}).get("selected_tokens_enabled", True)),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    score_stats_logger = TokenScoreStatsLogger(
        output_dir,
        method,
        interval=int(config.get("logging", {}).get("token_score_interval", 50)),
        bins=int(config.get("logging", {}).get("token_score_histogram_bins", 64)),
        raw_sample_size=int(
            config.get("logging", {}).get("token_score_raw_sample_size", 2048)
        ),
        enabled=distributed.is_main
        and bool(config.get("logging", {}).get("token_score_stats_enabled", True)),
    )
    tensorboard_logger = TensorBoardLogger(
        output_dir,
        dict(config.get("logging", {}).get("tensorboard", {})),
        enabled=distributed.is_main,
        resume_step=resume_step,
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
            f"Resuming {METHOD_DISPLAY_NAMES[method]} from optimizer step {resume_step}: "
            f"{resume_state.checkpoint}"
        )
        if resume_history is not None and resume_history["rewound"]:
            removed_row_count = sum(resume_history["removed_rows"].values())
            removed_row_count += int(resume_history["selector_rows_removed"])
            tqdm.write(
                f"Rewound existing outputs to step {resume_step}: removed "
                f"{removed_row_count} later log rows and "
                f"{len(resume_history['removed_paths'])} stale paths."
            )
    initial_evaluation = None
    if resume_step == 0 and should_run_training_evaluation(
        0, max_steps, training_eval_settings
    ):
        distributed.barrier()
        if distributed.is_main:
            tqdm.write("Evaluating the untouched base student at optimizer step 0...")
            initial_evaluation = _run_training_evaluation(
                training_student,
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
        desc=f"{METHOD_DISPLAY_NAMES[method]} B200",
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
        local_start, _ = contiguous_partition(
            len(global_indices), distributed.rank, distributed.world_size
        )
        prompt_indices, active_prompts = padded_local_indices(
            global_indices,
            distributed.rank,
            distributed.world_size,
        )
        prompt_records = [records[index] for index in prompt_indices]
        encoded, _ = tokenize_prompts(prompt_records, tokenizer, config["data"], device)
        encoded, indices, response_indices = expand_prompt_batch(
            encoded, prompt_indices, num_responses
        )
        active_trajectories = torch.tensor(
            [active for active in active_prompts for _ in range(num_responses)],
            dtype=torch.bool,
            device=device,
        )
        batch_records = [records[index] for index in indices]
        rollout_function = (
            rollout_engine.generate
            if rollout_engine is not None
            else generate_on_policy
        )
        rollout, rollout_time = _timed(
            device,
            rollout_function,
            training_student,
            encoded["input_ids"],
            encoded["attention_mask"],
            max_new_tokens=int(config["rollout"].get("max_new_tokens", 256)),
            temperature=float(config["rollout"].get("temperature", 1.0)),
            top_p=float(config["rollout"].get("top_p", 1.0)),
            eos_token_ids=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            seed=int(config["rollout"].get("seed", seed)),
            # Repeated prompts are separate requests with globally unique,
            # deterministic seeds, so all n trajectories are independent.
            sample_seed_offset=(
                step_index * batch_size * num_responses + local_start * num_responses
            ),
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
        objective_valid = rollout.valid_mask & active_trajectories.unsqueeze(1)
        rollout_hash = _rollout_hash(rollout.response_ids, objective_valid, distributed)
        use_joint_scoring = method in {"ta", "rac"} and bool(
            selector_cfg.get("joint_cross_scoring", True)
        )
        joint_scoring_time = 0.0
        if use_joint_scoring:
            if distributed.is_main:
                progress.set_postfix_str(
                    "stage=score-student-teacher-joint", refresh=True
                )
            joint_scores, joint_scoring_time = _timed(
                device,
                score_student_teacher_rollout,
                training_student,
                scoring_teacher,
                rollout,
                score_chunk_steps=int(selector_cfg.get("score_chunk_steps", 128)),
                top_k=top_k,
                student_temperature=float(config["rollout"].get("temperature", 1.0)),
                teacher_temperature=float(opd_config.get("teacher_temperature", 1.0)),
                micro_batch_size=int(selector_cfg.get("score_micro_batch_size", 1)),
                trim_padding=bool(selector_cfg.get("trim_padding", True)),
                length_bucketed=bool(selector_cfg.get("length_bucketed_scoring", True)),
            )
            student_scores, teacher_scores = joint_scores
            student_base_time = teacher_base_time = 0.0
        else:
            if distributed.is_main:
                progress.set_postfix_str("stage=score-student", refresh=True)
            student_scores, student_base_time = _timed(
                device,
                score_original_rollout,
                training_student,
                rollout,
                False,
                int(selector_cfg.get("score_chunk_steps", 128)),
                retain_response_logits=False,
                top_k=top_k,
                temperature=float(config["rollout"].get("temperature", 1.0)),
                micro_batch_size=int(selector_cfg.get("score_micro_batch_size", 1)),
                trim_padding=bool(selector_cfg.get("trim_padding", True)),
                length_bucketed=bool(selector_cfg.get("length_bucketed_scoring", True)),
            )
        if student_scores.top_k_ids is None or student_scores.top_k_log_probs is None:
            raise AssertionError(
                "Student scoring did not produce the common Top-K support"
            )
        if not use_joint_scoring:
            if distributed.is_main:
                progress.set_postfix_str("stage=score-teacher", refresh=True)
            teacher_scores, teacher_base_time = _timed(
                device,
                score_original_rollout,
                scoring_teacher,
                rollout,
                False,
                int(selector_cfg.get("score_chunk_steps", 128)),
                retain_response_logits=False,
                top_k=top_k,
                candidate_ids=student_scores.top_k_ids,
                temperature=float(opd_config.get("teacher_temperature", 1.0)),
                micro_batch_size=int(selector_cfg.get("score_micro_batch_size", 1)),
                trim_padding=bool(selector_cfg.get("trim_padding", True)),
                length_bucketed=bool(selector_cfg.get("length_bucketed_scoring", True)),
            )
        if (
            teacher_scores.candidate_log_probs is None
            or teacher_scores.top_k_ids is None
            or teacher_scores.top_k_log_probs is None
        ):
            raise AssertionError("Teacher scoring did not score the student Top-K IDs")
        valid = objective_valid
        finite_or_raise(
            "student sampled log-probs", student_scores.sampled_log_probs[valid]
        )
        finite_or_raise(
            "teacher sampled log-probs", teacher_scores.sampled_log_probs[valid]
        )
        opd_reference = build_topk_opd_reference(
            student_scores.top_k_ids,
            student_scores.top_k_log_probs,
            teacher_scores.candidate_log_probs,
            valid,
        )
        finite_or_raise("Top-K OPD advantages", opd_reference.advantages[valid])
        local_advantage_abs_sum = float(
            opd_reference.advantages[valid].abs().sum().item()
        )
        global_advantage_count = distributed.sum_int(
            opd_reference.advantages[valid].numel()
        )
        opd_advantage_abs_mean = distributed.sum_float(local_advantage_abs_sum) / max(
            global_advantage_count, 1
        )
        local_logprob_gap = (
            teacher_scores.sampled_log_probs[valid]
            - student_scores.sampled_log_probs[valid]
        )
        global_logprob_gap_count = distributed.sum_int(local_logprob_gap.numel())
        opd_teacher_student_logprob_gap = distributed.sum_float(
            float(local_logprob_gap.sum().item())
        ) / max(global_logprob_gap_count, 1)
        overlap_values = topk_overlap_fraction(
            student_scores.top_k_ids, teacher_scores.top_k_ids, valid
        )
        global_overlap_count = distributed.sum_int(overlap_values.numel())
        student_teacher_topk_overlap = distributed.sum_float(
            float(overlap_values.sum().item())
        ) / max(global_overlap_count, 1)
        local_divergence = -opd_reference.advantages.sum(dim=-1)[valid]
        global_divergence_count = distributed.sum_int(local_divergence.numel())
        student_teacher_topk_divergence = distributed.sum_float(
            float(local_divergence.sum().item())
        ) / max(global_divergence_count, 1)

        vllm_logprob_sanity: dict[str, Any] = {"enabled": False}
        sanity = dict(config["rollout"].get("vllm", {}).get("logprob_sanity", {}))
        if rollout_backend == "vllm" and bool(sanity.get("enabled", False)):
            comparable = valid & torch.isfinite(rollout.rollout_log_probs)
            differences = (
                rollout.rollout_log_probs[comparable]
                - student_scores.sampled_log_probs[comparable]
            ).abs()[: max(1, int(sanity.get("max_tokens_per_rank", 32)))]
            compared = distributed.sum_int(differences.numel())
            if compared == 0:
                raise RuntimeError(
                    "vLLM log-prob sanity was enabled, but the server returned no "
                    "token log-probabilities"
                )
            mean_abs = distributed.sum_float(float(differences.sum().item())) / max(
                compared, 1
            )
            local_max_abs = (
                float(differences.max().item()) if differences.numel() else 0.0
            )
            max_abs = distributed.max_float(local_max_abs)
            tolerance = float(sanity.get("tolerance", 0.05))
            vllm_logprob_sanity = {
                "enabled": True,
                "compared_tokens": compared,
                "mean_abs_error": mean_abs,
                "max_abs_error": max_abs,
                "tolerance": tolerance,
                "passed": max_abs <= tolerance,
            }
            if not vllm_logprob_sanity["passed"] and bool(
                sanity.get("fail_on_mismatch", False)
            ):
                raise RuntimeError(
                    "vLLM/HF student log-prob sanity failed after weight sync: "
                    f"max_abs_error={max_abs:.6f} > tolerance={tolerance:.6f}"
                )
        cross_diagnostics = {}
        ta_raw = ta_output = None
        global_ta_diagnostics: dict[str, torch.Tensor] = {}
        student_cross_score_time = 0.0
        if method == "opd":
            if distributed.is_main:
                progress.set_postfix_str("stage=selector-OPD-uniform", refresh=True)
            opd_raw, opd_selector_time = _timed(
                device,
                opd_selector.compute_scores,
                valid,
            )
            opd_globalized, opd_gather_time = _timed(
                device,
                _globalize_opd_output,
                opd_raw,
                valid,
                distributed,
            )
            primary, global_primary_diagnostics, primary_start, primary_end = (
                opd_globalized
            )
            del opd_raw
            ta_time = bellman_scan_time = 0.0
            selector_time = opd_selector_time + opd_gather_time
        else:
            if use_joint_scoring:
                student_on_teacher = student_scores
            else:
                student_on_teacher, student_cross_score_time = _timed(
                    device,
                    score_original_rollout,
                    training_student,
                    rollout,
                    False,
                    int(selector_cfg.get("score_chunk_steps", 128)),
                    retain_response_logits=False,
                    candidate_ids=teacher_scores.top_k_ids,
                    temperature=float(config["rollout"].get("temperature", 1.0)),
                    micro_batch_size=int(selector_cfg.get("score_micro_batch_size", 1)),
                    trim_padding=bool(selector_cfg.get("trim_padding", True)),
                    length_bucketed=bool(
                        selector_cfg.get("length_bucketed_scoring", True)
                    ),
                )
            if student_on_teacher.candidate_log_probs is None:
                raise AssertionError("Student did not score the teacher Top-K IDs")
            ta_raw, ta_raw_time = _timed(
                device,
                ta_selector.compute_scores_from_topk,
                student_scores.top_k_ids,
                teacher_scores.top_k_ids,
                student_scores.top_k_log_probs,
                teacher_scores.candidate_log_probs,
                teacher_scores.top_k_log_probs,
                student_on_teacher.candidate_log_probs,
                valid,
                normalize=False,
                token_chunk_size=int(selector_cfg.get("ta_vocab_chunk_tokens", 2048)),
            )
            del student_on_teacher
            ta_globalized, ta_normalization_time = _timed(
                device,
                _globalize_ta_output,
                ta_raw,
                valid,
                ta_selector,
                distributed,
            )
            ta_output, global_ta_diagnostics, ta_start, ta_end = ta_globalized
            ta_time = student_cross_score_time + ta_raw_time + ta_normalization_time
            if method == "rac":
                if distributed.is_main:
                    progress.set_postfix_str("stage=selector-Bellman-RAC", refresh=True)
                rac_raw, bellman_scan_time = _timed(
                    device,
                    rac_selector.compute_scores,
                    ta_output.scores,
                    student_scores.sampled_log_probs,
                    teacher_scores.sampled_log_probs,
                    valid,
                    normalize=False,
                )
                rac_globalized, rac_normalization_time = _timed(
                    device,
                    _globalize_rac_output,
                    rac_raw,
                    valid,
                    rac_selector,
                    distributed,
                )
                primary, global_primary_diagnostics, primary_start, primary_end = (
                    rac_globalized
                )
                del rac_raw
                if (primary_start, primary_end) != (ta_start, ta_end):
                    raise AssertionError("TA/RAC distributed token layouts differ")
                cross_diagnostics.update(
                    correlations(
                        global_ta_diagnostics["s_TA"],
                        global_primary_diagnostics,
                        torch.ones_like(
                            global_ta_diagnostics["s_TA"], dtype=torch.bool
                        ),
                    )
                )
                selector_time = bellman_scan_time + rac_normalization_time
            else:
                if distributed.is_main:
                    progress.set_postfix_str("stage=selector-TA", refresh=True)
                primary, selector_time, bellman_scan_time = (
                    ta_output,
                    ta_time,
                    0.0,
                )
                global_primary_diagnostics = global_ta_diagnostics
                primary_start, primary_end = ta_start, ta_end
        finite_or_raise(f"{method} selector", primary.scores[valid])
        score_key = "s_TA" if method == "ta" else "w"
        if method == "ta":
            selected, global_selected = _local_mask_from_global_budget(
                global_primary_diagnostics[score_key],
                valid,
                primary_start,
                primary_end,
                rho,
            )
            expected = math.ceil(rho * global_primary_diagnostics[score_key].numel())
            token_allocation = selected
        else:
            selected = valid.clone()
            global_selected = torch.ones_like(
                global_primary_diagnostics[score_key], dtype=torch.bool
            )
            expected = global_primary_diagnostics[score_key].numel()
            token_allocation = selected if method == "opd" else primary.scores
        # RAC selector computations run under inference_mode.  Materialize a
        # normal frozen tensor before it participates in a differentiable
        # weighted loss (the same guard used by TopKOPDReference).
        with torch.inference_mode(False):
            token_allocation = token_allocation.detach().clone()
        if distributed.sum_int(int(selected.sum().item())) != expected:
            raise AssertionError(f"{method} supervised-token count is incorrect")
        if not torch.equal(original_rollout, rollout.input_ids):
            raise AssertionError("Selector changed the original rollout")
        token_score_stats_path = (
            score_stats_logger.write(step, max_steps, global_primary_diagnostics)
            if distributed.is_main
            else None
        )
        sample_ids = [
            f"{stable_sample_id(record, index)}::response-{response_index}"
            for record, index, response_index in zip(
                batch_records, indices, response_indices
            )
        ]
        logged_selected = (
            token_logger.write(
                step=step,
                dataset_indices=indices,
                sample_ids=sample_ids,
                response_ids=rollout.response_ids,
                selected_mask=selected,
                diagnostics=primary.diagnostics,
                batch_index_offset=local_start * num_responses,
            )
            if method == "ta"
            else 0
        )
        global_logged_selected = distributed.sum_int(logged_selected)
        if (
            bool(config.get("logging", {}).get("selected_tokens_enabled", True))
            and method == "ta"
            and global_logged_selected != expected
        ):
            raise AssertionError(
                f"Detailed selector logs wrote {global_logged_selected}, "
                f"expected {expected}"
            )

        del student_scores, teacher_scores
        # Let PyTorch reuse the released scoring-logit blocks for backward.
        # Emptying the CUDA allocator every step is materially slower on B200.
        if bool(training.get("empty_cuda_cache_each_step", False)):
            torch.cuda.empty_cache()
        if distributed.is_main:
            progress.set_postfix_str("stage=train", refresh=True)
        train_metrics = _opd_train_step(
            training_student,
            optimizer,
            rollout,
            token_allocation,
            opd_reference,
            config,
            device,
            distributed,
            objective_valid_mask=objective_valid,
        )
        checkpoint = None
        save_checkpoints = bool(training.get("save_checkpoints", True))
        save_interval = int(training.get("save_interval", 100))
        should_save_checkpoint = save_checkpoints and (
            step == max_steps or (save_interval > 0 and step % save_interval == 0)
        )
        if should_save_checkpoint:
            if distributed.is_main:
                progress.set_postfix_str("stage=checkpoint", refresh=True)
            checkpoint = _save_checkpoint(
                training_student,
                tokenizer,
                optimizer,
                output_dir,
                step,
                step == max_steps,
                bool(training.get("save_optimizer", True)),
                distributed,
            )
        cuda_sync(device)
        local_wall_time = time.perf_counter() - step_started
        local_valid_tokens = int(valid.sum().item())
        response_lengths = valid.sum(dim=-1)
        active_responses = response_lengths.gt(0)
        local_response_count = int(active_responses.sum().item())
        if local_response_count:
            local_response_min = int(response_lengths[active_responses].min().item())
            local_response_max = int(response_lengths[active_responses].max().item())
        else:
            # An inactive-only rank is possible for a one-prompt dataset tail.
            # Use a neutral max and a high min sentinel for global reduction.
            local_response_min = int(config["rollout"]["max_new_tokens"]) + 1
            local_response_max = 0
        local_clipped_responses = int(
            (
                response_lengths.ge(int(config["rollout"]["max_new_tokens"]))
                & active_responses
            )
            .sum()
            .item()
        )
        eos_token_ids = tokenizer.eos_token_id
        eos_token_ids = (
            [eos_token_ids] if isinstance(eos_token_ids, int) else eos_token_ids
        )
        response_has_eos = torch.zeros_like(active_responses)
        for eos_token_id in eos_token_ids:
            response_has_eos |= (
                rollout.response_ids.eq(int(eos_token_id)) & valid
            ).any(dim=-1)
        local_eos_responses = int((response_has_eos & active_responses).sum().item())
        local_peak_allocated = torch.cuda.max_memory_allocated(device)
        local_peak_reserved = torch.cuda.max_memory_reserved(device)
        wall_time = distributed.max_float(local_wall_time)
        valid_tokens = distributed.sum_int(local_valid_tokens)
        trajectory_count = distributed.sum_int(local_response_count)
        response_min = -distributed.max_int(-local_response_min)
        response_max = distributed.max_int(local_response_max)
        clipped_responses = distributed.sum_int(local_clipped_responses)
        eos_responses = distributed.sum_int(local_eos_responses)
        global_tail_padding_prompts = distributed.sum_int(
            len(active_prompts) - sum(active_prompts)
        )
        peak_allocated = distributed.max_int(local_peak_allocated)
        peak_reserved = distributed.max_int(local_peak_reserved)
        peak_allocated_total = distributed.sum_int(local_peak_allocated)
        peak_reserved_total = distributed.sum_int(local_peak_reserved)
        aggregated_train_metrics = {
            "loss": train_metrics["loss"],
            "train_loss": train_metrics["loss"],
            "weighted_final_loss": train_metrics["weighted_final_loss"],
            "base_topk_opd_loss": train_metrics["base_topk_opd_loss"],
            "unweighted_opd_loss": train_metrics["unweighted_opd_loss"],
            "gradient_norm": distributed.max_float(train_metrics["gradient_norm"]),
            "grad_norm": distributed.max_float(train_metrics["gradient_norm"]),
            "training_forward_time": distributed.max_float(
                train_metrics["training_forward_time"]
            ),
            "backward_time": distributed.max_float(train_metrics["backward_time"]),
            "optimizer_time": distributed.max_float(train_metrics["optimizer_time"]),
            "global_weight_mass": train_metrics["global_weight_mass"],
            "opd_clip_fraction": train_metrics["clip_fraction"],
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
            "prompt_batch_size": len(global_indices),
            "global_prompt_batch_size": len(global_indices),
            "local_prompt_batch_size": len(prompt_indices),
            "local_real_prompt_count_rank0": sum(active_prompts),
            "tail_padding_prompt_count": global_tail_padding_prompts,
            "num_responses_per_prompt": num_responses,
            "trajectory_batch_size": len(global_indices) * num_responses,
            "global_trajectory_batch_size": len(global_indices) * num_responses,
            "local_trajectory_batch_size": len(prompt_indices) * num_responses,
            "num_trajectories": trajectory_count,
            "local_batch_size_rank0": (
                len(global_indices) // distributed.world_size
                + int(0 < len(global_indices) % distributed.world_size)
            )
            * num_responses,
            "configured_batch_size": batch_size,
            "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
            "distributed_world_size": distributed.world_size,
            "distributed_strategy": strategy,
            "global_batch_preserved": True,
            "global_ta_normalization": method in {"ta", "rac"},
            "global_token_budget": method == "ta",
            "uniform_full_response_mask": method == "opd",
            "all_response_tokens_supervised": method in {"opd", "rac"},
            "token_allocation_policy": {
                "opd": "uniform_all_valid_response_tokens",
                "ta": "hard_global_top_rho",
                "rac": "bellman_soft_all_valid_response_tokens",
            }[method],
            "objective_normalization": "global_weighted_token_mean",
            "opd_upstream_commit": UPSTREAM_OPD_COMMIT,
            "opd_candidate_support": "student_top_k",
            "opd_top_k": top_k,
            "opd_top_k_strategy": top_k_strategy,
            "opd_reward_weight_mode": reward_weight_mode,
            "opd_adv_estimator": advantage_estimator,
            "opd_loss_agg_mode": loss_aggregation,
            "opd_advantage_abs_mean": opd_advantage_abs_mean,
            "opd_teacher_student_logprob_gap": (opd_teacher_student_logprob_gap),
            "fused_optimizer": fused_optimizer,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **aggregated_train_metrics,
            "rollout_backend": rollout_backend,
            "rollout_time": distributed.max_float(rollout_time),
            "vllm_weight_sync_time": distributed.max_float(
                rollout_backend_metrics.get("weight_sync_time", 0.0)
            ),
            "vllm_full_weight_materialization_time": distributed.max_float(
                rollout_backend_metrics.get("full_weight_materialization_time", 0.0)
            ),
            "vllm_ipc_weight_sync_time": distributed.max_float(
                rollout_backend_metrics.get("ipc_weight_sync_time", 0.0)
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
            "vllm_logprob_sanity": vllm_logprob_sanity,
            "student_base_scoring_time": distributed.max_float(student_base_time),
            "teacher_base_scoring_time": distributed.max_float(teacher_base_time),
            # Legacy aggregate column: under joint scoring this represents the
            # complete two-model common scoring stage, not teacher-only time.
            "teacher_score_time_sec": distributed.max_float(
                joint_scoring_time if use_joint_scoring else teacher_base_time
            ),
            "student_cross_topk_scoring_time": distributed.max_float(
                student_cross_score_time
            ),
            "joint_student_teacher_scoring_time": distributed.max_float(
                joint_scoring_time
            ),
            "total_scoring_time_sec": distributed.max_float(
                joint_scoring_time
                + student_base_time
                + teacher_base_time
                + student_cross_score_time
            ),
            "ta_diagnostic_time": distributed.max_float(ta_time),
            "ta_local_score_time_sec": distributed.max_float(ta_time),
            "selector_time": distributed.max_float(selector_time),
            "bellman_scan_time_sec": distributed.max_float(bellman_scan_time),
            "forward_backward_time_sec": distributed.max_float(
                train_metrics["training_forward_time"] + train_metrics["backward_time"]
            ),
            "optimizer_time_sec": distributed.max_float(
                train_metrics["optimizer_time"]
            ),
            "wall_clock_step_time": wall_time,
            "tokens_per_second": valid_tokens / max(wall_time, 1e-12),
            "throughput_tokens_per_sec": valid_tokens / max(wall_time, 1e-12),
            "generated_tokens_per_second": valid_tokens
            / max(distributed.max_float(rollout_time), 1e-12),
            "training_tokens_per_second": valid_tokens
            / max(
                distributed.max_float(
                    train_metrics["training_forward_time"]
                    + train_metrics["backward_time"]
                    + train_metrics["optimizer_time"]
                ),
                1e-12,
            ),
            "num_valid_tokens": valid_tokens,
            "mean_response_length": valid_tokens / max(trajectory_count, 1),
            "min_response_length": response_min,
            "max_response_length": response_max,
            "response_clip_ratio": clipped_responses / max(trajectory_count, 1),
            "eos_fraction": eos_responses / max(trajectory_count, 1),
            "student_teacher_topk_overlap_ratio": student_teacher_topk_overlap,
            "student_teacher_topk_divergence": student_teacher_topk_divergence,
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "peak_gpu_allocated_gb": peak_allocated / 2**30,
            "peak_gpu_reserved_gb": peak_reserved / 2**30,
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
            "token_score_stats": (
                str(token_score_stats_path) if token_score_stats_path else None
            ),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "rollout_token_sha256": rollout_hash,
        }
        # The rollout server is already sleeping; release tensors before a
        # possible periodic-evaluation subprocess reserves its KV cache.
        del (
            encoded,
            rollout,
            original_rollout,
            opd_reference,
            selected,
            token_allocation,
            primary,
            ta_raw,
            ta_output,
            global_primary_diagnostics,
            global_ta_diagnostics,
            global_selected,
            valid,
            objective_valid,
            active_trajectories,
            rollout_backend_metrics,
        )
        if should_run_training_evaluation(step, max_steps, training_eval_settings):
            evaluation_checkpoint = checkpoint
            temporary_eval_snapshot = None
            if strategy == "fsdp" and evaluation_checkpoint is None:
                temporary_eval_snapshot = (
                    output_dir / ".evaluation_snapshots" / f"step-{step:06d}"
                )
                if distributed.is_main and temporary_eval_snapshot.exists():
                    shutil.rmtree(temporary_eval_snapshot)
                distributed.barrier()
                _save_inference_snapshot(
                    training_student,
                    tokenizer,
                    temporary_eval_snapshot,
                    distributed,
                )
                evaluation_checkpoint = temporary_eval_snapshot
            distributed.barrier()
            if distributed.is_main:
                progress.set_postfix_str("stage=evaluation", refresh=True)
                periodic_evaluation = _run_training_evaluation(
                    training_student,
                    tokenizer,
                    method,
                    step,
                    max_steps,
                    config,
                    output_dir,
                    resolved_config_path,
                    checkpoint=evaluation_checkpoint,
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
            if distributed.is_main and temporary_eval_snapshot is not None:
                shutil.rmtree(temporary_eval_snapshot)
            distributed.barrier()
        if distributed.is_main:
            _append_jsonl(metrics_path, final_metrics)
            _append_train_metrics_csv(output_dir / "train_metrics.csv", final_metrics)
            tensorboard_logger.write(step, final_metrics, method)
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
    tensorboard_logger.close()
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
        "selector_score_dir": (
            str((output_dir / "selector_scores").resolve()) if method == "ta" else None
        ),
        "token_score_stats_dir": str((output_dir / "token_score_stats").resolve()),
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
