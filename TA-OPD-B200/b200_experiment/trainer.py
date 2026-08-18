from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import save_config
from .data import epoch_batch_indices, read_records, stable_sample_id, tokenize_prompts
from .diagnostics import correlations, finite_or_raise, selector_summary
from .evaluation import BENCHMARK_ORDER, evaluate_loaded_suite
from .eval_schedule import (
    should_run_training_evaluation,
    training_evaluation_steps,
)
from .metadata import collect_metadata, save_metadata
from .models import load_models
from .scoring import (
    HFBranchProber,
    cuda_sync,
    generate_on_policy,
    score_original_rollout,
)
from .selector_logging import SelectedTokenLogger
from .selectors import RACSelector, TASelector, top_budget_mask


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
            output_dir / "resolved_config.yaml",
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
):
    training = config["training"]
    micro_batch = max(
        1, int(training.get("micro_batch_size", rollout.input_ids.shape[0]))
    )
    batch_size = rollout.input_ids.shape[0]
    eps_low, eps_high = (
        float(training.get("ppo_clip_low", 0.2)),
        float(training.get("ppo_clip_high", 0.28)),
    )
    advantage = (teacher_log_probs - old_student_log_probs).detach()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    model.config.use_cache = False
    forward_seconds = backward_seconds = loss_value = 0.0
    for begin in range(0, batch_size, micro_batch):
        end = min(begin + micro_batch, batch_size)
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
        loss = (
            (elementwise * masks).sum(-1) / masks.sum(-1).clamp_min(1.0)
        ).sum() / batch_size
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
    return {
        "loss": loss_value,
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
    if not torch.cuda.is_available():
        raise RuntimeError("B200 training requires CUDA")
    device = torch.device("cuda", 0)
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
    output_dir = Path(experiment["output_dir"]).resolve()
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and not bool(
        experiment.get("allow_existing_output", False)
    ):
        raise FileExistsError(f"Refusing to append to existing run: {metrics_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    records, data_files = read_records(
        config["data"]["path"], split=config["data"].get("split")
    )
    if not records:
        raise ValueError("Full DAPO dataset is empty")
    student, teacher, tokenizer, model_metadata = load_models(config, device)
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
    save_metadata(metadata, output_dir)
    optimizer, fused_optimizer = _make_optimizer(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        training,
    )
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
    rho = float(config["token_budget"]["rho"])
    token_logger = SelectedTokenLogger(
        output_dir,
        tokenizer,
        method,
        chunk_steps=int(config.get("logging", {}).get("selector_chunk_steps", 50)),
        enabled=bool(config.get("logging", {}).get("selected_tokens_enabled", True)),
    )
    training_eval_settings = config.get("training_evaluation", {})
    evaluation_steps = training_evaluation_steps(max_steps, training_eval_settings)
    if evaluation_steps:
        tqdm.write(
            f"Periodic evaluation ({training_eval_settings.get('backend', 'vllm')}): "
            + ", ".join(map(str, evaluation_steps))
        )
    initial_evaluation = None
    if should_run_training_evaluation(0, max_steps, training_eval_settings):
        tqdm.write("Evaluating the untouched base student at optimizer step 0...")
        initial_evaluation = _run_training_evaluation(
            student, tokenizer, method, 0, max_steps, config, output_dir
        )
    final_metrics: dict[str, Any] = {}
    progress = tqdm(
        range(max_steps),
        desc=f"{method.upper()}-OPD B200",
        unit="step",
        dynamic_ncols=True,
    )
    for step_index in progress:
        step, step_started = step_index + 1, time.perf_counter()
        progress.set_postfix_str("stage=rollout", refresh=True)
        torch.cuda.reset_peak_memory_stats(device)
        indices = epoch_batch_indices(len(records), batch_size, step_index, seed)
        batch_records = [records[index] for index in indices]
        encoded, _ = tokenize_prompts(batch_records, tokenizer, config["data"], device)
        rollout, rollout_time = _timed(
            device,
            generate_on_policy,
            student,
            encoded["input_ids"],
            encoded["attention_mask"],
            max_new_tokens=int(config["rollout"].get("max_new_tokens", 256)),
            temperature=float(config["rollout"].get("temperature", 1.0)),
            top_p=float(config["rollout"].get("top_p", 1.0)),
            eos_token_ids=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            seed=int(config["rollout"].get("seed", seed)) + step_index,
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
        rollout_hash = hashlib.sha256(
            rollout.input_ids.detach().cpu().numpy().tobytes()
        ).hexdigest()
        progress.set_postfix_str("stage=score-student", refresh=True)
        student_scores, student_base_time = _timed(
            device, score_original_rollout, student, rollout, True
        )
        progress.set_postfix_str("stage=score-teacher", refresh=True)
        teacher_scores, teacher_base_time = _timed(
            device, score_original_rollout, teacher, rollout, True
        )
        valid = rollout.valid_mask
        finite_or_raise("student probabilities", student_scores.probabilities[valid])
        finite_or_raise("teacher probabilities", teacher_scores.probabilities[valid])
        ta_output, ta_time = _timed(
            device,
            ta_selector.compute_scores,
            student_scores.probabilities,
            teacher_scores.probabilities,
            valid,
        )
        cross_diagnostics, prober = {}, None
        if method == "rac":
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
            )
            cross_diagnostics.update(
                correlations(ta_output.scores, primary.diagnostics, valid)
            )
            rac_probe_time, rac_branches = (
                prober.counterfactual_seconds,
                prober.evaluated_branches,
            )
        else:
            progress.set_postfix_str("stage=selector-TA", refresh=True)
            primary, selector_time, rac_probe_time, rac_branches = (
                ta_output,
                ta_time,
                0.0,
                0,
            )
        finite_or_raise(f"{method} selector", primary.scores[valid])
        selected = top_budget_mask(primary.scores, valid, rho).clone()
        expected = math.ceil(rho * int(valid.sum().item()))
        if int(selected.sum().item()) != expected:
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
        )
        if (
            bool(config.get("logging", {}).get("selected_tokens_enabled", True))
            and logged_selected != expected
        ):
            raise AssertionError(
                f"Detailed selector log wrote {logged_selected}, expected {expected}"
            )

        old_student = student_scores.sampled_log_probs.detach().clone()
        teacher_log_probs = teacher_scores.sampled_log_probs.detach().clone()
        del student_scores, teacher_scores, prober
        # Let PyTorch reuse the released probability/cache blocks for backward.
        # Emptying the CUDA allocator every step is materially slower on B200.
        if bool(training.get("empty_cuda_cache_each_step", False)):
            torch.cuda.empty_cache()
        progress.set_postfix_str("stage=train", refresh=True)
        train_metrics = _opd_train_step(
            student,
            optimizer,
            rollout,
            selected,
            old_student,
            teacher_log_probs,
            config,
            device,
        )
        checkpoint = None
        save_checkpoints = bool(training.get("save_checkpoints", True))
        save_interval = int(training.get("save_interval", 100))
        if save_checkpoints and (
            step == max_steps or (save_interval > 0 and step % save_interval == 0)
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
        cuda_sync(device)
        wall_time, valid_tokens = (
            time.perf_counter() - step_started,
            int(valid.sum().item()),
        )
        final_metrics = {
            "step": step,
            "epoch": step_index // steps_per_epoch,
            "step_in_epoch": step_index % steps_per_epoch,
            "method": method,
            "batch_size": len(indices),
            "configured_batch_size": batch_size,
            "micro_batch_size": int(training.get("micro_batch_size", batch_size)),
            "fused_optimizer": fused_optimizer,
            **train_metrics,
            "rollout_time": rollout_time,
            "student_base_scoring_time": student_base_time,
            "teacher_base_scoring_time": teacher_base_time,
            "ta_diagnostic_time": ta_time,
            "selector_time": selector_time,
            "rac_counterfactual_time": rac_probe_time,
            "rac_evaluated_branches": rac_branches,
            "wall_clock_step_time": wall_time,
            "tokens_per_second": valid_tokens / max(wall_time, 1e-12),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "selector": selector_summary(method, primary.diagnostics, valid, selected),
            "cross_selector": cross_diagnostics,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "rollout_token_sha256": rollout_hash,
        }
        # Release per-step tensors before the vLLM subprocess reserves its KV cache.
        del (
            encoded,
            rollout,
            original_rollout,
            old_student,
            teacher_log_probs,
            selected,
            primary,
            ta_output,
            valid,
        )
        if should_run_training_evaluation(step, max_steps, training_eval_settings):
            progress.set_postfix_str("stage=evaluation", refresh=True)
            periodic_evaluation = _run_training_evaluation(
                student,
                tokenizer,
                method,
                step,
                max_steps,
                config,
                output_dir,
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
        _append_jsonl(metrics_path, final_metrics)
        progress.set_postfix(
            loss=f"{train_metrics['loss']:.4f}",
            selected=f"{expected}/{valid_tokens}",
            selector=f"{selector_time:.2f}s",
            gpu=f"{final_metrics['peak_gpu_allocated_bytes'] / 2**30:.1f}GiB",
            refresh=True,
        )
        if bool(experiment.get("verbose_metrics", False)):
            tqdm.write(
                json.dumps(final_metrics, indent=2, ensure_ascii=False, allow_nan=True)
            )
    summary = {
        "status": "ok",
        "method": method,
        "steps": max_steps,
        "epochs": training.get("epochs"),
        "dataset_rows": len(records),
        "full_dataset": True,
        "last": final_metrics,
        "student_path": model_metadata["student_path"],
        "teacher_path": model_metadata["teacher_path"],
        "selector_score_dir": str((output_dir / "selector_scores").resolve()),
        "evaluation_history": str((output_dir / "eval_history.jsonl").resolve())
        if bool(training_eval_settings.get("enabled", False))
        else None,
        "initial_evaluation": initial_evaluation,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
    return summary
