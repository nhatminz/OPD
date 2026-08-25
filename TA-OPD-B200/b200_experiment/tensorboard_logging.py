from __future__ import annotations

from pathlib import Path
from typing import Any


BASE_TAGS = {
    "train/loss": "train_loss",
    "train/grad_norm": "grad_norm",
    "train/learning_rate": "lr",
    "train/global_step": "step",
    "distillation/topk_opd_loss": "base_topk_opd_loss",
    "distillation/student_entropy": "student_entropy",
    "distillation/teacher_entropy": "teacher_entropy",
    "distillation/entropy_gap": "entropy_gap",
    "distillation/topk_overlap_ratio": "student_teacher_topk_overlap_ratio",
    "distillation/topk_student_mass": "topk_student_mass",
    "distillation/topk_teacher_mass": "topk_teacher_mass",
    "distillation/topk_divergence_proxy_mean": "topk_divergence_proxy_mean",
    "distillation/topk_divergence_proxy_min": "topk_divergence_proxy_min",
    "distillation/topk_divergence_proxy_max": "topk_divergence_proxy_max",
    "optimization/ratio_mean": "opd_ratio_mean",
    "optimization/ratio_min": "opd_ratio_min",
    "optimization/ratio_max": "opd_ratio_max",
    "optimization/clip_fraction": "opd_clip_fraction",
    "optimization/advantage_mean": "opd_advantage_mean",
    "optimization/advantage_abs_mean": "opd_advantage_abs_mean",
    "optimization/advantage_min": "opd_advantage_min",
    "optimization/advantage_max": "opd_advantage_max",
    "opd/advantage_abs_mean": "opd_advantage_abs_mean",
    "opd/teacher_student_logprob_gap": "opd_teacher_student_logprob_gap",
    "opd/clip_fraction": "opd_clip_fraction",
    "rollout/response_length_mean": "mean_response_length",
    "rollout/response_length_min": "min_response_length",
    "rollout/response_length_max": "max_response_length",
    "rollout/response_clip_ratio": "response_clip_ratio",
    "rollout/tokens_per_second": "generated_tokens_per_second",
    "rollout/eos_fraction": "eos_fraction",
    "system/step_time": "wall_clock_step_time",
    "system/step_time_sec": "wall_clock_step_time",
    "system/rollout_time": "rollout_time",
    "system/tokens_per_second": "tokens_per_second",
    "system/peak_vram_gb": "peak_gpu_allocated_gb",
    "system/gpu_memory_allocated_gb": "peak_gpu_allocated_gb",
    "system/gpu_memory_reserved_gb": "peak_gpu_reserved_gb",
}

TA_TAGS = {
    "ta/D_mean": ("D", "mean"),
    "ta/C_mean": ("C", "mean"),
    "ta/teachability_mean": ("s_TA", "mean"),
    "ta/teachability_std": ("s_TA", "std"),
    "ta/selected_fraction": ("selected_fraction",),
}

RAC_TAGS = {
    "rac/local_teachability_mean": ("g", "mean"),
    "rac/alignment_mean": ("alignment", "mean"),
    "rac/V_mean": ("V", "mean"),
    "rac/V_std": ("V", "std"),
    "rac/weight_mean": ("w", "mean"),
    "rac/weight_std": ("w", "std"),
    "rac/weight_min": ("w", "min"),
    "rac/weight_max": ("w", "max"),
}


def _selector_value(selector: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = selector
    for key in path:
        value = value[key]
    return float(value)


def production_tensorboard_metrics(
    metrics: dict[str, Any], method: str
) -> dict[str, float]:
    """Select globally reduced production diagnostics for TensorBoard."""
    selected = {tag: float(metrics[field]) for tag, field in BASE_TAGS.items()}
    selector = metrics.get("selector", {})
    if method == "ta":
        selected["ta/selected_token_fraction"] = float(selector["selected_fraction"])
        selected.update(
            {
                tag: _selector_value(selector, path)
                for tag, path in TA_TAGS.items()
            }
        )
    elif method == "rac":
        valid_tokens = max(int(selector["valid_tokens"]), 1)
        selected["rac/effective_token_fraction"] = float(
            selector["effective_sample_size"] / valid_tokens
        )
        selected.update(
            {
                tag: _selector_value(selector, path)
                for tag, path in RAC_TAGS.items()
            }
        )
    sanity = metrics.get("vllm_logprob_sanity", {})
    if bool(sanity.get("enabled", False)):
        selected["debug/vllm_hf_logprob_mae"] = float(sanity["mean_abs_error"])
    return selected


class TensorBoardLogger:
    def __init__(
        self,
        output_dir: Path,
        settings: dict[str, Any],
        *,
        enabled: bool,
        resume_step: int,
    ):
        self.writer = None
        self.interval = max(1, int(settings.get("log_interval", 1)))
        if not enabled or not bool(settings.get("enabled", True)):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard logging is enabled but tensorboard is not installed; "
                "install requirements.txt"
            ) from error
        configured = settings.get("log_dir", "tensorboard")
        log_dir = Path(configured)
        if not log_dir.is_absolute():
            log_dir = output_dir / log_dir
        self.writer = SummaryWriter(
            log_dir=str(log_dir.resolve()),
            purge_step=(resume_step + 1 if resume_step > 0 else None),
            max_queue=10,
            flush_secs=int(settings.get("flush_secs", 30)),
        )

    def write(self, step: int, metrics: dict[str, Any], method: str) -> None:
        if self.writer is None or int(step) % self.interval != 0:
            return
        for tag, value in production_tensorboard_metrics(metrics, method).items():
            self.writer.add_scalar(tag, value, global_step=int(step))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
