from __future__ import annotations

from pathlib import Path
from typing import Any


BASE_TAGS = {
    "train/loss": "train_loss",
    "train/grad_norm": "grad_norm",
    "train/learning_rate": "lr",
    "opd/advantage_abs_mean": "opd_advantage_abs_mean",
    "opd/teacher_student_logprob_gap": "opd_teacher_student_logprob_gap",
    "opd/clip_fraction": "opd_clip_fraction",
    "rollout/response_length_mean": "mean_response_length",
    "rollout/eos_fraction": "eos_fraction",
    "system/step_time": "wall_clock_step_time",
    "system/rollout_time": "rollout_time",
    "system/tokens_per_second": "tokens_per_second",
    "system/peak_vram_gb": "peak_gpu_allocated_gb",
}


def production_tensorboard_metrics(
    metrics: dict[str, Any], method: str
) -> dict[str, float]:
    """Select only the intentionally small, globally reduced production set."""
    selected = {tag: float(metrics[field]) for tag, field in BASE_TAGS.items()}
    selector = metrics.get("selector", {})
    if method == "ta":
        selected["ta/selected_token_fraction"] = float(selector["selected_fraction"])
    elif method == "rac":
        valid_tokens = max(int(selector["valid_tokens"]), 1)
        selected["rac/effective_token_fraction"] = float(
            selector["effective_sample_size"] / valid_tokens
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
