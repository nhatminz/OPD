from __future__ import annotations

from typing import Any


def training_evaluation_steps(
    max_steps: int, settings: dict[str, Any]
) -> tuple[int, ...]:
    """Return a deterministic eval schedule after the final step count is known."""
    if not bool(settings.get("enabled", False)):
        return ()
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    target = settings.get("target_evaluations")
    if target is not None:
        target = int(target)
        if target <= 0:
            raise ValueError("training_evaluation.target_evaluations must be positive")
        if max_steps == 0 or target == 1:
            selected = {0}
        else:
            count = min(target, max_steps + 1)
            selected = {
                round(index * max_steps / (count - 1)) for index in range(count)
            }
    else:
        interval_value = settings.get("interval_steps", 100)
        if interval_value is None:
            raise ValueError(
                "Set training_evaluation.target_evaluations or interval_steps"
            )
        interval = int(interval_value)
        if interval <= 0:
            raise ValueError("training_evaluation.interval_steps must be positive")
        selected = set(range(0, max_steps + 1, interval))
        if bool(settings.get("eval_at_end", True)):
            selected.add(max_steps)

    if not bool(settings.get("eval_at_start", True)):
        selected.discard(0)
    if not bool(settings.get("eval_at_end", True)):
        selected.discard(max_steps)
    return tuple(sorted(selected))


def should_run_training_evaluation(
    step: int, max_steps: int, settings: dict[str, Any]
) -> bool:
    return step in training_evaluation_steps(max_steps, settings)
