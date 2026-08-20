from __future__ import annotations

from typing import Any

import torch

from .selectors import top_budget_mask


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.float(), y.float()
    if x.numel() < 2:
        return float("nan")
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return (
        float("nan")
        if denominator.item() == 0
        else float((x * y).sum().div(denominator).item())
    )


def _ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32, device=x.device)
    return ranks


def correlations(ta_scores, rac_diagnostics, valid_mask) -> dict[str, float]:
    ta = ta_scores[valid_mask]
    result = {}
    for name in ("g", "V", "w"):
        target = rac_diagnostics[name][valid_mask]
        result[f"pearson_sTA_{name}"] = _pearson(ta, target)
        result[f"spearman_sTA_{name}"] = _pearson(_ranks(ta), _ranks(target))
    for ratio in (0.05, 0.10):
        ta_mask = top_budget_mask(ta_scores, valid_mask, ratio)
        rac_mask = top_budget_mask(rac_diagnostics["V"], valid_mask, ratio)
        denominator = max(int(ta_mask.sum().item()), 1)
        result[f"top_{int(ratio * 100)}pct_overlap"] = float(
            (ta_mask & rac_mask).sum().item() / denominator
        )
    return result


def tensor_summary(
    values: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> dict[str, float]:
    if valid_mask is not None and values.shape == valid_mask.shape:
        values = values[valid_mask]
    finite = values.detach().float()
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    quantiles = torch.quantile(
        finite, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=finite.device)
    )
    return {
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        **{
            name: float(value)
            for name, value in zip(
                ("q05", "q25", "q50", "q75", "q95"), quantiles
            )
        },
    }


def selector_summary(
    method: str,
    diagnostics: dict[str, Any],
    valid_mask: torch.Tensor,
    selected: torch.Tensor,
):
    keys = (
        ("D", "C", "D_norm", "C_norm", "s_TA")
        if method == "ta"
        else (
            "g",
            "alignment",
            "R",
            "M",
            "V",
            "z",
            "w",
        )
    )
    result: dict[str, Any] = {}
    for key in keys:
        value = diagnostics.get(key)
        if torch.is_tensor(value):
            result[key] = tensor_summary(value, valid_mask)
    valid_count = int(valid_mask.sum().item())
    selected_count = int(selected.sum().item())
    result.update(
        valid_tokens=valid_count,
        selected_tokens=selected_count,
        selected_fraction=selected_count / max(valid_count, 1),
    )
    if method == "ta" and selected_count:
        result["selection_threshold"] = float(diagnostics["s_TA"][selected].min())
    if method == "rac" and "w" in diagnostics:
        weights = diagnostics["w"][valid_mask].detach().float()
        weight_sum = weights.sum()
        result.update(
            effective_token_weight_mass=float(weight_sum / max(valid_count, 1)),
            effective_sample_size=float(
                weight_sum.square() / weights.square().sum().clamp_min(1e-12)
            ),
        )
    return result


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        count = int((~torch.isfinite(tensor)).sum().item())
        raise FloatingPointError(f"{name} contains {count} non-finite values")
