from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SelectorOutput:
    scores: torch.Tensor
    diagnostics: dict[str, Any]


def robust_quantile_normalize(
    values: torch.Tensor,
    q_low: float = 0.05,
    q_high: float = 0.95,
    eps: float = 1e-12,
) -> torch.Tensor:
    values = values.float()
    finite_mask = torch.isfinite(values)
    if not finite_mask.any():
        return torch.zeros_like(values)
    finite = values[finite_mask]
    low = torch.quantile(finite, q_low, interpolation="linear")
    high = torch.quantile(finite, q_high, interpolation="linear")
    denominator = high - low
    if denominator.abs().item() < eps:
        return torch.zeros_like(values)
    normalized = ((values - low) / denominator).clamp_(0.0, 1.0)
    return torch.where(finite_mask, normalized, torch.zeros_like(normalized))


def scatter_valid(
    values: torch.Tensor, valid_mask: torch.Tensor, fill: float = 0.0
) -> torch.Tensor:
    output = torch.full(
        valid_mask.shape, fill, dtype=values.dtype, device=values.device
    )
    output[valid_mask] = values
    return output
