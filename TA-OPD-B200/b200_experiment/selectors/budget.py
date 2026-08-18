from __future__ import annotations

import math

import torch


def top_budget_mask(
    scores: torch.Tensor, valid_response_mask: torch.Tensor, rho: float
) -> torch.Tensor:
    """Globally keep exactly ceil(rho*N_valid), with stable flat-index tie breaking."""
    if scores.shape != valid_response_mask.shape:
        raise ValueError(
            f"scores/mask shape mismatch: {scores.shape} != {valid_response_mask.shape}"
        )
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    valid_flat = valid_response_mask.reshape(-1).bool()
    valid_indices = valid_flat.nonzero(as_tuple=False).squeeze(-1)
    result = torch.zeros_like(valid_flat)
    if valid_indices.numel() == 0:
        return result.reshape_as(valid_response_mask)
    budget = int(math.ceil(rho * valid_indices.numel()))
    valid_scores = torch.nan_to_num(
        scores.reshape(-1)[valid_indices].float(), nan=-torch.inf
    )
    order = torch.argsort(valid_scores, descending=True, stable=True)
    result[valid_indices[order[:budget]]] = True
    if int(result.sum().item()) != budget:
        raise AssertionError(
            f"Expected {budget} selected positions, got {int(result.sum().item())}"
        )
    return result.reshape_as(valid_response_mask)
