from __future__ import annotations

import torch

from .base import SelectorOutput


class OPDSelector:
    """Pure OPD: supervise every valid response token with uniform weight."""

    @torch.no_grad()
    def compute_scores(self, valid_mask: torch.Tensor) -> SelectorOutput:
        if valid_mask.ndim != 2:
            raise ValueError("Pure OPD expects a rank-2 [batch, time] valid mask")
        valid = valid_mask.bool()
        weights = valid.float()
        if weights.requires_grad or weights.grad_fn is not None:
            raise AssertionError("Pure OPD weights must be detached")
        return SelectorOutput(
            weights,
            {
                "w": weights,
                "weighting": "uniform_all_valid_response_tokens",
            },
        )
