from __future__ import annotations

import torch

from .base import SelectorOutput, robust_quantile_normalize, scatter_valid


def _validate_inputs(
    g: torch.Tensor, alignment: torch.Tensor, valid_mask: torch.Tensor, gamma: float
) -> None:
    if g.shape != alignment.shape or g.shape != valid_mask.shape:
        raise ValueError(
            "g, alignment, and valid_mask must have identical [batch, time] shapes"
        )
    if g.ndim != 2:
        raise ValueError("Bellman-RAC expects rank-2 [batch, time] tensors")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")


@torch.no_grad()
def bellman_reference_scan(
    g: torch.Tensor,
    alignment: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clear correctness-first recurrence, intentionally not used by default."""
    _validate_inputs(g, alignment, valid_mask, gamma)
    g = g.detach().float()
    alignment = alignment.detach().float()
    valid_mask = valid_mask.bool()
    returns = torch.zeros_like(g)
    masses = torch.zeros_like(g)
    for row in range(g.shape[0]):
        next_return = g.new_zeros(())
        next_mass = g.new_zeros(())
        for position in range(g.shape[1] - 1, -1, -1):
            if bool(valid_mask[row, position]):
                coefficient = float(gamma) * alignment[row, position]
                next_return = g[row, position] + coefficient * next_return
                next_mass = 1.0 + coefficient * next_mass
                returns[row, position] = next_return
                masses[row, position] = next_mass
            else:
                # An invalid position is a hard trajectory boundary.
                next_return = g.new_zeros(())
                next_mass = g.new_zeros(())
    values = torch.where(
        valid_mask, returns / (masses + float(eps)), torch.zeros_like(returns)
    )
    return returns, masses, values


@torch.no_grad()
def bellman_parallel_scan(
    g: torch.Tensor,
    alignment: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized O(log T) suffix scan of Bellman affine recurrences.

    Each position represents the affine map ``x -> g_t + c_t*x``. Affine-map
    composition is associative, so a Hillis-Steele suffix scan replaces a
    Python loop over T response tokens with ceil(log2(T)) batched GPU passes.
    Invalid positions have zero continuation and therefore reset trajectories.
    """
    _validate_inputs(g, alignment, valid_mask, gamma)
    valid = valid_mask.bool()
    returns = torch.where(valid, g.detach().float(), torch.zeros_like(g).float())
    masses = valid.float()
    coefficients = torch.where(
        valid,
        float(gamma) * alignment.detach().float(),
        torch.zeros_like(alignment).float(),
    )
    offset = 1
    width = g.shape[1]
    while offset < width:
        left_c = coefficients[:, :-offset]
        next_returns = returns.clone()
        next_masses = masses.clone()
        next_coefficients = coefficients.clone()
        next_returns[:, :-offset] = returns[:, :-offset] + left_c * returns[:, offset:]
        next_masses[:, :-offset] = masses[:, :-offset] + left_c * masses[:, offset:]
        next_coefficients[:, :-offset] = left_c * coefficients[:, offset:]
        returns, masses, coefficients = (
            next_returns,
            next_masses,
            next_coefficients,
        )
        offset *= 2
    values = torch.where(
        valid, returns / (masses + float(eps)), torch.zeros_like(returns)
    )
    return returns, masses, values


class RACSelector:
    """Bellman-RAC: future TA teachability and continuous all-token weights."""

    def __init__(
        self,
        gamma: float = 0.995,
        w_min: float = 0.10,
        beta: float = 2.0,
        q_low: float = 0.05,
        q_high: float = 0.95,
        eps: float = 1e-8,
        scan_backend: str = "parallel",
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not 0.0 < w_min <= 1.0:
            raise ValueError(f"w_min must be in (0, 1], got {w_min}")
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        if scan_backend not in {"parallel", "reference"}:
            raise ValueError(f"Unknown Bellman scan backend {scan_backend!r}")
        self.gamma = float(gamma)
        self.w_min = float(w_min)
        self.beta = float(beta)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = float(eps)
        self.scan_backend = scan_backend

    @torch.no_grad()
    def compute_scores(
        self,
        local_teachability: torch.Tensor,
        student_sampled_log_probs: torch.Tensor,
        teacher_sampled_log_probs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> SelectorOutput:
        if (
            local_teachability.shape != valid_mask.shape
            or student_sampled_log_probs.shape != valid_mask.shape
            or teacher_sampled_log_probs.shape != valid_mask.shape
        ):
            raise ValueError("Bellman-RAC received incompatible score/mask shapes")
        valid = valid_mask.bool()
        g = torch.where(
            valid,
            local_teachability.detach().float().clamp(0.0, 1.0),
            torch.zeros_like(local_teachability).float(),
        )
        log_ratio = (
            teacher_sampled_log_probs.detach().float()
            - student_sampled_log_probs.detach().float()
        )
        alignment = torch.where(
            valid,
            torch.exp(log_ratio.clamp(min=-20.0, max=0.0)),
            torch.zeros_like(log_ratio),
        )
        scan = (
            bellman_parallel_scan
            if self.scan_backend == "parallel"
            else bellman_reference_scan
        )
        returns, masses, values = scan(g, alignment, valid, self.gamma, self.eps)
        if normalize:
            z_valid = robust_quantile_normalize(
                values[valid], self.q_low, self.q_high, self.eps
            )
            z = scatter_valid(z_valid, valid)
            weights = scatter_valid(
                self.w_min + (1.0 - self.w_min) * z_valid.pow(self.beta), valid
            )
        else:
            z = torch.zeros_like(values)
            weights = torch.zeros_like(values)
        diagnostics = {
            "g": g,
            "alignment": alignment,
            "R": returns,
            "M": masses,
            "V": values,
            "z": z,
            "w": weights,
            "gamma": self.gamma,
            "w_min": self.w_min,
            "beta": self.beta,
            "scan_backend": self.scan_backend,
        }
        for value in (g, alignment, returns, masses, values, z, weights):
            if value.requires_grad or value.grad_fn is not None:
                raise AssertionError("Bellman-RAC statistics must be detached")
        return SelectorOutput(weights, diagnostics)
