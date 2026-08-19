from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .base import SelectorOutput, scatter_valid


class RACSelector:
    """Recoverable accessible corrective-mass selector: s_RAC = Delta*A*F."""

    def __init__(
        self,
        top_k: int = 16,
        branch_m: int = 2,
        eps: float = 1e-8,
        delta_mode: str = "full_vocab",
    ):
        if delta_mode not in {"full_vocab", "approx_topk"}:
            raise ValueError(f"Unknown RAC delta mode {delta_mode!r}")
        self.top_k = int(top_k)
        self.branch_m = int(branch_m)
        self.eps = float(eps)
        self.delta_mode = delta_mode

    @torch.inference_mode()
    def compute_scores(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        valid_mask: torch.Tensor,
        cplus_probe: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        student_topk: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> SelectorOutput:
        if (
            student_probs.shape != teacher_probs.shape
            or student_probs.shape[:2] != valid_mask.shape
        ):
            raise ValueError("RAC received incompatible probability/mask shapes")
        flat_valid = valid_mask.reshape(-1)
        flat_p = student_probs.reshape(-1, student_probs.shape[-1])
        flat_q = teacher_probs.reshape(-1, teacher_probs.shape[-1])
        p, q = (
            (flat_p, flat_q)
            if bool(flat_valid.all())
            else (flat_p[flat_valid], flat_q[flat_valid])
        )
        n_valid = p.shape[0]
        if n_valid == 0:
            zeros = torch.zeros_like(valid_mask, dtype=torch.float32)
            return SelectorOutput(
                zeros, {name: zeros for name in ("Delta", "A", "F", "B", "s_RAC")}
            )

        k = min(self.top_k, p.shape[-1])
        if student_topk is None:
            p_top_values, p_top_ids = torch.topk(p, k=k, dim=-1)
        else:
            p_top_values, p_top_ids = student_topk
            expected = (n_valid, k)
            if p_top_values.shape != expected or p_top_ids.shape != expected:
                raise ValueError(
                    f"Precomputed student top-K must have shape {expected}, got "
                    f"{p_top_values.shape} and {p_top_ids.shape}"
                )
        corrective_top = (q.gather(-1, p_top_ids) - p_top_values).clamp_min_(0.0)
        if self.delta_mode == "full_vocab":
            delta = (q - p).clamp_min_(0.0).sum(dim=-1)
        else:
            q_top_ids = torch.topk(q, k=k, dim=-1).indices
            union_ids = torch.cat((p_top_ids, q_top_ids), dim=-1)
            unique = ~torch.tril(
                union_ids.unsqueeze(-1).eq(union_ids.unsqueeze(-2)), diagonal=-1
            ).any(dim=-1)
            delta = 0.5 * (
                (q.gather(-1, union_ids) - p.gather(-1, union_ids)).abs() * unique
            ).sum(dim=-1)

        accessible_mass = corrective_top.sum(dim=-1)
        accessibility = (accessible_mass / (delta + self.eps)).clamp_(0.0, 1.0)
        m = min(self.branch_m, k)
        candidate_weights, candidate_offsets = torch.topk(corrective_top, k=m, dim=-1)
        candidate_ids = p_top_ids.gather(-1, candidate_offsets)
        candidate_mask = candidate_weights > 0
        flat_mask = candidate_mask.reshape(-1)
        branch_rows = (
            torch.arange(n_valid, device=p.device)
            .unsqueeze(1)
            .expand(-1, m)
            .reshape(-1)[flat_mask]
        )
        branch_tokens = candidate_ids.reshape(-1)[flat_mask]
        cplus_flat = torch.zeros(n_valid * m, dtype=torch.float32, device=p.device)
        if branch_tokens.numel() > 0:
            probed = cplus_probe(branch_rows, branch_tokens).float()
            if probed.shape != branch_tokens.shape:
                raise ValueError(
                    f"cplus_probe returned {probed.shape}; expected {branch_tokens.shape}"
                )
            if probed.requires_grad or probed.grad_fn is not None:
                raise AssertionError(
                    "RAC counterfactual probes must not carry an autograd graph"
                )
            cplus_flat[flat_mask] = probed.clamp(0.0, 1.0)
        cplus = cplus_flat.view(n_valid, m)
        weight_sum = (candidate_weights * candidate_mask).sum(dim=-1)
        future = (candidate_weights * cplus * candidate_mask).sum(dim=-1) / (
            weight_sum + self.eps
        )
        future = torch.where(
            candidate_mask.any(dim=-1) & (delta > self.eps),
            future,
            torch.zeros_like(future),
        )
        recoverability = accessibility * future
        score = delta * recoverability
        values: dict[str, Any] = {
            "Delta": delta,
            "A": accessibility,
            "F": future,
            "B": recoverability,
            "s_RAC": score,
            "positive_reachable_candidates": candidate_mask.sum(dim=-1).float(),
            "evaluated_branches": candidate_mask.sum(dim=-1).float(),
            "mean_Cplus": (cplus * candidate_mask).sum(dim=-1)
            / candidate_mask.sum(dim=-1).clamp_min(1),
            "candidate_ids": candidate_ids,
            "candidate_weights": candidate_weights,
            "delta_mode": self.delta_mode,
        }
        diagnostics = {
            key: scatter_valid(value, valid_mask)
            if torch.is_tensor(value) and value.ndim == 1
            else value
            for key, value in values.items()
        }
        return SelectorOutput(diagnostics["s_RAC"], diagnostics)
