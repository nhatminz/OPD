from __future__ import annotations

import torch

from .base import SelectorOutput, robust_quantile_normalize, scatter_valid


class TASelector:
    """Faithful tensor implementation of the released TA-OPD Dlearn selector."""

    def __init__(
        self,
        top_k: int = 16,
        q_low: float = 0.05,
        q_high: float = 0.95,
        eps: float = 1e-12,
    ):
        self.top_k = int(top_k)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = float(eps)

    @torch.inference_mode()
    def compute_scores(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        normalize: bool = True,
    ):
        if student_probs.shape != teacher_probs.shape:
            raise ValueError(
                "Student and teacher probability tensors must have identical shape"
            )
        if student_probs.shape[:2] != valid_mask.shape:
            raise ValueError("Probability leading dimensions must match valid_mask")
        flat_valid = valid_mask.reshape(-1)
        flat_p = student_probs.reshape(-1, student_probs.shape[-1])
        flat_q = teacher_probs.reshape(-1, teacher_probs.shape[-1])
        p, q = (
            (flat_p, flat_q)
            if bool(flat_valid.all())
            else (flat_p[flat_valid], flat_q[flat_valid])
        )
        if p.numel() == 0:
            zeros = torch.zeros_like(valid_mask, dtype=torch.float32)
            return SelectorOutput(
                zeros,
                {
                    **{name: zeros for name in ("D", "C", "D_norm", "C_norm", "s_TA")},
                    "_student_top_values": p.new_empty((0, 0)),
                    "_student_top_ids": torch.empty(
                        (0, 0), dtype=torch.long, device=p.device
                    ),
                },
            )

        k = min(self.top_k, p.shape[-1])
        p_top_values, p_top_ids = torch.topk(p, k=k, dim=-1)
        _, q_top_ids = torch.topk(q, k=k, dim=-1)
        union_ids = torch.cat((p_top_ids, q_top_ids), dim=-1)
        equal = union_ids.unsqueeze(-1).eq(union_ids.unsqueeze(-2))
        unique = ~torch.tril(equal, diagonal=-1).any(dim=-1)
        in_student = union_ids.unsqueeze(-1).eq(p_top_ids.unsqueeze(-2)).any(dim=-1)
        in_teacher = union_ids.unsqueeze(-1).eq(q_top_ids.unsqueeze(-2)).any(dim=-1)
        p_union = p.gather(-1, union_ids) * in_student * unique
        q_union = q.gather(-1, union_ids) * in_teacher * unique
        p_norm = (
            p_union / p_union.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        ).clamp_min(self.eps)
        q_norm = (
            q_union / q_union.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        ).clamp_min(self.eps)
        disagreement = (q_norm * (q_norm.log() - p_norm.log()) * unique).sum(dim=-1)

        p_ids_in_teacher = (
            p_top_ids.unsqueeze(-1).eq(q_top_ids.unsqueeze(-2)).any(dim=-1)
        )
        compatibility = (
            (q.gather(-1, p_top_ids) * p_ids_in_teacher).sum(dim=-1).clamp_(0.0, 1.0)
        )
        if normalize:
            d_norm = robust_quantile_normalize(
                disagreement, self.q_low, self.q_high, self.eps
            )
            c_norm = robust_quantile_normalize(
                compatibility, self.q_low, self.q_high, self.eps
            )
        else:
            d_norm = torch.zeros_like(disagreement)
            c_norm = torch.zeros_like(compatibility)
        score = d_norm * c_norm
        diagnostics = {
            "D": scatter_valid(disagreement, valid_mask),
            "C": scatter_valid(compatibility, valid_mask),
            "D_norm": scatter_valid(d_norm, valid_mask),
            "C_norm": scatter_valid(c_norm, valid_mask),
            "s_TA": scatter_valid(score, valid_mask),
            "compatibility_convention": "upstream_topk_intersection",
            # RAC consumes this exact already-computed student top-K. Keeping
            # these private tensors avoids a second full top-K over the vocab
            # without changing either selector's score.
            "_student_top_values": p_top_values,
            "_student_top_ids": p_top_ids,
        }
        return SelectorOutput(diagnostics["s_TA"], diagnostics)
