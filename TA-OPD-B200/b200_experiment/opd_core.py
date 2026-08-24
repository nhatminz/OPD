from __future__ import annotations

from dataclasses import dataclass

import torch


# Reference used for the main experiment.  Keep this visible in run metadata and
# tests so a future upstream change cannot silently alter the objective.
UPSTREAM_OPD_COMMIT = "ac26e38d6f1572eb027597b48a9f4e01f6915ef8"
UPSTREAM_TOP_K_STRATEGY = "only_stu"
UPSTREAM_REWARD_WEIGHT_MODE = "student_p"
UPSTREAM_ADV_ESTIMATOR = "token_reward_direct"
UPSTREAM_LOSS_AGG_MODE = "token-mean"


@dataclass(frozen=True)
class TopKOPDReference:
    """Frozen on-policy Top-K support and rewards for one rollout batch."""

    candidate_ids: torch.Tensor
    old_student_log_probs: torch.Tensor
    teacher_log_probs: torch.Tensor
    student_weights: torch.Tensor
    advantages: torch.Tensor


def _validate_topk_tensors(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    if student_log_probs.ndim != 3:
        raise ValueError("Top-K log-probabilities must have shape [batch, time, K]")
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student and teacher Top-K log-probabilities must match")
    if student_log_probs.shape[:2] != valid_mask.shape:
        raise ValueError("Top-K log-probabilities must align with valid_mask")
    if student_log_probs.shape[-1] <= 0:
        raise ValueError("Top-K support cannot be empty")


def build_topk_opd_reference(
    candidate_ids: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> TopKOPDReference:
    """Port thunlp/OPD's only_stu + student_p token reward exactly.

    Upstream computes ``rm_scores = -(S_logp - T_on_S) * softmax(S_logp)``
    across K, then uses ``token_reward_direct`` so these rewards become the
    detached candidate-wise advantages.  Teacher probabilities are deliberately
    *not* renormalized on the student support; only the student_p weights are.
    """
    _validate_topk_tensors(student_log_probs, teacher_log_probs, valid_mask)
    if candidate_ids.shape != student_log_probs.shape:
        raise ValueError("Student candidate IDs and Top-K log-probabilities must match")
    # ``score_original_rollout`` deliberately runs under inference_mode.  A
    # dtype conversion is allowed to be a no-op (FP32 -> FP32, int64 -> int64),
    # so ``detach().float()``/``detach().long()`` can otherwise leak inference
    # tensors into the differentiable PPO gather below.  Clone with inference
    # mode explicitly disabled to materialize ordinary frozen tensors that
    # autograd is allowed to save for backward.
    with torch.inference_mode(False):
        candidate_ids = candidate_ids.detach().clone().to(dtype=torch.long)
        student = student_log_probs.detach().clone().to(dtype=torch.float32)
        teacher = teacher_log_probs.detach().clone().to(dtype=torch.float32)
        valid = valid_mask.detach().clone().to(dtype=torch.bool)
        weights = torch.softmax(student, dim=-1)
        advantages = (teacher - student) * weights
        position_mask = valid.unsqueeze(-1)
        weights = torch.where(position_mask, weights, torch.zeros_like(weights))
        advantages = torch.where(
            position_mask, advantages, torch.zeros_like(advantages)
        )
    return TopKOPDReference(
        candidate_ids=candidate_ids,
        old_student_log_probs=student,
        teacher_log_probs=teacher,
        student_weights=weights,
        advantages=advantages,
    )


def topk_reference_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_k: int,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> TopKOPDReference:
    """Correctness/reference implementation used by synthetic unit tests."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("Student/teacher logits must share shape [batch, time, vocab]")
    if student_logits.shape[:2] != valid_mask.shape:
        raise ValueError("Logits must align with valid_mask")
    if student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("OPD scoring temperatures must be positive")
    k = min(int(top_k), student_logits.shape[-1])
    if k <= 0:
        raise ValueError("top_k must be positive")
    student_all = torch.log_softmax(
        student_logits.float() / float(student_temperature), dim=-1
    )
    teacher_all = torch.log_softmax(
        teacher_logits.float() / float(teacher_temperature), dim=-1
    )
    candidate_ids = torch.topk(student_all, k=k, dim=-1).indices
    return build_topk_opd_reference(
        candidate_ids,
        student_all.gather(-1, candidate_ids),
        teacher_all.gather(-1, candidate_ids),
        valid_mask,
    )


def gather_candidate_log_probs(
    response_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    temperature: float,
    chunk_steps: int,
) -> torch.Tensor:
    """Gather differentiable FP32 log-probs without retaining full FP32 vocab."""
    if response_logits.ndim != 3 or candidate_ids.ndim != 3:
        raise ValueError("Expected logits [B,T,V] and candidate IDs [B,T,K]")
    if response_logits.shape[:2] != candidate_ids.shape[:2]:
        raise ValueError("Candidate IDs do not align with response logits")
    if temperature <= 0:
        raise ValueError("OPD scoring temperature must be positive")
    width = response_logits.shape[1]
    chunks = []
    for begin in range(0, width, max(1, int(chunk_steps))):
        end = min(begin + max(1, int(chunk_steps)), width)
        logits = response_logits[:, begin:end].float() / float(temperature)
        selected = logits.gather(-1, candidate_ids[:, begin:end])
        chunks.append(selected - torch.logsumexp(logits, dim=-1, keepdim=True))
    return torch.cat(chunks, dim=1)


def topk_candidate_ppo_loss(
    current_log_probs: torch.Tensor,
    reference: TopKOPDReference,
    *,
    clip_low: float,
    clip_high: float,
    dual_clip: float = 3.0,
) -> torch.Tensor:
    """Upstream candidate-wise PPO loss, summed over K to one loss per position."""
    if current_log_probs.shape != reference.old_student_log_probs.shape:
        raise ValueError("Current and frozen Top-K log-probabilities must match")
    if dual_clip <= 1.0:
        raise ValueError("dual_clip must be greater than 1")
    log_ratio = (current_log_probs - reference.old_student_log_probs).clamp(
        min=-20.0, max=20.0
    )
    ratio = torch.exp(log_ratio)
    advantages = reference.advantages
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * ratio.clamp(
        1.0 - float(clip_low), 1.0 + float(clip_high)
    )
    upper_clipped = torch.maximum(loss_unclipped, loss_clipped)
    dual_clipped = torch.minimum(-advantages * float(dual_clip), upper_clipped)
    candidate_loss = torch.where(advantages < 0, dual_clipped, upper_clipped)
    return candidate_loss.sum(dim=-1)


def weighted_token_sums(
    per_position_loss: torch.Tensor,
    position_weights: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return numerator/denominator for the common global token-mean."""
    if (
        per_position_loss.shape != valid_mask.shape
        or position_weights.shape != valid_mask.shape
    ):
        raise ValueError("Loss, position weights, and valid mask must have one shape")
    weights = torch.where(
        valid_mask.bool(),
        position_weights.detach().float(),
        torch.zeros_like(position_weights, dtype=torch.float32),
    )
    if bool((weights < 0).any()):
        raise ValueError("Position weights cannot be negative")
    return (per_position_loss * weights).sum(), weights.sum()


def topk_overlap_fraction(
    student_ids: torch.Tensor, teacher_ids: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Mean set-overlap fraction |S_student ∩ S_teacher| / K per position."""
    if (
        student_ids.shape != teacher_ids.shape
        or student_ids.shape[:2] != valid_mask.shape
    ):
        raise ValueError("Top-K ID tensors must match each other and valid_mask")
    overlap = (
        student_ids.unsqueeze(-1)
        .eq(teacher_ids.unsqueeze(-2))
        .any(dim=-1)
        .float()
        .mean(dim=-1)
    )
    return overlap[valid_mask.bool()]
