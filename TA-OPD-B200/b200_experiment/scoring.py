from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask.eq(0), 0)


def _sample_top_p(
    probabilities: torch.Tensor, top_p: float, generator: torch.Generator
) -> torch.Tensor:
    if top_p >= 1.0:
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
    sorted_probs, sorted_ids = torch.sort(probabilities, dim=-1, descending=True)
    remove = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    offset = torch.multinomial(sorted_probs, 1, generator=generator)
    return sorted_ids.gather(-1, offset).squeeze(-1)


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_ids: torch.Tensor
    valid_mask: torch.Tensor
    rollout_log_probs: torch.Tensor
    prompt_width: int


@torch.inference_mode()
def generate_on_policy(
    model,
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_ids: int | list[int],
    pad_token_id: int,
    seed: int,
    sample_seed_offset: int = 0,
) -> RolloutBatch:
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    device = prompt_ids.device
    generators = []
    for row in range(prompt_ids.shape[0]):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + int(sample_seed_offset) + row)
        generators.append(generator)
    model.eval()
    prompt_width = prompt_ids.shape[1]
    attention = prompt_attention_mask.clone()
    output = model(
        input_ids=prompt_ids,
        attention_mask=attention,
        position_ids=position_ids_from_mask(attention),
        use_cache=True,
        return_dict=True,
    )
    cache = output.past_key_values
    next_logits = output.logits[:, -1, :]
    active = torch.ones(prompt_ids.shape[0], dtype=torch.bool, device=device)
    response_tokens, response_valid, response_log_probs = [], [], []
    full_ids = prompt_ids
    for _ in range(max_new_tokens):
        logits = next_logits.float()
        if temperature <= 0:
            next_token = logits.argmax(dim=-1)
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.stack(
                [
                    _sample_top_p(
                        probabilities[row : row + 1], top_p, generators[row]
                    ).squeeze(0)
                    for row in range(probabilities.shape[0])
                ]
            )
        next_token = torch.where(
            active, next_token, torch.full_like(next_token, pad_token_id)
        )
        log_prob = (
            F.log_softmax(logits, dim=-1).gather(-1, next_token[:, None]).squeeze(-1)
        )
        response_tokens.append(next_token)
        response_valid.append(active.clone())
        response_log_probs.append(
            torch.where(active, log_prob, torch.zeros_like(log_prob))
        )
        full_ids = torch.cat((full_ids, next_token[:, None]), dim=1)
        attention = torch.cat((attention, active.long()[:, None]), dim=1)
        is_eos = torch.zeros_like(active)
        for eos_id in eos_token_ids:
            is_eos |= next_token.eq(int(eos_id))
        active &= ~is_eos
        if not active.any():
            break
        absolute_position = full_ids.shape[1] - 1
        incremental_position = attention.sum(dim=-1, keepdim=True) - 1
        output = model(
            input_ids=next_token[:, None],
            attention_mask=attention,
            position_ids=incremental_position,
            cache_position=torch.tensor([absolute_position], device=device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        next_logits = output.logits[:, -1, :]
    del output, cache
    return RolloutBatch(
        input_ids=full_ids,
        attention_mask=attention,
        response_ids=torch.stack(response_tokens, dim=1),
        valid_mask=torch.stack(response_valid, dim=1),
        rollout_log_probs=torch.stack(response_log_probs, dim=1),
        prompt_width=prompt_width,
    )


@dataclass
class BaseScores:
    response_logits: torch.Tensor | None
    log_normalizers: torch.Tensor
    sampled_log_probs: torch.Tensor
    cache: object | None = None


@torch.inference_mode()
def score_original_rollout(
    model,
    rollout: RolloutBatch,
    keep_cache: bool = False,
    score_chunk_steps: int = 128,
    retain_response_logits: bool = True,
) -> BaseScores:
    """Score original positions without retaining full-vocabulary FP32 probs.

    Pure OPD needs only sampled-token log-probabilities, so callers may avoid
    retaining a second full-vocabulary response-logit copy. TA/RAC retain the
    BF16 logits because their local teachability computation requires top-K
    student/teacher distributions.
    """
    model.eval()
    output = model(
        input_ids=rollout.input_ids,
        attention_mask=rollout.attention_mask,
        position_ids=position_ids_from_mask(rollout.attention_mask),
        use_cache=keep_cache,
        return_dict=True,
    )
    start = rollout.prompt_width - 1
    width = rollout.response_ids.shape[1]
    # Keep model dtype (BF16 on B200); FP32 reductions are chunked over time.
    response_logits_view = output.logits[:, start : start + width, :]
    normalizer_chunks, sampled_chunks = [], []
    chunk_steps = max(1, int(score_chunk_steps))
    for begin in range(0, width, chunk_steps):
        end = min(begin + chunk_steps, width)
        chunk = response_logits_view[:, begin:end].float()
        normalizers = torch.logsumexp(chunk, dim=-1)
        sampled = (
            chunk.gather(-1, rollout.response_ids[:, begin:end].unsqueeze(-1)).squeeze(
                -1
            )
            - normalizers
        )
        normalizer_chunks.append(normalizers)
        sampled_chunks.append(sampled)
    response_logits = (
        response_logits_view.detach().clone() if retain_response_logits else None
    )
    return BaseScores(
        response_logits,
        torch.cat(normalizer_chunks, dim=1),
        torch.cat(sampled_chunks, dim=1),
        output.past_key_values if keep_cache else None,
    )
