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
    top_k_ids: torch.Tensor | None = None
    top_k_log_probs: torch.Tensor | None = None
    candidate_log_probs: torch.Tensor | None = None
    cache: object | None = None


@torch.inference_mode()
def score_original_rollout(
    model,
    rollout: RolloutBatch,
    keep_cache: bool = False,
    score_chunk_steps: int = 128,
    retain_response_logits: bool = True,
    top_k: int = 0,
    candidate_ids: torch.Tensor | None = None,
    temperature: float = 1.0,
    micro_batch_size: int | None = None,
) -> BaseScores:
    """Score sampled and Top-K actions without retaining full-vocabulary FP32 probs."""
    if temperature <= 0:
        raise ValueError("Scoring temperature must be positive")
    model.eval()
    start = rollout.prompt_width - 1
    width = rollout.response_ids.shape[1]
    batch_size = rollout.input_ids.shape[0]
    score_micro_batch = (
        batch_size if micro_batch_size is None else max(1, int(micro_batch_size))
    )
    if keep_cache and score_micro_batch < batch_size:
        raise ValueError("keep_cache is incompatible with scoring micro-batches")
    normalizer_batches, sampled_batches = [], []
    top_id_batches, top_log_prob_batches, candidate_log_prob_batches = [], [], []
    response_logit_batches = []
    cache = None
    for batch_begin in range(0, batch_size, score_micro_batch):
        batch_end = min(batch_begin + score_micro_batch, batch_size)
        input_ids = rollout.input_ids[batch_begin:batch_end]
        attention_mask = rollout.attention_mask[batch_begin:batch_end]
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids_from_mask(attention_mask),
            use_cache=keep_cache,
            return_dict=True,
        )
        # Keep model dtype (BF16 on B200); FP32 reductions are chunked over time.
        response_logits_view = output.logits[:, start : start + width, :]
        normalizer_chunks, sampled_chunks = [], []
        top_id_chunks, top_log_prob_chunks, candidate_log_prob_chunks = [], [], []
        chunk_steps = max(1, int(score_chunk_steps))
        for begin in range(0, width, chunk_steps):
            end = min(begin + chunk_steps, width)
            raw_chunk = response_logits_view[:, begin:end].float()
            normalizers = torch.logsumexp(raw_chunk, dim=-1)
            chunk = raw_chunk / float(temperature)
            scaled_normalizers = torch.logsumexp(chunk, dim=-1)
            sampled = (
                chunk.gather(
                    -1,
                    rollout.response_ids[
                        batch_begin:batch_end, begin:end
                    ].unsqueeze(-1),
                ).squeeze(-1)
                - scaled_normalizers
            )
            normalizer_chunks.append(normalizers)
            sampled_chunks.append(sampled)
            if top_k > 0:
                k = min(int(top_k), chunk.shape[-1])
                top_values, top_ids = torch.topk(chunk, k=k, dim=-1)
                top_id_chunks.append(top_ids)
                top_log_prob_chunks.append(
                    top_values - scaled_normalizers.unsqueeze(-1)
                )
            if candidate_ids is not None:
                ids = candidate_ids[batch_begin:batch_end, begin:end]
                candidate_log_prob_chunks.append(
                    chunk.gather(-1, ids) - scaled_normalizers.unsqueeze(-1)
                )
        normalizer_batches.append(torch.cat(normalizer_chunks, dim=1))
        sampled_batches.append(torch.cat(sampled_chunks, dim=1))
        if top_id_chunks:
            top_id_batches.append(torch.cat(top_id_chunks, dim=1))
            top_log_prob_batches.append(torch.cat(top_log_prob_chunks, dim=1))
        if candidate_log_prob_chunks:
            candidate_log_prob_batches.append(
                torch.cat(candidate_log_prob_chunks, dim=1)
            )
        if retain_response_logits:
            response_logit_batches.append(response_logits_view.detach().clone())
        if keep_cache:
            cache = output.past_key_values
        del output, response_logits_view
    return BaseScores(
        response_logits=(
            torch.cat(response_logit_batches, dim=0)
            if response_logit_batches
            else None
        ),
        log_normalizers=torch.cat(normalizer_batches, dim=0),
        sampled_log_probs=torch.cat(sampled_batches, dim=0),
        top_k_ids=torch.cat(top_id_batches, dim=0) if top_id_batches else None,
        top_k_log_probs=(
            torch.cat(top_log_prob_batches, dim=0)
            if top_log_prob_batches
            else None
        ),
        candidate_log_probs=(
            torch.cat(candidate_log_prob_batches, dim=0)
            if candidate_log_prob_batches
            else None
        ),
        cache=cache,
    )
