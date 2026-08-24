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
    scaled_log_normalizers: torch.Tensor
    sampled_log_probs: torch.Tensor
    top_k_ids: torch.Tensor | None = None
    top_k_log_probs: torch.Tensor | None = None
    candidate_log_probs: torch.Tensor | None = None
    cache: object | None = None


def supports_response_only_logits(model) -> bool:
    """Qwen3 can skip the expensive LM head on prompt hidden states."""
    base_model = getattr(model, "module", model)
    return str(getattr(getattr(base_model, "config", None), "model_type", "")) in {
        "qwen3",
        "qwen3_moe",
    }


def _pad_response_time(value: torch.Tensor, width: int, fill_value: float | int = 0):
    if value.shape[1] == width:
        return value
    shape = list(value.shape)
    shape[1] = width
    padded = torch.full(shape, fill_value, dtype=value.dtype, device=value.device)
    padded[:, : value.shape[1]] = value
    return padded


def _reduce_response_logits(
    response_logits: torch.Tensor,
    response_ids: torch.Tensor,
    *,
    score_chunk_steps: int,
    retain_response_logits: bool,
    top_k: int,
    candidate_ids: torch.Tensor | None,
    temperature: float,
) -> BaseScores:
    """Reduce one compact response-logit micro-batch to the tensors consumers use."""
    normalizer_chunks, scaled_normalizer_chunks, sampled_chunks = [], [], []
    top_id_chunks, top_log_prob_chunks, candidate_log_prob_chunks = [], [], []
    width = response_logits.shape[1]
    chunk_steps = max(1, int(score_chunk_steps))
    for begin in range(0, width, chunk_steps):
        end = min(begin + chunk_steps, width)
        raw_chunk = response_logits[:, begin:end].float()
        normalizers = torch.logsumexp(raw_chunk, dim=-1)
        if float(temperature) == 1.0:
            chunk = raw_chunk
            scaled_normalizers = normalizers
        else:
            chunk = raw_chunk / float(temperature)
            scaled_normalizers = torch.logsumexp(chunk, dim=-1)
        sampled = (
            chunk.gather(
                -1,
                response_ids[:, begin:end].unsqueeze(-1),
            ).squeeze(-1)
            - scaled_normalizers
        )
        normalizer_chunks.append(normalizers)
        scaled_normalizer_chunks.append(scaled_normalizers)
        sampled_chunks.append(sampled)
        if top_k > 0:
            k = min(int(top_k), chunk.shape[-1])
            top_values, top_ids = torch.topk(chunk, k=k, dim=-1)
            top_id_chunks.append(top_ids)
            top_log_prob_chunks.append(top_values - scaled_normalizers.unsqueeze(-1))
        if candidate_ids is not None:
            candidate_log_prob_chunks.append(
                chunk.gather(-1, candidate_ids[:, begin:end])
                - scaled_normalizers.unsqueeze(-1)
            )
    return BaseScores(
        response_logits=(
            response_logits.detach().clone() if retain_response_logits else None
        ),
        log_normalizers=torch.cat(normalizer_chunks, dim=1),
        scaled_log_normalizers=torch.cat(scaled_normalizer_chunks, dim=1),
        sampled_log_probs=torch.cat(sampled_chunks, dim=1),
        top_k_ids=(torch.cat(top_id_chunks, dim=1) if top_id_chunks else None),
        top_k_log_probs=(
            torch.cat(top_log_prob_chunks, dim=1) if top_log_prob_chunks else None
        ),
        candidate_log_probs=(
            torch.cat(candidate_log_prob_chunks, dim=1)
            if candidate_log_prob_chunks
            else None
        ),
    )


def _gather_with_scaled_normalizers(
    response_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    scaled_log_normalizers: torch.Tensor,
    *,
    score_chunk_steps: int,
    temperature: float,
) -> torch.Tensor:
    """Gather a second support without another vocab reduction or model forward."""
    chunks = []
    width = response_logits.shape[1]
    chunk_steps = max(1, int(score_chunk_steps))
    for begin in range(0, width, chunk_steps):
        end = min(begin + chunk_steps, width)
        logits = response_logits[:, begin:end].float()
        if float(temperature) != 1.0:
            logits = logits / float(temperature)
        chunks.append(
            logits.gather(-1, candidate_ids[:, begin:end])
            - scaled_log_normalizers[:, begin:end].unsqueeze(-1)
        )
    return torch.cat(chunks, dim=1)


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
    trim_padding: bool = True,
    length_bucketed: bool = True,
) -> BaseScores:
    """Score actions while skipping right padding and unnecessary prompt logits."""
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
    response_lengths = rollout.valid_mask.long().sum(dim=-1)
    if bool(response_lengths.le(0).any()):
        raise ValueError(
            "Every rollout trajectory must contain at least one valid token"
        )
    prefix_mask = torch.arange(width, device=rollout.valid_mask.device).unsqueeze(0)
    prefix_mask = prefix_mask < response_lengths.unsqueeze(1)
    if not torch.equal(prefix_mask, rollout.valid_mask.bool()):
        raise ValueError("Rollout valid_mask must be a contiguous response prefix")
    order = torch.arange(batch_size, device=rollout.input_ids.device)
    if length_bucketed and not keep_cache and score_micro_batch > 1:
        order = torch.argsort(response_lengths, descending=True, stable=True)
    normalizer_batches, scaled_normalizer_batches, sampled_batches = [], [], []
    top_id_batches, top_log_prob_batches, candidate_log_prob_batches = [], [], []
    response_logit_batches = []
    cache = None
    for batch_begin in range(0, batch_size, score_micro_batch):
        indices = order[batch_begin : batch_begin + score_micro_batch]
        local_width = width
        if trim_padding and not keep_cache:
            local_width = int(response_lengths.index_select(0, indices).max().item())
        # To score W response tokens we only need the prompt and the first W-1
        # response inputs. The final sampled token and every post-EOS pad token
        # cannot affect any requested causal logit.
        input_stop = start + local_width if trim_padding and not keep_cache else None
        input_ids = rollout.input_ids.index_select(0, indices)[:, :input_stop]
        attention_mask = rollout.attention_mask.index_select(0, indices)[:, :input_stop]
        response_ids = rollout.response_ids.index_select(0, indices)[:, :local_width]
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids_from_mask(attention_mask),
            "use_cache": keep_cache,
            "return_dict": True,
        }
        response_only_logits = input_stop is not None and supports_response_only_logits(
            model
        )
        if response_only_logits:
            forward_kwargs["logits_to_keep"] = local_width
        output = model(
            **forward_kwargs,
        )
        # Keep model dtype (BF16 on B200); FP32 reductions are chunked over time.
        response_logits_view = (
            output.logits[:, -local_width:, :]
            if response_only_logits
            else output.logits[:, start : start + local_width, :]
        )
        if response_logits_view.shape[1] != local_width:
            raise RuntimeError("Model did not return every requested response logit")
        local_candidate_ids = (
            candidate_ids.index_select(0, indices)[:, :local_width]
            if candidate_ids is not None
            else None
        )
        local_scores = _reduce_response_logits(
            response_logits_view,
            response_ids,
            score_chunk_steps=score_chunk_steps,
            retain_response_logits=retain_response_logits,
            top_k=top_k,
            candidate_ids=local_candidate_ids,
            temperature=temperature,
        )
        normalizer_batches.append(
            _pad_response_time(local_scores.log_normalizers, width)
        )
        scaled_normalizer_batches.append(
            _pad_response_time(local_scores.scaled_log_normalizers, width)
        )
        sampled_batches.append(
            _pad_response_time(local_scores.sampled_log_probs, width)
        )
        if local_scores.top_k_ids is not None:
            top_id_batches.append(_pad_response_time(local_scores.top_k_ids, width))
            top_log_prob_batches.append(
                _pad_response_time(local_scores.top_k_log_probs, width)
            )
        if local_scores.candidate_log_probs is not None:
            candidate_log_prob_batches.append(
                _pad_response_time(local_scores.candidate_log_probs, width)
            )
        if local_scores.response_logits is not None:
            response_logit_batches.append(
                _pad_response_time(local_scores.response_logits, width)
            )
        if keep_cache:
            cache = output.past_key_values
        del output, response_logits_view
    restore_order = torch.argsort(order)

    def restored(values: list[torch.Tensor]) -> torch.Tensor | None:
        return (
            torch.cat(values, dim=0).index_select(0, restore_order) if values else None
        )

    return BaseScores(
        response_logits=(
            restored(response_logit_batches) if response_logit_batches else None
        ),
        log_normalizers=restored(normalizer_batches),
        scaled_log_normalizers=restored(scaled_normalizer_batches),
        sampled_log_probs=restored(sampled_batches),
        top_k_ids=restored(top_id_batches),
        top_k_log_probs=restored(top_log_prob_batches),
        candidate_log_probs=restored(candidate_log_prob_batches),
        cache=cache,
    )


@torch.inference_mode()
def score_student_teacher_rollout(
    student,
    teacher,
    rollout: RolloutBatch,
    *,
    score_chunk_steps: int = 128,
    top_k: int,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
    micro_batch_size: int | None = None,
    trim_padding: bool = True,
    length_bucketed: bool = True,
) -> tuple[BaseScores, BaseScores]:
    """Compute common and bidirectional Top-K scores in two model forwards.

    TA-OPD and RAC need student/teacher Top-K plus each model evaluated on the
    other's support.  The older path scored the student twice.  Here both BF16
    logit views coexist only for one bounded micro-batch, allowing the second
    gather before either view is released.  All returned values are identical
    to three independent ``score_original_rollout`` calls.
    """
    if student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("Scoring temperatures must be positive")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    student.eval()
    teacher.eval()
    start = rollout.prompt_width - 1
    width = rollout.response_ids.shape[1]
    batch_size = rollout.input_ids.shape[0]
    score_micro_batch = (
        batch_size if micro_batch_size is None else max(1, int(micro_batch_size))
    )
    response_lengths = rollout.valid_mask.long().sum(dim=-1)
    if bool(response_lengths.le(0).any()):
        raise ValueError(
            "Every rollout trajectory must contain at least one valid token"
        )
    prefix_mask = torch.arange(width, device=rollout.valid_mask.device).unsqueeze(0)
    prefix_mask = prefix_mask < response_lengths.unsqueeze(1)
    if not torch.equal(prefix_mask, rollout.valid_mask.bool()):
        raise ValueError("Rollout valid_mask must be a contiguous response prefix")
    order = torch.arange(batch_size, device=rollout.input_ids.device)
    if length_bucketed and score_micro_batch > 1:
        order = torch.argsort(response_lengths, descending=True, stable=True)

    fields = (
        "log_normalizers",
        "scaled_log_normalizers",
        "sampled_log_probs",
        "top_k_ids",
        "top_k_log_probs",
        "candidate_log_probs",
    )
    student_batches: dict[str, list[torch.Tensor]] = {field: [] for field in fields}
    teacher_batches: dict[str, list[torch.Tensor]] = {field: [] for field in fields}

    for batch_begin in range(0, batch_size, score_micro_batch):
        indices = order[batch_begin : batch_begin + score_micro_batch]
        local_width = width
        if trim_padding:
            local_width = int(response_lengths.index_select(0, indices).max().item())
        input_stop = start + local_width if trim_padding else None
        input_ids = rollout.input_ids.index_select(0, indices)[:, :input_stop]
        attention_mask = rollout.attention_mask.index_select(0, indices)[:, :input_stop]
        response_ids = rollout.response_ids.index_select(0, indices)[:, :local_width]
        common_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids_from_mask(attention_mask),
            "use_cache": False,
            "return_dict": True,
        }

        student_kwargs = dict(common_kwargs)
        student_response_only = (
            input_stop is not None and supports_response_only_logits(student)
        )
        if student_response_only:
            student_kwargs["logits_to_keep"] = local_width
        student_output = student(**student_kwargs)
        student_logits = (
            student_output.logits[:, -local_width:, :]
            if student_response_only
            else student_output.logits[:, start : start + local_width, :]
        )
        student_local = _reduce_response_logits(
            student_logits,
            response_ids,
            score_chunk_steps=score_chunk_steps,
            retain_response_logits=False,
            top_k=top_k,
            candidate_ids=None,
            temperature=student_temperature,
        )
        if student_local.top_k_ids is None:
            raise AssertionError("Student joint scoring did not produce Top-K IDs")

        teacher_kwargs = dict(common_kwargs)
        teacher_response_only = (
            input_stop is not None and supports_response_only_logits(teacher)
        )
        if teacher_response_only:
            teacher_kwargs["logits_to_keep"] = local_width
        teacher_output = teacher(**teacher_kwargs)
        teacher_logits = (
            teacher_output.logits[:, -local_width:, :]
            if teacher_response_only
            else teacher_output.logits[:, start : start + local_width, :]
        )
        teacher_local = _reduce_response_logits(
            teacher_logits,
            response_ids,
            score_chunk_steps=score_chunk_steps,
            retain_response_logits=False,
            top_k=top_k,
            candidate_ids=student_local.top_k_ids,
            temperature=teacher_temperature,
        )
        if teacher_local.top_k_ids is None:
            raise AssertionError("Teacher joint scoring did not produce Top-K IDs")
        student_local.candidate_log_probs = _gather_with_scaled_normalizers(
            student_logits,
            teacher_local.top_k_ids,
            student_local.scaled_log_normalizers,
            score_chunk_steps=score_chunk_steps,
            temperature=student_temperature,
        )

        for field in fields:
            student_value = getattr(student_local, field)
            teacher_value = getattr(teacher_local, field)
            if student_value is not None:
                student_batches[field].append(_pad_response_time(student_value, width))
            if teacher_value is not None:
                teacher_batches[field].append(_pad_response_time(teacher_value, width))
        del student_output, teacher_output, student_logits, teacher_logits

    restore_order = torch.argsort(order)

    def finish(batches: dict[str, list[torch.Tensor]]) -> BaseScores:
        def restored(field: str) -> torch.Tensor | None:
            values = batches[field]
            return (
                torch.cat(values, dim=0).index_select(0, restore_order)
                if values
                else None
            )

        return BaseScores(
            response_logits=None,
            log_normalizers=restored("log_normalizers"),
            scaled_log_normalizers=restored("scaled_log_normalizers"),
            sampled_log_probs=restored("sampled_log_probs"),
            top_k_ids=restored("top_k_ids"),
            top_k_log_probs=restored("top_k_log_probs"),
            candidate_log_probs=restored("candidate_log_probs"),
        )

    return finish(student_batches), finish(teacher_batches)
