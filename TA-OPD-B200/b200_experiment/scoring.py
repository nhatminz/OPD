from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DynamicCache


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
) -> RolloutBatch:
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    device = prompt_ids.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
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
            next_token = _sample_top_p(
                torch.softmax(logits / temperature, dim=-1), top_p, generator
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
    probabilities: torch.Tensor
    sampled_log_probs: torch.Tensor
    cache: object


@torch.inference_mode()
def score_original_rollout(
    model, rollout: RolloutBatch, keep_cache: bool = True
) -> BaseScores:
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
    logits = output.logits[:, start : start + width, :].float()
    probabilities = torch.softmax(logits, dim=-1)
    sampled = torch.log(
        probabilities.gather(-1, rollout.response_ids.unsqueeze(-1))
        .squeeze(-1)
        .clamp_min(1e-30)
    )
    return BaseScores(
        probabilities, sampled, output.past_key_values if keep_cache else None
    )


def _clone_selected_cache(
    base_cache, batch_indices: torch.Tensor, prefix_length: int
) -> DynamicCache:
    if hasattr(base_cache, "layers"):
        cache_data = []
        for layer in base_cache.layers:
            if not getattr(layer, "is_initialized", False):
                cache_data.append((None, None))
            else:
                cache_data.append(
                    (
                        layer.keys.index_select(0, batch_indices)[
                            ..., :prefix_length, :
                        ],
                        layer.values.index_select(0, batch_indices)[
                            ..., :prefix_length, :
                        ],
                    )
                )
        return DynamicCache(cache_data)
    if hasattr(base_cache, "key_cache"):
        cloned = DynamicCache()
        for layer_index, (keys, values) in enumerate(
            zip(base_cache.key_cache, base_cache.value_cache)
        ):
            cloned.update(
                keys.index_select(0, batch_indices)[..., :prefix_length, :],
                values.index_select(0, batch_indices)[..., :prefix_length, :],
                layer_index,
            )
        return cloned
    raise TypeError(f"Unsupported Transformers cache type: {type(base_cache).__name__}")


class HFBranchProber:
    """No-grad counterfactual C+ probes with batched branch and KV-cache reuse."""

    def __init__(
        self,
        student,
        teacher,
        rollout,
        student_cache,
        teacher_cache,
        *,
        top_k,
        mode="kv_cache",
        chunk_size=256,
    ):
        if mode not in {"kv_cache", "full_prefix"}:
            raise ValueError(f"Unknown RAC probe mode {mode!r}")
        self.student, self.teacher, self.rollout = student, teacher, rollout
        self.student_cache, self.teacher_cache = student_cache, teacher_cache
        self.top_k, self.mode, self.chunk_size = (
            int(top_k),
            mode,
            max(1, int(chunk_size)),
        )
        self.valid_coordinates = rollout.valid_mask.nonzero(as_tuple=False)
        self.counterfactual_seconds = 0.0
        self.evaluated_branches = 0

    @torch.inference_mode()
    def __call__(
        self, valid_rows: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        original = self.rollout.input_ids.clone()
        cuda_sync(candidate_ids.device)
        started = time.perf_counter()
        result = (
            self._probe_kv(valid_rows, candidate_ids)
            if self.mode == "kv_cache"
            else self._probe_full_prefix(valid_rows, candidate_ids)
        )
        cuda_sync(candidate_ids.device)
        self.counterfactual_seconds += time.perf_counter() - started
        self.evaluated_branches += candidate_ids.numel()
        if not torch.equal(original, self.rollout.input_ids):
            raise AssertionError("Counterfactual probes modified the original rollout")
        if result.requires_grad or result.grad_fn is not None:
            raise AssertionError("Counterfactual probes created an autograd graph")
        return result

    def _cplus(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        ids = torch.topk(
            student_logits, k=min(self.top_k, student_logits.shape[-1]), dim=-1
        ).indices
        teacher_logits = teacher_logits.float()
        return torch.exp(
            teacher_logits.gather(-1, ids)
            - torch.logsumexp(teacher_logits, -1, keepdim=True)
        ).sum(-1)

    def _probe_kv(
        self, valid_rows: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.student_cache is None or self.teacher_cache is None:
            raise ValueError("kv_cache mode requires student and teacher caches")
        coords = self.valid_coordinates.index_select(0, valid_rows)
        absolute_positions = self.rollout.prompt_width + coords[:, 1]
        result = torch.empty_like(candidate_ids, dtype=torch.float32)
        for prefix_tensor in torch.unique(absolute_positions, sorted=True):
            prefix_length = int(prefix_tensor.item())
            group = (
                absolute_positions.eq(prefix_length).nonzero(as_tuple=False).squeeze(-1)
            )
            for offset in range(0, group.numel(), self.chunk_size):
                selected = group[offset : offset + self.chunk_size]
                batch_indices = coords.index_select(0, selected)[:, 0]
                tokens = candidate_ids.index_select(0, selected)
                attention = torch.cat(
                    (
                        self.rollout.attention_mask.index_select(0, batch_indices)[
                            :, :prefix_length
                        ],
                        torch.ones(
                            (selected.numel(), 1),
                            dtype=torch.long,
                            device=tokens.device,
                        ),
                    ),
                    dim=1,
                )
                position = attention.sum(dim=-1, keepdim=True) - 1
                cache_position = torch.tensor([prefix_length], device=tokens.device)
                student_output = self.student(
                    input_ids=tokens[:, None],
                    attention_mask=attention,
                    position_ids=position,
                    cache_position=cache_position,
                    past_key_values=_clone_selected_cache(
                        self.student_cache, batch_indices, prefix_length
                    ),
                    use_cache=False,
                    return_dict=True,
                )
                teacher_output = self.teacher(
                    input_ids=tokens[:, None],
                    attention_mask=attention,
                    position_ids=position,
                    cache_position=cache_position,
                    past_key_values=_clone_selected_cache(
                        self.teacher_cache, batch_indices, prefix_length
                    ),
                    use_cache=False,
                    return_dict=True,
                )
                result[selected] = self._cplus(
                    student_output.logits[:, -1, :], teacher_output.logits[:, -1, :]
                )
        return result

    def _probe_full_prefix(
        self, valid_rows: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        coords = self.valid_coordinates.index_select(0, valid_rows)
        outputs = []
        for begin in range(0, candidate_ids.numel(), self.chunk_size):
            end = min(begin + self.chunk_size, candidate_ids.numel())
            chunk_coords, chunk_tokens = coords[begin:end], candidate_ids[begin:end]
            absolute = self.rollout.prompt_width + chunk_coords[:, 1]
            lengths, max_length = absolute + 1, int((absolute + 1).max().item())
            ids = torch.zeros(
                (end - begin, max_length), dtype=torch.long, device=candidate_ids.device
            )
            attention = torch.zeros_like(ids)
            for row, (coord, token, prefix) in enumerate(
                zip(chunk_coords, chunk_tokens, absolute)
            ):
                batch_index, prefix_length = int(coord[0]), int(prefix)
                ids[row, :prefix_length] = self.rollout.input_ids[
                    batch_index, :prefix_length
                ]
                attention[row, :prefix_length] = self.rollout.attention_mask[
                    batch_index, :prefix_length
                ]
                ids[row, prefix_length], attention[row, prefix_length] = token, 1
            positions = position_ids_from_mask(attention)
            indices = torch.arange(end - begin, device=ids.device)
            student_logits = self.student(
                input_ids=ids,
                attention_mask=attention,
                position_ids=positions,
                use_cache=False,
            ).logits[indices, lengths - 1]
            teacher_logits = self.teacher(
                input_ids=ids,
                attention_mask=attention,
                position_ids=positions,
                use_cache=False,
            ).logits[indices, lengths - 1]
            outputs.append(self._cplus(student_logits, teacher_logits))
        return (
            torch.cat(outputs)
            if outputs
            else torch.empty(0, device=candidate_ids.device)
        )

    @torch.inference_mode()
    def validate_kv_equivalence(
        self, valid_row: int = 0, candidate_id: int | None = None
    ) -> float:
        if self.valid_coordinates.numel() == 0:
            raise ValueError("Cannot validate an empty rollout")
        coordinate = self.valid_coordinates[valid_row]
        batch_index = int(coordinate[0])
        prefix_length = self.rollout.prompt_width + int(coordinate[1])
        candidate_id = (
            candidate_id
            if candidate_id is not None
            else int(self.rollout.input_ids[batch_index, prefix_length])
        )
        token = torch.tensor([[candidate_id]], device=self.rollout.input_ids.device)
        prefix_ids = self.rollout.input_ids[
            batch_index : batch_index + 1, :prefix_length
        ]
        prefix_attention = self.rollout.attention_mask[
            batch_index : batch_index + 1, :prefix_length
        ]
        full_ids = torch.cat((prefix_ids, token), dim=1)
        full_attention = torch.cat((prefix_attention, torch.ones_like(token)), dim=1)
        batch_tensor = torch.tensor([batch_index], device=token.device)
        errors = []
        for model, cache in (
            (self.student, self.student_cache),
            (self.teacher, self.teacher_cache),
        ):
            full_logits = (
                model(
                    input_ids=full_ids,
                    attention_mask=full_attention,
                    position_ids=position_ids_from_mask(full_attention),
                    use_cache=False,
                )
                .logits[:, -1, :]
                .float()
            )
            cached_logits = (
                model(
                    input_ids=token,
                    attention_mask=full_attention,
                    position_ids=full_attention.sum(-1, keepdim=True) - 1,
                    cache_position=torch.tensor([prefix_length], device=token.device),
                    past_key_values=_clone_selected_cache(
                        cache, batch_tensor, prefix_length
                    ),
                    use_cache=False,
                )
                .logits[:, -1, :]
                .float()
            )
            errors.append(
                (torch.softmax(full_logits, -1) - torch.softmax(cached_logits, -1))
                .abs()
                .max()
            )
        return float(torch.stack(errors).max().item())
