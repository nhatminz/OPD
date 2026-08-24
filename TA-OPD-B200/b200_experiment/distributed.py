from __future__ import annotations

import os
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Small, explicit wrapper around single-node distributed state."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def sum_int(self, value: int) -> int:
        tensor = torch.tensor(value, dtype=torch.int64, device=self.device)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())

    def sum_float(self, value: float) -> float:
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def mean_float(self, value: float) -> float:
        return self.sum_float(value) / self.world_size

    def max_float(self, value: float) -> float:
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def max_int(self, value: int) -> int:
        tensor = torch.tensor(value, dtype=torch.int64, device=self.device)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor.item())

    def any(self, value: bool) -> bool:
        return self.max_int(int(value)) != 0

    def all_gather_objects(self, value):
        if not self.enabled:
            return [value]
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, value)
        return gathered

    def broadcast_object(self, value, source: int = 0):
        if not self.enabled:
            return value
        values = [value if self.rank == source else None]
        dist.broadcast_object_list(values, src=source, device=self.device)
        return values[0]

    def all_gather_variable_1d(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, int, int, tuple[int, ...]]:
        """Gather a variable-length 1-D CUDA tensor on every worker.

        Rank order is retained. With contiguous batch partitioning this is also
        the original global sample/response-position order, which makes stable
        top-budget tie breaking invariant to the number of workers.
        """
        values = values.contiguous().reshape(-1)
        if not self.enabled:
            return values, 0, values.numel(), (values.numel(),)
        local_length = torch.tensor(
            [values.numel()], dtype=torch.int64, device=self.device
        )
        gathered_lengths = [
            torch.zeros_like(local_length) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_lengths, local_length)
        lengths = tuple(int(item.item()) for item in gathered_lengths)
        maximum = max(lengths, default=0)
        if maximum == 0:
            empty = values.new_empty((0,))
            return empty, 0, 0, lengths
        padded = values.new_zeros((maximum,))
        if values.numel():
            padded[: values.numel()] = values
        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded)
        combined = torch.cat(
            [tensor[:length] for tensor, length in zip(gathered, lengths)], dim=0
        )
        start = sum(lengths[: self.rank])
        return combined, start, start + lengths[self.rank], lengths

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(backend: str = "nccl") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK={rank} is invalid for WORLD_SIZE={world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("B200 training requires CUDA")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA "
            "devices are visible"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank, local_rank, world_size, device)


def contiguous_partition(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Balanced contiguous partition with no padding or repeated examples."""
    if total < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(
            f"Invalid partition arguments: total={total}, rank={rank}, world={world_size}"
        )
    quotient, remainder = divmod(total, world_size)
    start = rank * quotient + min(rank, remainder)
    length = quotient + int(rank < remainder)
    return start, start + length


@dataclass(frozen=True)
class BatchLayout:
    global_prompt_batch_size: int
    world_size: int
    local_prompt_batch_size: int
    num_responses: int
    global_trajectory_batch_size: int
    local_trajectory_batch_size: int
    micro_batch_size_per_gpu: int
    micro_batches_per_gpu: int


def batch_layout(
    global_prompt_batch_size: int,
    num_responses: int,
    world_size: int,
    micro_batch_size_per_gpu: int,
) -> BatchLayout:
    """Resolve explicit global/local batch semantics for one training step."""
    values = {
        "global_prompt_batch_size": int(global_prompt_batch_size),
        "num_responses": int(num_responses),
        "world_size": int(world_size),
        "micro_batch_size_per_gpu": int(micro_batch_size_per_gpu),
    }
    invalid = {name: value for name, value in values.items() if value <= 0}
    if invalid:
        raise ValueError(f"Batch layout values must be positive: {invalid}")
    local_prompts = (
        values["global_prompt_batch_size"] + values["world_size"] - 1
    ) // values["world_size"]
    return BatchLayout(
        global_prompt_batch_size=values["global_prompt_batch_size"],
        world_size=values["world_size"],
        local_prompt_batch_size=local_prompts,
        num_responses=values["num_responses"],
        global_trajectory_batch_size=(
            values["global_prompt_batch_size"] * values["num_responses"]
        ),
        local_trajectory_batch_size=local_prompts * values["num_responses"],
        micro_batch_size_per_gpu=values["micro_batch_size_per_gpu"],
        micro_batches_per_gpu=(
            local_prompts * values["num_responses"]
            + values["micro_batch_size_per_gpu"]
            - 1
        )
        // values["micro_batch_size_per_gpu"],
    )


def padded_local_indices(
    global_indices: list[int], rank: int, world_size: int
) -> tuple[list[int], list[bool]]:
    """Return equal-sized rank batches while marking deterministic tail padding.

    Every real index remains present exactly once. At an uneven dataset tail the
    shorter ranks repeat one local index only to keep FSDP collective schedules
    identical; callers must exclude entries whose active flag is false from the
    objective and all scientific metrics.
    """
    total = len(global_indices)
    if total <= 0:
        raise ValueError("Global batch must contain at least one real prompt")
    start, end = contiguous_partition(total, rank, world_size)
    local = list(global_indices[start:end])
    target = (total + world_size - 1) // world_size
    active = [True] * len(local)
    # Empty ranks (possible for a one-sample final tail) use a global sample as
    # collective-only filler. It remains inactive and contributes nowhere.
    filler = local[-1] if local else global_indices[-1]
    while len(local) < target:
        local.append(filler)
        active.append(False)
    return local, active


def unique_free_port(context: DistributedContext, attempts: int = 10) -> int:
    """Choose distinct localhost ports across workers before starting servers."""
    for _ in range(max(1, attempts)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind(("127.0.0.1", 0))
            port = int(handle.getsockname()[1])
        if not context.enabled:
            return port
        local = torch.tensor([port], dtype=torch.int64, device=context.device)
        gathered = [torch.zeros_like(local) for _ in range(context.world_size)]
        dist.all_gather(gathered, local)
        ports = [int(item.item()) for item in gathered]
        if len(set(ports)) == context.world_size:
            return port
    raise RuntimeError("Could not allocate distinct vLLM server ports across workers")


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
