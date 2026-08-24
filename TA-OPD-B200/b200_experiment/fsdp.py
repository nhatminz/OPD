from __future__ import annotations

import functools
import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .distributed import DistributedContext, unwrap_model


FSDP = FullyShardedDataParallel


def distributed_strategy(config: dict[str, Any], context: DistributedContext) -> str:
    requested = str(config.get("distributed", {}).get("strategy", "fsdp")).lower()
    if requested not in {"fsdp", "ddp"}:
        raise ValueError("distributed.strategy must be 'fsdp' or 'ddp'")
    return requested if context.enabled else "single_process"


def is_fsdp_model(model) -> bool:
    return isinstance(model, FSDP)


def _transformer_layer_classes(model, fsdp_config: dict[str, Any]) -> set[type]:
    configured = fsdp_config.get("transformer_layer_cls_names")
    if configured is None:
        configured = getattr(model, "_no_split_modules", None)
    if isinstance(configured, str):
        configured = [configured]
    names = {str(name) for name in (configured or ["Qwen3DecoderLayer"])}
    classes = {
        type(module) for module in model.modules() if type(module).__name__ in names
    }
    if not classes:
        raise ValueError(
            "FSDP could not find a configured transformer layer to wrap; "
            f"requested {sorted(names)}"
        )
    return classes


def _mixed_precision() -> MixedPrecision:
    # Match the upstream OPD/verl engineering recipe: BF16 parameters and
    # forward computation, with FP32 reductions/buffers for stable collectives.
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )


def wrap_fsdp_model(
    model,
    config: dict[str, Any],
    context: DistributedContext,
    *,
    role: str,
):
    """FULL_SHARD a Qwen-style student or frozen teacher per decoder layer."""
    if not context.enabled:
        return model
    fsdp_config = dict(config.get("distributed", {}).get("fsdp", {}))
    if role not in {"student", "teacher"}:
        raise ValueError(f"FSDP role must be student or teacher, got {role!r}")
    if str(fsdp_config.get("sharding_strategy", "FULL_SHARD")).upper() != "FULL_SHARD":
        raise ValueError(
            "This implementation requires FSDP sharding_strategy=FULL_SHARD"
        )
    if not bool(fsdp_config.get("use_orig_params", True)):
        raise ValueError(
            "This implementation requires distributed.fsdp.use_orig_params=true"
        )
    layer_classes = _transformer_layer_classes(model, fsdp_config)
    hf_parameter_shapes = {
        name: tuple(parameter.shape) for name, parameter in model.named_parameters()
    }
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_classes,
    )
    teacher_cpu_offload = bool(fsdp_config.get("teacher_cpu_offload", False))
    cpu_offload = (
        CPUOffload(offload_params=True)
        if role == "teacher" and teacher_cpu_offload
        else None
    )
    wrapped = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=_mixed_precision(),
        cpu_offload=cpu_offload,
        device_id=context.device,
        sync_module_states=True,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )
    # A small, immutable schema lets the CUDA-IPC path prove that recursive
    # FSDP wrapping did not leak wrapper names or shard shapes to vLLM.
    wrapped._b200_hf_parameter_shapes = hf_parameter_shapes
    return wrapped


@contextmanager
def materialize_full_parameters(model) -> Iterator[Any]:
    """Yield the original HF module with complete GPU parameters on every rank."""
    if not is_fsdp_model(model):
        yield unwrap_model(model)
        return
    with FSDP.summon_full_params(
        model,
        recurse=True,
        writeback=False,
        rank0_only=False,
        offload_to_cpu=False,
    ):
        yield unwrap_model(model)


def validated_hf_named_parameters(model, full_model) -> list[tuple[str, Any]]:
    """Return complete original HF parameters and validate names/shapes."""
    normalized = [
        (name.replace("_fsdp_wrapped_module.", ""), parameter)
        for name, parameter in full_model.named_parameters()
    ]
    names = [name for name, _ in normalized]
    if any("_flat_param" in name or "_fsdp_wrapped_module" in name for name in names):
        raise RuntimeError(
            "FSDP exposed non-Hugging-Face parameter names to vLLM: "
            + ", ".join(names[:5])
        )
    if len(names) != len(set(names)):
        raise RuntimeError("FSDP parameter-name normalization produced duplicates")
    expected = getattr(model, "_b200_hf_parameter_shapes", None)
    if expected is not None:
        actual = {name: tuple(parameter.shape) for name, parameter in normalized}
        if actual != expected:
            missing = sorted(set(expected) - set(actual))[:5]
            unexpected = sorted(set(actual) - set(expected))[:5]
            wrong_shapes = [
                (name, expected[name], actual[name])
                for name in sorted(set(expected) & set(actual))
                if expected[name] != actual[name]
            ][:5]
            raise RuntimeError(
                "FSDP full parameters do not match the original HF schema: "
                f"missing={missing}, unexpected={unexpected}, "
                f"wrong_shapes={wrong_shapes}"
            )
        if any(parameter.device.type != "cuda" for _, parameter in normalized):
            raise RuntimeError("FSDP full parameters must remain on GPU for CUDA IPC")
    return normalized


def clip_grad_norm(model, max_norm: float) -> torch.Tensor:
    if is_fsdp_model(model):
        return model.clip_grad_norm_(max_norm=float(max_norm))
    return torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(max_norm),
    )


def full_model_state_dict(model) -> dict[str, torch.Tensor] | None:
    """Collect an HF-named, CPU full state dict on rank 0."""
    if not is_fsdp_model(model):
        return unwrap_model(model).state_dict()
    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=state_config,
        ):
            return model.state_dict()


def full_optimizer_state_dict(model, optimizer) -> dict[str, Any] | None:
    """Collect a world-size-portable full optimizer state on rank 0."""
    if not is_fsdp_model(model):
        return optimizer.state_dict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)


def scatter_full_optimizer_state_dict(
    full_state: dict[str, Any] | None,
    model,
    optimizer,
) -> dict[str, Any]:
    if not is_fsdp_model(model):
        if full_state is None:
            raise ValueError("Optimizer state is missing")
        return full_state
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return FSDP.scatter_full_optim_state_dict(
            full_state,
            model,
            optim=optimizer,
        )
