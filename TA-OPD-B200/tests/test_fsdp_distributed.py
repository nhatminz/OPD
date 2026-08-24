from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

from b200_experiment.distributed import DistributedContext
from b200_experiment.fsdp import (
    full_model_state_dict,
    full_optimizer_state_dict,
    materialize_full_parameters,
    scatter_full_optimizer_state_dict,
    validated_hf_named_parameters,
    wrap_fsdp_model,
)
from b200_experiment.opd_core import topk_reference_from_logits
from b200_experiment.scoring import RolloutBatch
from b200_experiment.trainer import _opd_train_step


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(hidden))


class _TinyQwen3LM(nn.Module):
    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 8)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(8) for _ in range(2)])
        self.lm_head = nn.Linear(8, 17, bias=False)
        self.config = type(
            "Config", (), {"use_cache": False, "model_type": "tiny_qwen3"}
        )()

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        del attention_mask, use_cache, return_dict
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return type("Output", (), {"logits": self.lm_head(hidden)})()


def _config() -> dict:
    return {
        "distributed": {
            "strategy": "fsdp",
            "fsdp": {
                "transformer_layer_cls_names": ["Qwen3DecoderLayer"],
                "teacher_cpu_offload": False,
                "use_no_sync": False,
            },
        },
        "rollout": {"temperature": 1.0},
        "selector": {"score_chunk_steps": 2},
        "training": {
            "micro_batch_size_per_gpu": 1,
            "ppo_clip_low": 0.2,
            "ppo_clip_high": 0.28,
            "ppo_dual_clip": 3.0,
            "max_grad_norm": 1.0,
        },
    }


def _fsdp_worker(rank: int, rendezvous: str, output_root: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(123)
        device = torch.device("cuda", rank)
        context = DistributedContext(rank, rank, 2, device)
        student = _TinyQwen3LM()
        teacher = _TinyQwen3LM()
        teacher.requires_grad_(False)
        wrapped_student = wrap_fsdp_model(student, _config(), context, role="student")
        wrapped_teacher = wrap_fsdp_model(teacher, _config(), context, role="teacher")
        if not isinstance(wrapped_student, FullyShardedDataParallel):
            raise AssertionError("Student root is not FSDP")
        if wrapped_student.sharding_strategy != ShardingStrategy.FULL_SHARD:
            raise AssertionError("Student does not use FULL_SHARD")
        if wrapped_teacher.sharding_strategy != ShardingStrategy.FULL_SHARD:
            raise AssertionError("Teacher does not use FULL_SHARD")
        if any(parameter.requires_grad for parameter in wrapped_teacher.parameters()):
            raise AssertionError("Teacher unexpectedly has trainable parameters")

        input_ids = torch.tensor(
            [[1 + rank, 2 + rank, 3 + rank, 4 + rank]], device=device
        )
        rollout = RolloutBatch(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            response_ids=input_ids[:, 2:],
            valid_mask=torch.ones((1, 2), dtype=torch.bool, device=device),
            rollout_log_probs=torch.zeros((1, 2), device=device),
            prompt_width=2,
        )
        with torch.inference_mode():
            teacher_output = wrapped_teacher(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
        if teacher_output.logits.shape != (1, 4, 17):
            raise AssertionError("Sharded teacher forward returned a wrong shape")

        with torch.no_grad():
            output = wrapped_student(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits[:, 1:3]
            reference = topk_reference_from_logits(
                logits,
                logits + torch.linspace(-0.2, 0.2, 17, device=device),
                rollout.valid_mask,
                top_k=5,
            )
        optimizer = torch.optim.AdamW(wrapped_student.parameters(), lr=1e-3)
        metrics = _opd_train_step(
            wrapped_student,
            optimizer,
            rollout,
            rollout.valid_mask,
            reference,
            _config(),
            device,
            context,
        )
        if not torch.isfinite(torch.tensor(metrics["loss"], device=device)):
            raise AssertionError("FSDP OPD loss is not finite")

        with materialize_full_parameters(wrapped_student) as full_model:
            named_parameters = validated_hf_named_parameters(
                wrapped_student, full_model
            )
            names = [name for name, _ in named_parameters]
            if set(names) != set(wrapped_student._b200_hf_parameter_shapes):
                raise AssertionError(f"Wrong HF parameter names: {names}")
            dist.barrier()

        model_state = full_model_state_dict(wrapped_student)
        optimizer_state = full_optimizer_state_dict(wrapped_student, optimizer)
        if rank == 0:
            if not model_state or not optimizer_state:
                raise AssertionError("Rank 0 did not receive full checkpoint states")
            torch.save(model_state, Path(output_root) / "model-state.pt")
            torch.save(metrics, Path(output_root) / "train-metrics.pt")
        sharded_optimizer = scatter_full_optimizer_state_dict(
            optimizer_state if rank == 0 else None,
            wrapped_student,
            optimizer,
        )
        optimizer.load_state_dict(sharded_optimizer)

        # Simulate the worst one-sample global tail. Rank 1 repeats a
        # deterministic filler so both ranks execute one forward/backward,
        # while only rank 0 contributes to the global objective.
        tail_ids = torch.tensor([[1, 2, 3, 4]], device=device)
        tail_active = torch.tensor(
            [[True, True]] if rank == 0 else [[False, False]],
            device=device,
        )
        tail_rollout = RolloutBatch(
            input_ids=tail_ids,
            attention_mask=torch.ones_like(tail_ids),
            response_ids=tail_ids[:, 2:],
            valid_mask=torch.ones((1, 2), dtype=torch.bool, device=device),
            rollout_log_probs=torch.zeros((1, 2), device=device),
            prompt_width=2,
        )
        with torch.no_grad():
            tail_output = wrapped_student(
                input_ids=tail_ids,
                attention_mask=torch.ones_like(tail_ids),
                use_cache=False,
                return_dict=True,
            )
            tail_logits = tail_output.logits[:, 1:3]
            tail_reference = topk_reference_from_logits(
                tail_logits,
                tail_logits + torch.linspace(-0.2, 0.2, 17, device=device),
                tail_active,
                top_k=5,
            )
        _opd_train_step(
            wrapped_student,
            optimizer,
            tail_rollout,
            tail_active,
            tail_reference,
            _config(),
            device,
            context,
            objective_valid_mask=tail_active,
        )
        (Path(output_root) / f"tail-rank-{rank}.ok").touch()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.device_count() >= 2
    and dist.is_nccl_available(),
    "requires two CUDA GPUs with NCCL",
)
class FSDPDistributedTests(unittest.TestCase):
    def test_two_gpu_sharding_train_export_and_resume_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            rendezvous = str(Path(temporary) / "fsdp-rendezvous")
            mp.spawn(
                _fsdp_worker,
                args=(rendezvous, temporary),
                nprocs=2,
                join=True,
            )
            state = torch.load(
                Path(temporary) / "model-state.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(state)
            self.assertTrue(all("_flat_param" not in name for name in state))
            torch.manual_seed(123)
            reference_model = _TinyQwen3LM()
            input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
            rollout = RolloutBatch(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                response_ids=input_ids[:, 2:],
                valid_mask=torch.ones((2, 2), dtype=torch.bool),
                rollout_log_probs=torch.zeros((2, 2)),
                prompt_width=2,
            )
            with torch.no_grad():
                output = reference_model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    return_dict=True,
                )
                logits = output.logits[:, 1:3]
                reference = topk_reference_from_logits(
                    logits,
                    logits + torch.linspace(-0.2, 0.2, 17),
                    rollout.valid_mask,
                    top_k=5,
                )
            reference_metrics = _opd_train_step(
                reference_model,
                torch.optim.AdamW(reference_model.parameters(), lr=1e-3),
                rollout,
                rollout.valid_mask,
                reference,
                _config(),
                torch.device("cpu"),
                DistributedContext(0, 0, 1, torch.device("cpu")),
            )
            fsdp_metrics = torch.load(
                Path(temporary) / "train-metrics.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertAlmostEqual(
                fsdp_metrics["loss"], reference_metrics["loss"], delta=2e-3
            )
            reference_state = reference_model.state_dict()
            for name, expected in reference_state.items():
                self.assertTrue(
                    torch.allclose(state[name], expected, atol=2e-3, rtol=2e-3),
                    name,
                )
            self.assertTrue((Path(temporary) / "tail-rank-0.ok").is_file())
            self.assertTrue((Path(temporary) / "tail-rank-1.ok").is_file())


if __name__ == "__main__":
    unittest.main()
