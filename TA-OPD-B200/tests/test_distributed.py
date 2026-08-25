from __future__ import annotations

import copy
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from b200_experiment.data import expand_prompt_batch
from b200_experiment.distributed import (
    DistributedContext,
    batch_layout,
    contiguous_partition,
    initialize_distributed,
    isolate_distributed_subprocess_environment,
    padded_local_indices,
    unique_free_port,
)
from b200_experiment.selectors import TASelector
from b200_experiment.trainer import (
    _globalize_ta_output,
    _local_mask_from_global_budget,
    _opd_train_step,
    _optimizer_steps_per_epoch,
    _rollout_position_after_optimizer_steps,
)
from b200_experiment.opd_core import (
    build_topk_opd_reference,
    topk_reference_from_logits,
    weighted_token_sums,
)
from b200_experiment.scoring import (
    RolloutBatch,
    position_ids_from_mask,
    score_original_rollout,
)


class _TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 5)
        self.output = nn.Linear(5, 11, bias=False)
        self.config = type("Config", (), {"use_cache": False})()
        self.forward_calls = 0
        self.position_id_calls = []

    def forward(
        self, input_ids, attention_mask, position_ids=None, use_cache=False, return_dict=True
    ):
        del use_cache, return_dict
        self.forward_calls += 1
        self.position_id_calls.append(
            None if position_ids is None else position_ids.detach().clone()
        )
        if position_ids is not None and not torch.equal(
            position_ids, position_ids_from_mask(attention_mask)
        ):
            raise AssertionError("Forward received inconsistent position_ids")
        logits = self.output(self.embedding(input_ids))
        return type("Output", (), {"logits": logits})()


def _tiny_batch(start: int, end: int) -> RolloutBatch:
    global_ids = torch.tensor(
        [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], dtype=torch.long
    )
    ids = global_ids[start:end]
    responses = ids[:, 2:]
    valid = torch.ones_like(responses, dtype=torch.bool)
    return RolloutBatch(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        response_ids=responses,
        valid_mask=valid,
        rollout_log_probs=torch.zeros_like(responses, dtype=torch.float32),
        prompt_width=2,
    )


def _tiny_config(
    micro_batch_size_per_gpu: int = 3,
    ppo_mini_batch_size: int | None = None,
) -> dict:
    training = {
        "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "ppo_clip_low": 0.2,
        "ppo_clip_high": 0.28,
        "max_grad_norm": 100.0,
    }
    if ppo_mini_batch_size is not None:
        training["ppo_mini_batch_size"] = ppo_mini_batch_size
    return {
        "rollout": {"temperature": 1.0},
        "selector": {"score_chunk_steps": 2},
        "training": training,
    }


@torch.no_grad()
def _tiny_reference(model: nn.Module, rollout: RolloutBatch):
    output = model(
        input_ids=rollout.input_ids,
        attention_mask=rollout.attention_mask,
        use_cache=False,
        return_dict=True,
    )
    width = rollout.response_ids.shape[1]
    logits = output.logits[:, rollout.prompt_width - 1 :][:, :width]
    teacher_bias = torch.linspace(-0.2, 0.2, logits.shape[-1])
    return topk_reference_from_logits(
        logits, logits + teacher_bias, rollout.valid_mask, top_k=5
    )


def _parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.parameters()]
    )


def _ddp_step_worker(rank: int, rendezvous: str, output_root: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(99)
        model = _TinyCausalLM()
        wrapped = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        start, end = contiguous_partition(3, rank, 2)
        rollout = _tiny_batch(start, end)
        opd_reference = _tiny_reference(model, rollout)
        context = DistributedContext(rank, rank, 2, torch.device("cpu"))
        _opd_train_step(
            wrapped,
            optimizer,
            rollout,
            rollout.valid_mask,
            opd_reference,
            _tiny_config(),
            torch.device("cpu"),
            context,
        )
        torch.save(_parameters(model), Path(output_root) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def _gloo_gather_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        context = DistributedContext(rank, rank, 2, torch.device("cpu"))
        local = torch.arange(rank + 2, dtype=torch.float32) + rank * 10
        combined, start, end, lengths = context.all_gather_variable_1d(local)
        expected = torch.tensor([0.0, 1.0, 10.0, 11.0, 12.0])
        if not torch.equal(combined, expected):
            raise AssertionError(f"Unexpected gathered values: {combined}")
        if lengths != (2, 3) or end - start != rank + 2:
            raise AssertionError(
                f"Unexpected distributed layout: {(start, end, lengths)}"
            )
        port = unique_free_port(context)
        port_tensor = torch.tensor([port])
        ports = [torch.zeros_like(port_tensor) for _ in range(2)]
        dist.all_gather(ports, port_tensor)
        if len({int(item.item()) for item in ports}) != 2:
            raise AssertionError(f"vLLM ports are not unique: {ports}")

        torch.manual_seed(321)
        p = torch.softmax(torch.randn(3, 4, 29), dim=-1)
        q = torch.softmax(torch.randn(3, 4, 29), dim=-1)
        valid = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool
        )
        selector = TASelector(top_k=8)
        expected_ta = selector.compute_scores(p, q, valid)
        batch_start, batch_end = contiguous_partition(3, rank, 2)
        local_valid = valid[batch_start:batch_end]
        local_raw = selector.compute_scores(
            p[batch_start:batch_end],
            q[batch_start:batch_end],
            local_valid,
            normalize=False,
        )
        local_ta, global_ta, token_start, token_end = _globalize_ta_output(
            local_raw, local_valid, selector, context
        )
        if not torch.equal(
            local_ta.scores[local_valid],
            expected_ta.scores[valid][token_start:token_end],
        ):
            raise AssertionError("Distributed TA normalization changed scores")
        local_selected, global_selected = _local_mask_from_global_budget(
            global_ta["s_TA"],
            local_valid,
            token_start,
            token_end,
            0.33,
        )
        selected_count = torch.tensor([int(local_selected.sum())])
        dist.all_reduce(selected_count)
        if int(selected_count.item()) != math.ceil(0.33 * int(valid.sum())):
            raise AssertionError(
                f"Distributed global budget selected {selected_count.item()} tokens"
            )
        if int(global_selected.sum()) != int(selected_count.item()):
            raise AssertionError("Workers disagree on the global selected mask")
    finally:
        dist.destroy_process_group()


class DistributedInvariantTests(unittest.TestCase):
    def test_nccl_process_group_is_bound_to_the_local_cuda_device(self):
        with patch.dict(
            os.environ,
            {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"},
            clear=False,
        ), patch(
            "b200_experiment.distributed.torch.cuda.is_available", return_value=True
        ), patch(
            "b200_experiment.distributed.torch.cuda.device_count", return_value=2
        ), patch(
            "b200_experiment.distributed.torch.cuda.set_device"
        ) as set_device, patch(
            "b200_experiment.distributed.dist.is_initialized", return_value=False
        ), patch(
            "b200_experiment.distributed.dist.init_process_group"
        ) as initialize:
            context = initialize_distributed()

        self.assertEqual(context.device, torch.device("cuda", 1))
        set_device.assert_called_once_with(1)
        initialize.assert_called_once_with(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", 1),
        )

    def test_cuda_barrier_names_its_local_device(self):
        context = DistributedContext(1, 1, 2, torch.device("cuda", 1))
        with patch("b200_experiment.distributed.dist.barrier") as barrier:
            context.barrier()
        barrier.assert_called_once_with(device_ids=[1])

    def test_standalone_child_drops_all_torchrun_rendezvous_metadata(self):
        source = {
            "PATH": "/bin",
            "CUDA_VISIBLE_DEVICES": "2,4",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "2",
            "GROUP_WORLD_SIZE": "1",
            "ROLE_NAME": "default",
            "MASTER_ADDR": "10.0.0.2",
            "MASTER_PORT": "29500",
            "TORCHELASTIC_USE_AGENT_STORE": "True",
            "TORCHELASTIC_RESTART_COUNT": "0",
            "TORCHELASTIC_FUTURE_FIELD": "must-also-be-removed",
        }

        isolated = isolate_distributed_subprocess_environment(source)

        self.assertEqual(isolated, {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "2,4"})
        self.assertIn("TORCHELASTIC_USE_AGENT_STORE", source)

    def test_production_batch_layout_is_explicit(self):
        layout = batch_layout(64, 1, 2, 8, 16)
        self.assertEqual(layout.global_prompt_batch_size, 64)
        self.assertEqual(layout.local_prompt_batch_size, 32)
        self.assertEqual(layout.global_trajectory_batch_size, 64)
        self.assertEqual(layout.local_trajectory_batch_size, 32)
        self.assertEqual(layout.ppo_mini_batch_size, 16)
        self.assertEqual(layout.optimizer_steps_per_full_rollout, 4)
        self.assertEqual(layout.micro_batch_size_per_gpu, 8)
        self.assertEqual(layout.micro_batches_per_gpu, 4)

    def test_tail_padding_preserves_every_real_sample_exactly_once(self):
        indices = list(range(5))
        rank_batches = [padded_local_indices(indices, rank, 2) for rank in range(2)]
        self.assertEqual([len(batch) for batch, _ in rank_batches], [3, 3])
        real = [
            index
            for (batch, active) in rank_batches
            for index, is_active in zip(batch, active)
            if is_active
        ]
        self.assertEqual(real, indices)
        self.assertEqual(
            sum(not flag for _, flags in rank_batches for flag in flags), 1
        )
        singleton = [padded_local_indices([42], rank, 2) for rank in range(2)]
        self.assertEqual(singleton, [([42], [True]), ([42], [False])])

    def test_microbatch_8_splits_32_local_trajectories_into_four_forwards(self):
        model = _TinyCausalLM()
        rows = torch.tensor(
            [[1 + index % 7, 2, 3, 4] for index in range(32)], dtype=torch.long
        )
        rollout = RolloutBatch(
            input_ids=rows,
            attention_mask=torch.ones_like(rows),
            response_ids=rows[:, 2:],
            valid_mask=torch.ones((32, 2), dtype=torch.bool),
            rollout_log_probs=torch.zeros((32, 2)),
            prompt_width=2,
        )
        reference = _tiny_reference(model, rollout)
        model.forward_calls = 0
        _opd_train_step(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            rollout,
            rollout.valid_mask,
            reference,
            _tiny_config(8),
            torch.device("cpu"),
            DistributedContext(0, 0, 1, torch.device("cpu")),
        )
        self.assertEqual(model.forward_calls, 4)

    def test_scoring_and_training_forward_share_left_padding_position_ids(self):
        model = _TinyCausalLM()
        rollout = RolloutBatch(
            input_ids=torch.tensor([[0, 1, 2, 3, 4]]),
            attention_mask=torch.tensor([[0, 1, 1, 1, 1]]),
            response_ids=torch.tensor([[3, 4]]),
            valid_mask=torch.ones((1, 2), dtype=torch.bool),
            rollout_log_probs=torch.zeros((1, 2)),
            prompt_width=3,
        )
        scores = score_original_rollout(
            model,
            rollout,
            retain_response_logits=False,
            top_k=5,
            trim_padding=True,
        )
        scoring_position_ids = model.position_id_calls[-1]
        # Use the exact scoring support so this verifies the actual
        # differentiable _opd_train_step forward, not only the helper.
        reference = build_topk_opd_reference(
            scores.top_k_ids,
            scores.top_k_log_probs,
            scores.top_k_log_probs,
            rollout.valid_mask,
        )
        model.position_id_calls.clear()
        _opd_train_step(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            rollout,
            rollout.valid_mask,
            reference,
            _tiny_config(1),
            torch.device("cpu"),
            DistributedContext(0, 0, 1, torch.device("cpu")),
        )
        training_position_ids = model.position_id_calls[-1]

        self.assertTrue(torch.equal(scoring_position_ids, training_position_ids))
        self.assertTrue(
            torch.equal(training_position_ids, torch.tensor([[0, 0, 1, 2]]))
        )

    def test_gradient_accumulation_does_not_change_the_objective(self):
        torch.manual_seed(7)
        first = _TinyCausalLM()
        second = copy.deepcopy(first)
        rollout = _tiny_batch(0, 3)
        reference = _tiny_reference(first, rollout)
        for model, micro in ((first, 1), (second, 3)):
            _opd_train_step(
                model,
                torch.optim.SGD(model.parameters(), lr=0.05),
                rollout,
                rollout.valid_mask,
                reference,
                _tiny_config(micro),
                torch.device("cpu"),
                DistributedContext(0, 0, 1, torch.device("cpu")),
            )
        self.assertTrue(
            torch.allclose(
                _parameters(first), _parameters(second), atol=1e-7, rtol=1e-6
            )
        )

    def test_ppo_minibatches_step_twice_for_sixteen_expanded_trajectories(self):
        encoded = {
            "input_ids": torch.tensor(
                [[1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.long
            ),
            "attention_mask": torch.ones(4, 2, dtype=torch.long),
        }
        expanded, indices, response_indices = expand_prompt_batch(
            encoded, [0, 1, 2, 3], 4
        )
        self.assertEqual(expanded["input_ids"].shape[0], 16)
        self.assertEqual(len(indices), 16)
        self.assertEqual(response_indices, [0, 1, 2, 3] * 4)
        rows = torch.cat(
            [expanded["input_ids"], torch.full((16, 2), 4, dtype=torch.long)],
            dim=1,
        )
        rollout = RolloutBatch(
            input_ids=rows,
            attention_mask=torch.ones_like(rows),
            response_ids=rows[:, 2:],
            valid_mask=torch.ones((16, 2), dtype=torch.bool),
            rollout_log_probs=torch.zeros((16, 2)),
            prompt_width=2,
        )
        model = _TinyCausalLM()
        reference = _tiny_reference(model, rollout)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        original_step = optimizer.step
        step_calls = 0

        def counted_step(*args, **kwargs):
            nonlocal step_calls
            step_calls += 1
            return original_step(*args, **kwargs)

        optimizer.step = counted_step
        metrics = _opd_train_step(
            model,
            optimizer,
            rollout,
            rollout.valid_mask,
            reference,
            _tiny_config(2, 8),
            torch.device("cpu"),
            DistributedContext(0, 0, 1, torch.device("cpu")),
        )

        self.assertEqual(step_calls, 2)
        self.assertEqual(metrics["optimizer_steps"], 2)
        self.assertEqual(
            [item["ppo_minibatch_trajectory_count"] for item in metrics["minibatches"]],
            [8.0, 8.0],
        )

    def test_partial_ppo_minibatch_and_optimizer_step_schedule(self):
        # Prompt batches are 4, 4, 2; with n=4 this is 16, 16, 8
        # trajectories and therefore 2, 2, 1 optimizer steps at PPO size 8.
        self.assertEqual(_optimizer_steps_per_epoch(10, 4, 4, 8), 5)
        self.assertEqual(
            _rollout_position_after_optimizer_steps(1, 10, 4, 4, 8),
            (0, 1),
        )
        self.assertEqual(
            _rollout_position_after_optimizer_steps(4, 10, 4, 4, 8),
            (2, 0),
        )
        self.assertEqual(
            _rollout_position_after_optimizer_steps(5, 10, 4, 4, 8),
            (3, 0),
        )

        model = _TinyCausalLM()
        rows = torch.tensor(
            [[1 + index % 7, 2, 3, 4] for index in range(10)], dtype=torch.long
        )
        rollout = RolloutBatch(
            input_ids=rows,
            attention_mask=torch.ones_like(rows),
            response_ids=rows[:, 2:],
            valid_mask=torch.ones((10, 2), dtype=torch.bool),
            rollout_log_probs=torch.zeros((10, 2)),
            prompt_width=2,
        )
        metrics = _opd_train_step(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            rollout,
            rollout.valid_mask,
            _tiny_reference(model, rollout),
            _tiny_config(8, 8),
            torch.device("cpu"),
            DistributedContext(0, 0, 1, torch.device("cpu")),
        )
        self.assertEqual(metrics["optimizer_steps"], 2)
        self.assertEqual(
            [item["ppo_minibatch_trajectory_count"] for item in metrics["minibatches"]],
            [8.0, 2.0],
        )

    def test_microbatch_partition_does_not_change_two_ppo_updates(self):
        torch.manual_seed(17)
        reference_model = _TinyCausalLM()
        microbatched_model = copy.deepcopy(reference_model)
        rows = torch.tensor(
            [[1 + index % 7, 2, 3, 4] for index in range(16)], dtype=torch.long
        )
        rollout = RolloutBatch(
            input_ids=rows,
            attention_mask=torch.ones_like(rows),
            response_ids=rows[:, 2:],
            valid_mask=torch.ones((16, 2), dtype=torch.bool),
            rollout_log_probs=torch.zeros((16, 2)),
            prompt_width=2,
        )
        reference = _tiny_reference(reference_model, rollout)
        for model, micro in ((reference_model, 8), (microbatched_model, 2)):
            metrics = _opd_train_step(
                model,
                torch.optim.SGD(model.parameters(), lr=0.02),
                rollout,
                rollout.valid_mask,
                reference,
                _tiny_config(micro, 8),
                torch.device("cpu"),
                DistributedContext(0, 0, 1, torch.device("cpu")),
            )
            self.assertEqual(metrics["optimizer_steps"], 2)
        self.assertTrue(
            torch.allclose(
                _parameters(reference_model),
                _parameters(microbatched_model),
                atol=1e-7,
                rtol=1e-6,
            )
        )

    def test_full_opd_reduction_is_global_token_mean(self):
        elementwise = torch.tensor([[1.0, 3.0, 99.0], [10.0, 20.0, 30.0]])
        full_valid_mask = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )

        numerator, denominator = weighted_token_sums(
            elementwise, full_valid_mask.float(), full_valid_mask
        )

        self.assertEqual(float(numerator), 64.0)
        self.assertEqual(float(denominator), 5.0)
        self.assertAlmostEqual(float(numerator / denominator), 12.8, places=6)

    def test_variable_length_collective_uses_rank_order(self):
        if not dist.is_gloo_available():
            self.skipTest("PyTorch was built without Gloo")
        with tempfile.TemporaryDirectory() as temporary:
            rendezvous = str(Path(temporary) / "rendezvous")
            mp.spawn(_gloo_gather_worker, args=(rendezvous,), nprocs=2, join=True)

    def test_uneven_ddp_batch_matches_single_process_global_mean(self):
        if not dist.is_gloo_available():
            self.skipTest("PyTorch was built without Gloo")
        torch.manual_seed(99)
        reference = _TinyCausalLM()
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        rollout = _tiny_batch(0, 3)
        opd_reference = _tiny_reference(reference, rollout)
        _opd_train_step(
            reference,
            optimizer,
            rollout,
            rollout.valid_mask,
            opd_reference,
            _tiny_config(),
            torch.device("cpu"),
            DistributedContext(0, 0, 1, torch.device("cpu")),
        )
        expected = _parameters(reference)
        with tempfile.TemporaryDirectory() as temporary:
            rendezvous = str(Path(temporary) / "ddp-rendezvous")
            mp.spawn(
                _ddp_step_worker,
                args=(rendezvous, temporary),
                nprocs=2,
                join=True,
            )
            rank0 = torch.load(Path(temporary) / "rank-0.pt", weights_only=True)
            rank1 = torch.load(Path(temporary) / "rank-1.pt", weights_only=True)
        self.assertTrue(torch.allclose(rank0, rank1, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(rank0, expected, atol=1e-7, rtol=1e-6))

    def test_contiguous_partitions_cover_without_padding_or_repetition(self):
        for total, world_size in ((64, 1), (64, 2), (64, 3), (53, 4)):
            partitions = [
                contiguous_partition(total, rank, world_size)
                for rank in range(world_size)
            ]
            flattened = [
                index for start, end in partitions for index in range(start, end)
            ]
            self.assertEqual(flattened, list(range(total)))
            lengths = [end - start for start, end in partitions]
            self.assertLessEqual(max(lengths) - min(lengths), 1)

    def test_world_one_global_ta_path_matches_original_normalization(self):
        torch.manual_seed(123)
        p = torch.softmax(torch.randn(3, 4, 37), dim=-1)
        q = torch.softmax(torch.randn(3, 4, 37), dim=-1)
        valid = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool
        )
        selector = TASelector(top_k=8)
        expected = selector.compute_scores(p, q, valid)
        raw = selector.compute_scores(p, q, valid, normalize=False)
        context = DistributedContext(0, 0, 1, torch.device("cpu"))
        actual, global_diagnostics, start, end = _globalize_ta_output(
            raw, valid, selector, context
        )
        self.assertEqual((start, end), (0, int(valid.sum())))
        self.assertTrue(torch.equal(actual.scores, expected.scores))
        self.assertTrue(torch.equal(global_diagnostics["s_TA"], expected.scores[valid]))

    def test_global_budget_is_not_rounded_per_worker(self):
        scores = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6])
        rho = 0.33
        expected = math.ceil(rho * scores.numel())
        first_valid = torch.ones(1, 3, dtype=torch.bool)
        second_valid = torch.ones(1, 4, dtype=torch.bool)
        first, global_selected = _local_mask_from_global_budget(
            scores, first_valid, 0, 3, rho
        )
        second, repeated_global = _local_mask_from_global_budget(
            scores, second_valid, 3, 7, rho
        )
        self.assertTrue(torch.equal(global_selected, repeated_global))
        self.assertEqual(int(first.sum() + second.sum()), expected)


if __name__ == "__main__":
    unittest.main()
