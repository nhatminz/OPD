from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from b200_experiment.distributed import (
    DistributedContext,
    contiguous_partition,
    unique_free_port,
)
from b200_experiment.selectors import TASelector
from b200_experiment.trainer import (
    _globalize_ta_output,
    _local_mask_from_global_budget,
    _opd_train_step,
    _sum_per_response_masked_means,
)
from b200_experiment.scoring import RolloutBatch


class _TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 5)
        self.output = nn.Linear(5, 11, bias=False)
        self.config = type("Config", (), {"use_cache": False})()

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        del attention_mask, use_cache, return_dict
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


def _tiny_config() -> dict:
    return {
        "training": {
            "micro_batch_size": 3,
            "ppo_clip_low": 0.2,
            "ppo_clip_high": 0.28,
            "max_grad_norm": 100.0,
        }
    }


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
        old = torch.zeros_like(rollout.response_ids, dtype=torch.float32)
        teacher = torch.full_like(old, 0.2)
        context = DistributedContext(rank, rank, 2, torch.device("cpu"))
        _opd_train_step(
            wrapped,
            optimizer,
            rollout,
            rollout.valid_mask,
            old,
            teacher,
            _tiny_config(),
            torch.device("cpu"),
            context,
            3,
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
    def test_full_opd_reduction_means_each_variable_length_response_first(self):
        elementwise = torch.tensor([[1.0, 3.0, 99.0], [10.0, 20.0, 30.0]])
        full_valid_mask = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )

        summed_response_means = _sum_per_response_masked_means(
            elementwise, full_valid_mask
        )

        self.assertEqual(float(summed_response_means), 22.0)
        self.assertEqual(float(summed_response_means / 2), 11.0)

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
        old = torch.zeros_like(rollout.response_ids, dtype=torch.float32)
        teacher = torch.full_like(old, 0.2)
        _opd_train_step(
            reference,
            optimizer,
            rollout,
            rollout.valid_mask,
            old,
            teacher,
            _tiny_config(),
            torch.device("cpu"),
            DistributedContext(0, 0, 1, torch.device("cpu")),
            3,
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
