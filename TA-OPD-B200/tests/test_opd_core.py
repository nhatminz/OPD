from __future__ import annotations

import unittest

import torch

from b200_experiment.opd_core import (
    build_topk_opd_reference,
    topk_candidate_ppo_loss,
    topk_reference_from_logits,
    weighted_token_sums,
)


class TopKOPDCoreTests(unittest.TestCase):
    def test_only_stu_student_p_matches_upstream_formula_with_k16(self):
        torch.manual_seed(7)
        student_logits = torch.randn(2, 3, 29)
        teacher_logits = torch.randn(2, 3, 29)
        valid = torch.tensor([[True, True, False], [True, True, True]])

        reference = topk_reference_from_logits(
            student_logits, teacher_logits, valid, top_k=16
        )

        student_all = torch.log_softmax(student_logits.float(), dim=-1)
        teacher_all = torch.log_softmax(teacher_logits.float(), dim=-1)
        ids = torch.topk(student_all, k=16, dim=-1).indices
        student_on_s = student_all.gather(-1, ids)
        teacher_on_s = teacher_all.gather(-1, ids)
        upstream_weights = torch.softmax(student_on_s, dim=-1)
        upstream_reward = -(student_on_s - teacher_on_s) * upstream_weights
        upstream_reward = upstream_reward * valid.unsqueeze(-1)

        self.assertEqual(reference.candidate_ids.shape[-1], 16)
        self.assertTrue(torch.equal(reference.candidate_ids, ids))
        self.assertTrue(
            torch.allclose(reference.advantages, upstream_reward, atol=1e-6)
        )

    def test_policy_loss_uses_all_candidates_not_only_sampled_token(self):
        student = torch.log_softmax(torch.arange(16, dtype=torch.float32), dim=0)
        teacher = torch.log_softmax(torch.arange(15, -1, -1, dtype=torch.float32), dim=0)
        student = student.reshape(1, 1, 16)
        teacher = teacher.reshape(1, 1, 16)
        valid = torch.ones(1, 1, dtype=torch.bool)
        reference = build_topk_opd_reference(
            torch.arange(16).reshape(1, 1, 16), student, teacher, valid
        )
        current = student.clone().requires_grad_(True)

        loss = topk_candidate_ppo_loss(
            current, reference, clip_low=0.2, clip_high=0.28, dual_clip=3.0
        )
        loss.sum().backward()

        self.assertEqual(current.grad.shape[-1], 16)
        self.assertEqual(int(current.grad.ne(0).sum()), 16)
        expected = -reference.advantages.sum(dim=-1)
        self.assertTrue(torch.allclose(loss.detach(), expected, atol=1e-6))

    def test_all_methods_share_signal_and_only_position_weights_change(self):
        per_position = torch.tensor([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
        valid = torch.tensor([[True, True, False], [True, True, True]])
        allocations = {
            "opd": torch.ones_like(per_position),
            "ta": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "rac": torch.tensor([[0.1, 0.2, 0.0], [0.4, 0.8, 1.0]]),
        }
        for weights in allocations.values():
            numerator, denominator = weighted_token_sums(
                per_position, weights, valid
            )
            expected_weights = weights * valid
            self.assertTrue(
                torch.allclose(
                    numerator / denominator,
                    (per_position * expected_weights).sum()
                    / expected_weights.sum(),
                )
            )

    def test_global_token_mean_and_ddp_shards_match_single_process(self):
        losses = torch.tensor([[1.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
        valid = torch.tensor([[True, True, False], [True, True, True]])
        cases = (
            torch.ones_like(losses),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
            torch.tensor([[0.2, 0.4, 0.0], [0.3, 0.7, 1.0]]),
        )
        for weights in cases:
            full_num, full_den = weighted_token_sums(losses, weights, valid)
            shard0 = weighted_token_sums(losses[:1], weights[:1], valid[:1])
            shard1 = weighted_token_sums(losses[1:], weights[1:], valid[1:])
            ddp_value = (shard0[0] + shard1[0]) / (shard0[1] + shard1[1])
            self.assertTrue(torch.allclose(ddp_value, full_num / full_den))


if __name__ == "__main__":
    unittest.main()
