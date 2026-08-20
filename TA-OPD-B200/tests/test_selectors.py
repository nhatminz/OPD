from __future__ import annotations

import math
import unittest

import torch

from b200_experiment.selectors import (
    RACSelector,
    TASelector,
    bellman_parallel_scan,
    bellman_reference_scan,
    top_budget_mask,
)
from b200_experiment.selectors.base import robust_quantile_normalize


def _literal_ta_reference(p, q, valid, top_k):
    disagreements, compatibilities = [], []
    for p_row, q_row in zip(p[valid], q[valid]):
        student_ids = torch.topk(p_row, top_k).indices
        teacher_ids = torch.topk(q_row, top_k).indices
        union = torch.unique(torch.cat((student_ids, teacher_ids)), sorted=False)
        p_union, q_union = p_row[union], q_row[union]
        p_union, q_union = p_union / p_union.sum(), q_union / q_union.sum()
        disagreements.append((q_union * (q_union.log() - p_union.log())).sum())
        compatibilities.append(q_row[student_ids].sum())
    d, c = torch.stack(disagreements), torch.stack(compatibilities)
    return robust_quantile_normalize(d) * robust_quantile_normalize(c)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.p = torch.softmax(torch.randn(3, 5, 41), dim=-1)
        self.q = torch.softmax(torch.randn(3, 5, 41), dim=-1)
        self.valid = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool
        )

    def test_ta_matches_literal_union_kl_and_teacher_mass_definition(self):
        output = TASelector(top_k=8).compute_scores(self.p, self.q, self.valid)
        expected = _literal_ta_reference(self.p, self.q, self.valid, 8)
        self.assertTrue(torch.allclose(output.scores[self.valid], expected, atol=2e-6))
        student_ids = torch.topk(self.p[self.valid], 8, dim=-1).indices
        expected_c = self.q[self.valid].gather(-1, student_ids).sum(-1)
        self.assertTrue(
            torch.allclose(output.diagnostics["C"][self.valid], expected_c, atol=1e-7)
        )

    def test_compact_logits_path_matches_probability_reference(self):
        student_logits = self.p.log()
        teacher_logits = self.q.log()
        zeros = torch.zeros(self.p.shape[:2])
        selector = TASelector(top_k=8)
        reference = selector.compute_scores(self.p, self.q, self.valid)
        compact = selector.compute_scores_from_logits(
            student_logits,
            teacher_logits,
            zeros,
            zeros,
            self.valid,
            token_chunk_size=2,
        )
        for key in ("D", "C", "D_norm", "C_norm", "s_TA"):
            self.assertTrue(
                torch.allclose(
                    compact.diagnostics[key],
                    reference.diagnostics[key],
                    atol=2e-6,
                    rtol=2e-6,
                ),
                key,
            )

    def test_hand_computable_bellman_recurrence(self):
        g = torch.tensor([[0.2, 0.4, 1.0]])
        alignment = torch.tensor([[0.5, 1.0, 0.25]])
        valid = torch.ones_like(g, dtype=torch.bool)
        returns, masses, values = bellman_reference_scan(
            g, alignment, valid, gamma=0.5
        )
        expected_r = torch.tensor([[0.425, 0.9, 1.0]])
        expected_m = torch.tensor([[1.375, 1.5, 1.0]])
        self.assertTrue(torch.allclose(returns, expected_r))
        self.assertTrue(torch.allclose(masses, expected_m))
        self.assertTrue(torch.allclose(values, expected_r / expected_m))

    def test_parallel_scan_matches_reference_with_padding_and_resets(self):
        torch.manual_seed(11)
        g = torch.rand(4, 17)
        alignment = torch.rand(4, 17)
        valid = torch.tensor(
            [
                [1] * 17,
                [1] * 9 + [0] * 8,
                [1, 1, 0, 1, 1] + [0] * 12,
                [0] * 17,
            ],
            dtype=torch.bool,
        )
        reference = bellman_reference_scan(g, alignment, valid, 0.995)
        optimized = bellman_parallel_scan(g, alignment, valid, 0.995)
        for expected, actual in zip(reference, optimized):
            self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))
        # The gap at position 2 must prevent positions 3-4 leaking backward.
        isolated = bellman_parallel_scan(g[:, :2], alignment[:, :2], valid[:, :2], 0.995)
        self.assertTrue(torch.allclose(optimized[2][2, :2], isolated[2][2, :2]))

    def test_bellman_rac_bounds_padding_and_detachment(self):
        g = TASelector(top_k=8).compute_scores(self.p, self.q, self.valid).scores
        student_logp = torch.log(
            self.p.gather(-1, torch.zeros((*self.p.shape[:2], 1), dtype=torch.long)).squeeze(-1)
        ).requires_grad_()
        teacher_logp = torch.log(
            self.q.gather(-1, torch.zeros((*self.q.shape[:2], 1), dtype=torch.long)).squeeze(-1)
        )
        output = RACSelector().compute_scores(
            g.detach().clone().requires_grad_(), student_logp, teacher_logp, self.valid
        )
        diagnostics = output.diagnostics
        for key in ("g", "V", "z"):
            values = diagnostics[key][self.valid]
            self.assertGreaterEqual(float(values.min()), -1e-6)
            self.assertLessEqual(float(values.max()), 1.0 + 1e-6)
        alignment = diagnostics["alignment"][self.valid]
        self.assertGreater(float(alignment.min()), 0.0)
        self.assertLessEqual(float(alignment.max()), 1.0)
        weights = diagnostics["w"][self.valid]
        self.assertGreaterEqual(float(weights.min()), 0.10 - 1e-6)
        self.assertLessEqual(float(weights.max()), 1.0 + 1e-6)
        for key in ("g", "alignment", "R", "M", "V", "z", "w"):
            self.assertTrue(torch.equal(diagnostics[key][~self.valid], torch.zeros_like(diagnostics[key][~self.valid])))
            self.assertFalse(diagnostics[key].requires_grad)
            self.assertIsNone(diagnostics[key].grad_fn)

    def test_rac_weighted_loss_only_backpropagates_through_opd_losses(self):
        weights = RACSelector(w_min=0.2).compute_scores(
            torch.tensor([[0.0, 0.5, 1.0]]),
            torch.tensor([[-1.0, -1.0, -1.0]]),
            torch.tensor([[-1.0, -2.0, -0.5]]),
            torch.ones(1, 3, dtype=torch.bool),
        ).scores
        opd = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        loss = (weights * opd).sum() / weights.sum()
        loss.backward()
        self.assertTrue(torch.allclose(opd.grad, weights / weights.sum()))
        self.assertFalse(weights.requires_grad)

    def test_ta_hard_global_budget_is_exact_under_ties(self):
        ta = TASelector(top_k=8).compute_scores(self.p, self.q, self.valid).scores
        for rho in (0.05, 0.10, 0.33, 1.0):
            expected = math.ceil(rho * int(self.valid.sum()))
            self.assertEqual(int(top_budget_mask(ta, self.valid, rho).sum()), expected)
        tied = top_budget_mask(torch.zeros_like(ta), self.valid, 0.33).reshape(-1)
        indices = self.valid.reshape(-1).nonzero().squeeze(-1)
        self.assertTrue(
            torch.equal(tied.nonzero().squeeze(-1), indices[: math.ceil(0.33 * len(indices))])
        )


if __name__ == "__main__":
    unittest.main()
