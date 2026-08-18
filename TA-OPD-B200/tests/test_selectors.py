from __future__ import annotations

import math
import unittest

import torch

from b200_experiment.selectors import RACSelector, TASelector, top_budget_mask


class SelectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.p = torch.softmax(torch.randn(3, 5, 41), dim=-1)
        self.q = torch.softmax(torch.randn(3, 5, 41), dim=-1)
        self.valid = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool
        )

    @staticmethod
    def probe(rows, tokens):
        return (0.1 + 0.8 * ((rows * 17 + tokens) % 101).float() / 100).detach()

    def test_rac_definition_bounds_and_no_grad(self):
        output = RACSelector(top_k=8, branch_m=2).compute_scores(
            self.p, self.q, self.valid, self.probe
        )
        expected_delta = (self.q[self.valid] - self.p[self.valid]).clamp_min(0).sum(-1)
        self.assertTrue(
            torch.allclose(
                output.diagnostics["Delta"][self.valid], expected_delta, atol=1e-7
            )
        )
        self.assertTrue(
            torch.allclose(
                output.scores,
                output.diagnostics["Delta"]
                * output.diagnostics["A"]
                * output.diagnostics["F"],
            )
        )
        for key in ("Delta", "A", "F", "B"):
            values = output.diagnostics[key][self.valid]
            self.assertGreaterEqual(float(values.min()), -1e-6)
            self.assertLessEqual(float(values.max()), 1.0 + 1e-6)
        self.assertFalse(output.scores.requires_grad)
        self.assertIsNone(output.scores.grad_fn)

    def test_shared_global_budget_is_exact_under_ties(self):
        ta = TASelector(top_k=8).compute_scores(self.p, self.q, self.valid).scores
        rac = (
            RACSelector(top_k=8, branch_m=2)
            .compute_scores(self.p, self.q, self.valid, self.probe)
            .scores
        )
        for rho in (0.05, 0.10, 0.33, 1.0):
            expected = math.ceil(rho * int(self.valid.sum()))
            self.assertEqual(int(top_budget_mask(ta, self.valid, rho).sum()), expected)
            self.assertEqual(int(top_budget_mask(rac, self.valid, rho).sum()), expected)
        tied = top_budget_mask(torch.zeros_like(ta), self.valid, 0.33).reshape(-1)
        indices = self.valid.reshape(-1).nonzero().squeeze(-1)
        self.assertTrue(
            torch.equal(
                tied.nonzero().squeeze(-1), indices[: math.ceil(0.33 * len(indices))]
            )
        )

    def test_ta_matches_frozen_official_release_fixture(self):
        # Frozen from wyy-code/TA-OPD's KLf_union/Cmass plus batch-quantile
        # Dlearn path, so this standalone test never imports the old project.
        expected = torch.tensor(
            [
                0.20581491,
                0.0,
                0.22721194,
                0.13622078,
                0.14612526,
                0.20439357,
                0.27680814,
                0.0,
                0.0,
            ]
        )
        actual = (
            TASelector(top_k=8)
            .compute_scores(self.p, self.q, self.valid)
            .scores[self.valid]
        )
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-5))

    def test_identical_models_have_zero_rac(self):
        output = RACSelector(top_k=8, branch_m=2).compute_scores(
            self.p, self.p, self.valid, self.probe
        )
        self.assertTrue(
            torch.equal(output.scores[self.valid], torch.zeros(int(self.valid.sum())))
        )


if __name__ == "__main__":
    unittest.main()
