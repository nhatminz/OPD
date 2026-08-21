from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from b200_experiment.scoring import RolloutBatch, score_original_rollout


class _FakeModel:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def eval(self):
        return self

    def __call__(self, **_kwargs):
        return SimpleNamespace(logits=self.logits, past_key_values=None)


class ScoringTests(unittest.TestCase):
    def test_pure_opd_can_score_sampled_tokens_without_retaining_logits(self):
        torch.manual_seed(5)
        logits = torch.randn(2, 5, 11)
        rollout = RolloutBatch(
            input_ids=torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            attention_mask=torch.ones(2, 5, dtype=torch.long),
            response_ids=torch.tensor([[3, 4, 5], [8, 9, 10]]),
            valid_mask=torch.ones(2, 3, dtype=torch.bool),
            rollout_log_probs=torch.zeros(2, 3),
            prompt_width=3,
        )
        retained = score_original_rollout(
            _FakeModel(logits), rollout, score_chunk_steps=2
        )
        compact = score_original_rollout(
            _FakeModel(logits),
            rollout,
            score_chunk_steps=2,
            retain_response_logits=False,
        )

        self.assertIsNotNone(retained.response_logits)
        self.assertIsNone(compact.response_logits)
        self.assertTrue(
            torch.equal(compact.sampled_log_probs, retained.sampled_log_probs)
        )
        self.assertTrue(
            torch.equal(compact.log_normalizers, retained.log_normalizers)
        )


if __name__ == "__main__":
    unittest.main()
