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

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        rows = torch.where(input_ids[:, 0].eq(1), 0, 1)
        return SimpleNamespace(
            logits=self.logits.index_select(0, rows), past_key_values=None
        )


class ScoringTests(unittest.TestCase):
    def test_pure_opd_scores_topk_without_retaining_full_logits(self):
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
            top_k=4,
            micro_batch_size=1,
        )

        self.assertIsNotNone(retained.response_logits)
        self.assertIsNone(compact.response_logits)
        self.assertTrue(
            torch.equal(compact.sampled_log_probs, retained.sampled_log_probs)
        )
        self.assertTrue(
            torch.equal(compact.log_normalizers, retained.log_normalizers)
        )
        self.assertEqual(compact.top_k_ids.shape, (2, 3, 4))
        expected_log_probs = torch.log_softmax(logits[:, 2:5].float(), dim=-1)
        self.assertTrue(
            torch.allclose(
                compact.top_k_log_probs,
                expected_log_probs.gather(-1, compact.top_k_ids),
            )
        )

        teacher = score_original_rollout(
            _FakeModel(logits + 0.5),
            rollout,
            score_chunk_steps=2,
            retain_response_logits=False,
            top_k=4,
            candidate_ids=compact.top_k_ids,
        )
        self.assertEqual(teacher.candidate_log_probs.shape, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
