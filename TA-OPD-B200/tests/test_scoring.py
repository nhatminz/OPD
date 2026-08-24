from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from b200_experiment.scoring import (
    RolloutBatch,
    score_original_rollout,
    score_student_teacher_rollout,
)


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


class _RecordingQwenModel(_FakeModel):
    def __init__(self, logits: torch.Tensor):
        super().__init__(logits)
        self.config = SimpleNamespace(model_type="qwen3")
        self.calls = []
        self.input_id_calls = []

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        self.input_id_calls.append(input_ids.detach().clone())
        rows = input_ids[:, 0].long() - 1
        logits = self.logits.index_select(0, rows)[:, : input_ids.shape[1]]
        logits_to_keep = kwargs.get("logits_to_keep")
        self.calls.append((input_ids.shape[0], input_ids.shape[1], logits_to_keep))
        if logits_to_keep is not None:
            logits = logits[:, -int(logits_to_keep) :]
        return SimpleNamespace(logits=logits, past_key_values=None)


class ScoringTests(unittest.TestCase):
    def test_joint_bidirectional_scoring_matches_three_independent_forwards(self):
        torch.manual_seed(23)
        student_logits = torch.randn(4, 7, 17)
        teacher_logits = torch.randn(4, 7, 17)
        rollout = RolloutBatch(
            input_ids=torch.tensor(
                [
                    [1, 10, 11, 5, 0, 0, 0],
                    [2, 10, 11, 6, 7, 8, 9],
                    [3, 10, 11, 4, 5, 0, 0],
                    [4, 10, 11, 3, 4, 5, 0],
                ]
            ),
            attention_mask=torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0],
                ]
            ),
            response_ids=torch.tensor(
                [[5, 0, 0, 0], [6, 7, 8, 9], [4, 5, 0, 0], [3, 4, 5, 0]]
            ),
            valid_mask=torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, True, True],
                    [True, True, False, False],
                    [True, True, True, False],
                ]
            ),
            rollout_log_probs=torch.zeros(4, 4),
            prompt_width=3,
        )
        student_temperature, teacher_temperature = 0.8, 1.2
        independent_student = score_original_rollout(
            _RecordingQwenModel(student_logits),
            rollout,
            retain_response_logits=False,
            top_k=5,
            temperature=student_temperature,
            micro_batch_size=2,
        )
        independent_teacher = score_original_rollout(
            _RecordingQwenModel(teacher_logits),
            rollout,
            retain_response_logits=False,
            top_k=5,
            candidate_ids=independent_student.top_k_ids,
            temperature=teacher_temperature,
            micro_batch_size=2,
        )
        independent_cross = score_original_rollout(
            _RecordingQwenModel(student_logits),
            rollout,
            retain_response_logits=False,
            candidate_ids=independent_teacher.top_k_ids,
            temperature=student_temperature,
            micro_batch_size=2,
        )
        joint_student_model = _RecordingQwenModel(student_logits)
        joint_teacher_model = _RecordingQwenModel(teacher_logits)
        joint_student, joint_teacher = score_student_teacher_rollout(
            joint_student_model,
            joint_teacher_model,
            rollout,
            top_k=5,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
            micro_batch_size=2,
        )

        # Teacher scoring consumes the exact same student-tokenizer IDs. There
        # is deliberately no decode -> teacher-tokenize boundary.
        self.assertEqual(
            len(joint_student_model.input_id_calls),
            len(joint_teacher_model.input_id_calls),
        )
        for student_ids, teacher_ids in zip(
            joint_student_model.input_id_calls,
            joint_teacher_model.input_id_calls,
        ):
            self.assertTrue(torch.equal(student_ids, teacher_ids))

        valid = rollout.valid_mask
        self.assertTrue(
            torch.equal(
                joint_student.top_k_ids[valid], independent_student.top_k_ids[valid]
            )
        )
        self.assertTrue(
            torch.equal(
                joint_teacher.top_k_ids[valid], independent_teacher.top_k_ids[valid]
            )
        )
        for actual, expected in (
            (joint_student.sampled_log_probs, independent_student.sampled_log_probs),
            (joint_student.top_k_log_probs, independent_student.top_k_log_probs),
            (
                joint_student.candidate_log_probs,
                independent_cross.candidate_log_probs,
            ),
            (joint_teacher.sampled_log_probs, independent_teacher.sampled_log_probs),
            (joint_teacher.top_k_log_probs, independent_teacher.top_k_log_probs),
            (
                joint_teacher.candidate_log_probs,
                independent_teacher.candidate_log_probs,
            ),
        ):
            self.assertTrue(torch.allclose(actual[valid], expected[valid], atol=1e-6))

    def test_length_bucketing_trims_padding_and_keeps_valid_scores_exact(self):
        torch.manual_seed(17)
        logits = torch.randn(4, 7, 13)
        rollout = RolloutBatch(
            input_ids=torch.tensor(
                [
                    [1, 10, 11, 5, 0, 0, 0],
                    [2, 10, 11, 6, 7, 8, 9],
                    [3, 10, 11, 4, 5, 0, 0],
                    [4, 10, 11, 3, 4, 5, 0],
                ]
            ),
            attention_mask=torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0],
                ]
            ),
            response_ids=torch.tensor(
                [[5, 0, 0, 0], [6, 7, 8, 9], [4, 5, 0, 0], [3, 4, 5, 0]]
            ),
            valid_mask=torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, True, True],
                    [True, True, False, False],
                    [True, True, True, False],
                ]
            ),
            rollout_log_probs=torch.zeros(4, 4),
            prompt_width=3,
        )
        optimized_model = _RecordingQwenModel(logits)
        optimized = score_original_rollout(
            optimized_model,
            rollout,
            retain_response_logits=False,
            top_k=4,
            micro_batch_size=2,
        )
        reference = score_original_rollout(
            _RecordingQwenModel(logits),
            rollout,
            retain_response_logits=False,
            top_k=4,
            micro_batch_size=2,
            trim_padding=False,
            length_bucketed=False,
        )

        # Sorted buckets contain lengths [4,3] then [2,1]. The last sampled
        # input token and all right padding are omitted as causally irrelevant.
        self.assertEqual(optimized_model.calls, [(2, 6, 4), (2, 4, 2)])
        valid = rollout.valid_mask
        self.assertTrue(
            torch.allclose(
                optimized.sampled_log_probs[valid],
                reference.sampled_log_probs[valid],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(optimized.top_k_ids[valid], reference.top_k_ids[valid])
        )
        self.assertTrue(
            torch.allclose(
                optimized.top_k_log_probs[valid],
                reference.top_k_log_probs[valid],
                atol=1e-6,
            )
        )

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
        self.assertTrue(torch.equal(compact.log_normalizers, retained.log_normalizers))
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
