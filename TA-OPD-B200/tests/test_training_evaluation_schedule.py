from __future__ import annotations

import unittest

from b200_experiment.eval_schedule import (
    should_run_training_evaluation,
    training_evaluation_steps,
)


class TrainingEvaluationScheduleTests(unittest.TestCase):
    def test_target_count_is_spread_evenly_including_endpoints(self):
        settings = {
            "enabled": True,
            "target_evaluations": 16,
            "interval_steps": None,
            "eval_at_start": True,
            "eval_at_end": True,
        }
        selected = [
            step
            for step in range(273)
            if should_run_training_evaluation(step, 272, settings)
        ]
        self.assertEqual(len(selected), 16)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 272)
        gaps = [right - left for left, right in zip(selected, selected[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 1)

    def test_fixed_interval_remains_an_explicit_fallback(self):
        settings = {
            "enabled": True,
            "target_evaluations": None,
            "interval_steps": 100,
            "eval_at_start": True,
            "eval_at_end": True,
        }
        self.assertEqual(training_evaluation_steps(272, settings), (0, 100, 200, 272))

    def test_autotune_can_disable_all_periodic_evaluation(self):
        settings = {"enabled": False, "target_evaluations": 16}
        self.assertFalse(should_run_training_evaluation(0, 1, settings))
        self.assertFalse(should_run_training_evaluation(1, 1, settings))

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            should_run_training_evaluation(
                1, 10, {"enabled": True, "interval_steps": 0}
            )

    def test_invalid_target_is_rejected(self):
        with self.assertRaises(ValueError):
            training_evaluation_steps(10, {"enabled": True, "target_evaluations": 0})


if __name__ == "__main__":
    unittest.main()
