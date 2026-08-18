from __future__ import annotations

import unittest

from b200_experiment.trainer import should_run_training_evaluation


class TrainingEvaluationScheduleTests(unittest.TestCase):
    def test_step_zero_interval_and_final_are_evaluated_once(self):
        settings = {
            "enabled": True,
            "interval_steps": 100,
            "eval_at_start": True,
            "eval_at_end": True,
        }
        selected = [
            step
            for step in range(273)
            if should_run_training_evaluation(step, 272, settings)
        ]
        self.assertEqual(selected, [0, 100, 200, 272])

    def test_autotune_can_disable_all_periodic_evaluation(self):
        settings = {"enabled": False, "interval_steps": 100}
        self.assertFalse(should_run_training_evaluation(0, 1, settings))
        self.assertFalse(should_run_training_evaluation(1, 1, settings))

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            should_run_training_evaluation(
                1, 10, {"enabled": True, "interval_steps": 0}
            )


if __name__ == "__main__":
    unittest.main()
