from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from b200_experiment.tensorboard_logging import (
    BASE_TAGS,
    RAC_TAGS,
    TA_TAGS,
    TensorBoardLogger,
    production_tensorboard_metrics,
)


def _metrics() -> dict:
    values = {field: float(index + 1) for index, field in enumerate(BASE_TAGS.values())}
    values.update(
        selector={
            "selected_fraction": 0.125,
            "valid_tokens": 20,
            "effective_sample_size": 15.0,
            "D": {"mean": 0.2},
            "C": {"mean": 0.3},
            "s_TA": {"mean": 0.4, "std": 0.05},
            "g": {"mean": 0.4},
            "alignment": {"mean": 0.6},
            "V": {"mean": 0.7, "std": 0.08},
            "w": {"mean": 0.8, "std": 0.09, "min": 0.1, "max": 1.0},
        },
        vllm_logprob_sanity={"enabled": False},
        deliberately_unlogged_metric=999.0,
    )
    return values


class TensorBoardMetricTests(unittest.TestCase):
    def test_opd_writes_all_common_diagnostic_tags(self):
        selected = production_tensorboard_metrics(_metrics(), "opd")
        self.assertEqual(set(selected), set(BASE_TAGS))
        self.assertNotIn("distillation/kl_mean", selected)
        self.assertIn("distillation/topk_divergence_proxy_mean", selected)

    def test_ta_adds_requested_selector_diagnostics(self):
        selected = production_tensorboard_metrics(_metrics(), "ta")
        self.assertEqual(
            set(selected),
            set(BASE_TAGS) | set(TA_TAGS) | {"ta/selected_token_fraction"},
        )
        self.assertEqual(selected["ta/selected_token_fraction"], 0.125)
        self.assertEqual(selected["ta/teachability_std"], 0.05)

    def test_rac_effective_fraction_is_normalized(self):
        selected = production_tensorboard_metrics(_metrics(), "rac")
        self.assertEqual(
            set(selected),
            set(BASE_TAGS) | set(RAC_TAGS) | {"rac/effective_token_fraction"},
        )
        self.assertEqual(selected["rac/effective_token_fraction"], 0.75)
        self.assertEqual(selected["rac/weight_min"], 0.1)

    def test_debug_logprob_tag_is_opt_in(self):
        metrics = _metrics()
        metrics["vllm_logprob_sanity"] = {
            "enabled": True,
            "mean_abs_error": 0.0125,
        }
        selected = production_tensorboard_metrics(metrics, "opd")
        self.assertEqual(selected["debug/vllm_hf_logprob_mae"], 0.0125)

    def test_tensorboard_writer_receives_expanded_metrics(self):
        logger = object.__new__(TensorBoardLogger)
        logger.writer = MagicMock()
        logger.interval = 1

        logger.write(7, _metrics(), "rac")

        written = {
            call.args[0]: (call.args[1], call.kwargs["global_step"])
            for call in logger.writer.add_scalar.call_args_list
        }
        self.assertEqual(written["train/global_step"][1], 7)
        self.assertIn("distillation/student_entropy", written)
        self.assertIn("optimization/ratio_mean", written)
        self.assertIn("rollout/response_length_max", written)
        self.assertIn("system/gpu_memory_reserved_gb", written)
        self.assertIn("rac/V_std", written)


if __name__ == "__main__":
    unittest.main()
