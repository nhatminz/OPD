from __future__ import annotations

import unittest

from b200_experiment.tensorboard_logging import (
    BASE_TAGS,
    production_tensorboard_metrics,
)


def _metrics() -> dict:
    values = {field: float(index + 1) for index, field in enumerate(BASE_TAGS.values())}
    values.update(
        selector={
            "selected_fraction": 0.125,
            "valid_tokens": 20,
            "effective_sample_size": 15.0,
        },
        vllm_logprob_sanity={"enabled": False},
        deliberately_unlogged_metric=999.0,
    )
    return values


class TensorBoardMetricTests(unittest.TestCase):
    def test_opd_writes_only_the_requested_base_tags(self):
        self.assertEqual(
            set(production_tensorboard_metrics(_metrics(), "opd")),
            set(BASE_TAGS),
        )

    def test_ta_adds_only_selected_token_fraction(self):
        selected = production_tensorboard_metrics(_metrics(), "ta")
        self.assertEqual(set(selected), set(BASE_TAGS) | {"ta/selected_token_fraction"})
        self.assertEqual(selected["ta/selected_token_fraction"], 0.125)

    def test_rac_effective_fraction_is_normalized(self):
        selected = production_tensorboard_metrics(_metrics(), "rac")
        self.assertEqual(
            set(selected), set(BASE_TAGS) | {"rac/effective_token_fraction"}
        )
        self.assertEqual(selected["rac/effective_token_fraction"], 0.75)

    def test_debug_logprob_tag_is_opt_in(self):
        metrics = _metrics()
        metrics["vllm_logprob_sanity"] = {
            "enabled": True,
            "mean_abs_error": 0.0125,
        }
        selected = production_tensorboard_metrics(metrics, "opd")
        self.assertEqual(selected["debug/vllm_hf_logprob_mae"], 0.0125)


if __name__ == "__main__":
    unittest.main()
