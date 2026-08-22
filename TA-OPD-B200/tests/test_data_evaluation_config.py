from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from b200_experiment.config import load_config, resolve_runtime_paths
import torch

from b200_experiment.data import epoch_batch_indices, expand_prompt_batch, read_records
from b200_experiment.evaluation import aggregate_evaluations, load_benchmark


class DataEvaluationConfigTests(unittest.TestCase):
    def test_one_prompt_expands_to_four_independent_trajectory_entries(self):
        encoded = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        expanded, indices, response_indices = expand_prompt_batch(
            encoded, [17], 4
        )
        self.assertEqual(expanded["input_ids"].shape[0], 4)
        self.assertEqual(indices, [17, 17, 17, 17])
        self.assertEqual(response_indices, [0, 1, 2, 3])

    def test_aggregation_includes_pure_opd_in_controlled_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dirs = {}
            for index, method in enumerate(("Base", "OPD", "TA-OPD", "RAC")):
                output = root / method.lower().replace("-", "_")
                output.mkdir()
                (output / "summary.json").write_text(
                    json.dumps(
                        {
                            "benchmarks": {
                                benchmark: {"accuracy": 0.1 * (index + 1)}
                                for benchmark in ("MATH-500", "AIME24", "AIME25")
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                model_dirs[method] = output

            rows = aggregate_evaluations(model_dirs, root / "results")

            self.assertEqual([row["Method"] for row in rows], list(model_dirs))
            self.assertEqual(rows[1]["MATH-500"], 0.2)
    def test_epoch_batches_cover_full_dataset_without_repetition(self):
        batches = [epoch_batch_indices(10, 4, step, 1234) for step in range(3)]
        flattened = [item for batch in batches for item in batch]
        self.assertEqual(len(flattened), 10)
        self.assertEqual(sorted(flattened), list(range(10)))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])

    def test_directory_loader_reads_all_parquet_parts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "all").mkdir()
            (root / "en").mkdir()
            pd.DataFrame([{"prompt": "a"}, {"prompt": "b"}]).to_parquet(
                root / "all/part-0.parquet"
            )
            pd.DataFrame([{"prompt": "c"}]).to_parquet(root / "all/part-1.parquet")
            pd.DataFrame([{"prompt": "duplicate"}]).to_parquet(
                root / "en/part-0.parquet"
            )
            records, files = read_records(root, split="all")
            self.assertEqual(len(records), 3)
            self.assertEqual(len(files), 2)

    def test_b200_aime24_and_aime25_schema_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aime24 = root / "test-00000-of-00001.parquet"
            pd.DataFrame(
                [{"problem": "1+1?", "answer": 2, "solution": "two"}]
            ).to_parquet(aime24)
            aime25 = root / "test.jsonl"
            aime25.write_text(
                json.dumps({"question": "2+2?", "ground_truth": "4"}) + "\n",
                encoding="utf-8",
            )
            rows24, schema24 = load_benchmark("AIME24", {"path": str(aime24)})
            rows25, schema25 = load_benchmark("AIME25", {"path": str(aime25)})
            self.assertEqual(rows24[0]["answer"], "2")
            self.assertEqual(rows25[0]["problem"], "2+2?")
            self.assertEqual(schema24["answer_key"], "answer")
            self.assertEqual(schema25["answer_key"], "ground_truth")

    def test_configs_have_exact_paths_and_fair_shared_settings(self):
        root = Path(__file__).resolve().parents[1]
        base = resolve_runtime_paths(load_config(root / "configs/qwen3_b200_base.yaml"))
        opd = resolve_runtime_paths(load_config(root / "configs/qwen3_b200_opd.yaml"))
        ta = resolve_runtime_paths(load_config(root / "configs/qwen3_b200_ta.yaml"))
        rac = resolve_runtime_paths(load_config(root / "configs/qwen3_b200_rac.yaml"))
        self.assertEqual(
            base["models"]["teacher_path"], "/workspace/storage-shared/models/Qwen3-4B"
        )
        self.assertEqual(
            base["models"]["student_path"],
            "/workspace/storage-shared/models/Qwen3-1.7B",
        )
        self.assertEqual(
            base["evaluation"]["benchmarks"]["AIME24"]["path"],
            "/workspace/storage-shared/nlp/minhpn19/data/eval/aime24",
        )
        self.assertEqual(
            base["evaluation"]["benchmarks"]["AIME25"]["path"],
            "/workspace/storage-shared/nlp/minhpn19/data/eval/aime25",
        )
        self.assertIsNone(base["training_evaluation"]["target_evaluations"])
        self.assertEqual(base["training_evaluation"]["interval_steps"], 50)
        self.assertEqual(base["training_evaluation"]["backend"], "vllm")
        self.assertEqual(base["training_evaluation"]["temperature"], 0.7)
        self.assertEqual(base["training_evaluation"]["top_p"], 0.95)
        self.assertEqual(base["training_evaluation"]["num_responses"], 16)
        self.assertIsNone(base["training_evaluation"]["limit"])
        self.assertEqual(base["evaluation"]["backend"], "vllm")
        self.assertEqual(base["evaluation"]["temperature"], 0.7)
        self.assertEqual(base["evaluation"]["top_p"], 0.95)
        self.assertEqual(base["evaluation"]["num_responses"], 16)
        self.assertIsNone(base["evaluation"]["limit"])
        self.assertEqual(base["rollout"]["backend"], "vllm")
        self.assertEqual(base["rollout"]["batch_size"], 16)
        self.assertEqual(base["rollout"]["num_responses"], 4)
        self.assertEqual(base["rollout"]["temperature"], 1.0)
        self.assertEqual(base["training"]["micro_batch_size"], 1)
        self.assertEqual(base["data"]["max_prompt_tokens"], 1024)
        self.assertEqual(base["rollout"]["max_new_tokens"], 7168)
        self.assertEqual(base["rollout"]["vllm"]["max_model_len"], 9216)
        self.assertEqual(base["selector"]["top_k"], 16)
        self.assertEqual(base["opd"]["top_k_strategy"], "only_stu")
        self.assertEqual(base["opd"]["reward_weight_mode"], "student_p")
        self.assertEqual(base["opd"]["adv_estimator"], "token_reward_direct")
        self.assertEqual(base["opd"]["loss_agg_mode"], "token-mean")
        self.assertEqual(base["selector"]["rac_gamma"], 0.995)
        self.assertEqual(base["selector"]["rac_w_min"], 0.10)
        self.assertEqual(base["selector"]["rac_beta"], 2.0)
        self.assertEqual(opd["experiment"]["method"], "opd")
        self.assertEqual(ta["experiment"]["method"], "ta")
        self.assertEqual(rac["experiment"]["method"], "rac")
        for section in (
            "models",
            "paths",
            "data",
            "rollout",
            "opd",
            "selector",
            "token_budget",
            "training",
            "training_evaluation",
        ):
            self.assertEqual(opd[section], ta[section], section)
            self.assertEqual(ta[section], rac[section], section)


if __name__ == "__main__":
    unittest.main()
