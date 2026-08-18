from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from b200_experiment.config import load_config
from b200_experiment.data import epoch_batch_indices, read_records
from b200_experiment.evaluation import load_benchmark


class DataEvaluationConfigTests(unittest.TestCase):
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
        base = load_config(root / "configs/qwen3_b200_base.yaml")
        ta = load_config(root / "configs/qwen3_b200_ta.yaml")
        rac = load_config(root / "configs/qwen3_b200_rac.yaml")
        self.assertEqual(
            base["models"]["teacher_path"], "/workspace/storage-shared/models/Qwen3-4B"
        )
        self.assertEqual(
            base["models"]["student_path"],
            "/workspace/storage-shared/models/Qwen3-1.7B",
        )
        self.assertEqual(
            base["evaluation"]["benchmarks"]["AIME24"]["path"],
            "/workspace/storage-shared/nlp/minhpn19/data/eval/aime24/test-00000-of-00001.parquet",
        )
        self.assertEqual(
            base["evaluation"]["benchmarks"]["AIME25"]["path"],
            "/workspace/storage-shared/nlp/minhpn19/data/eval/aime25/test.jsonl",
        )
        self.assertEqual(base["training_evaluation"]["target_evaluations"], 16)
        self.assertEqual(base["training_evaluation"]["backend"], "vllm")
        self.assertIsNone(base["training_evaluation"]["limit"])
        self.assertEqual(base["evaluation"]["backend"], "vllm")
        self.assertIsNone(base["evaluation"]["limit"])
        self.assertEqual(base["rollout"]["backend"], "vllm")
        self.assertEqual(base["rollout"]["batch_size"], 64)
        self.assertEqual(base["training"]["micro_batch_size"], 64)
        for section in (
            "models",
            "data",
            "rollout",
            "selector",
            "token_budget",
            "training",
            "training_evaluation",
        ):
            self.assertEqual(ta[section], rac[section], section)


if __name__ == "__main__":
    unittest.main()
