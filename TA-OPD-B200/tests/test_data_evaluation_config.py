from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from b200_experiment.config import load_config, resolve_runtime_paths
import torch

from b200_experiment.data import (
    epoch_batch_indices,
    expand_prompt_batch,
    read_records,
    record_messages,
    render_record_prompt,
    stable_sample_id,
    validate_prompt_records,
)
from b200_experiment.evaluation import (
    BENCHMARK_ORDER,
    aggregate_evaluations,
    load_benchmark,
)
from b200_experiment.math_prompts import (
    MATH_USER_INSTRUCTION,
    build_math_user_prompt,
    render_math_prompt,
)


class _TemplateTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return (
            f"thinking={kwargs.get('enable_thinking')}|"
            f"generation={kwargs['add_generation_prompt']}|"
            f"{messages[0]['role']}:{messages[0]['content']}"
        )


class DataEvaluationConfigTests(unittest.TestCase):
    def test_one_response_keeps_one_independent_trajectory_per_prompt(self):
        encoded = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
        }
        expanded, indices, response_indices = expand_prompt_batch(encoded, [7, 9], 1)
        self.assertTrue(torch.equal(expanded["input_ids"], encoded["input_ids"]))
        self.assertEqual(indices, [7, 9])
        self.assertEqual(response_indices, [0, 0])

    def test_one_prompt_expands_to_four_independent_trajectory_entries(self):
        encoded = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        expanded, indices, response_indices = expand_prompt_batch(encoded, [17], 4)
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

    def test_competition_math_train_schema_uses_problem_and_unique_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train-00000-of-00001.parquet"
            pd.DataFrame(
                [
                    {
                        "problem": "Find $1+1$.",
                        "solution": "It is two.",
                        "answer": "2",
                        "subject": "Prealgebra",
                        "level": 1,
                        "unique_id": "train/prealgebra/1",
                    }
                ]
            ).to_parquet(path)

            records, files = read_records(path, split=None)

            self.assertEqual(files, [path.resolve()])
            self.assertEqual(
                record_messages(records[0], prompt_key="problem"),
                [
                    {
                        "role": "user",
                        "content": (
                            "Find $1+1$. "
                            "Let's think step by step and output the final answer "
                            "within \\boxed{}."
                        ),
                    }
                ],
            )
            self.assertEqual(stable_sample_id(records[0], 0), "train/prealgebra/1")

    def test_train_and_eval_share_one_idempotent_canonical_math_prompt(self):
        problem = "Find $3+4$."
        record = {"problem": problem}
        data_config = {
            "prompt_key": "problem",
            "prefer_source_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        tokenizer = _TemplateTokenizer()

        train_user_prompt = record_messages(record, prompt_key="problem")[0][
            "content"
        ]
        eval_user_prompt = build_math_user_prompt(problem)
        rendered_train_prompt = render_record_prompt(record, tokenizer, data_config)
        rendered_eval_prompt = render_math_prompt(tokenizer, problem, data_config)

        self.assertEqual(train_user_prompt, eval_user_prompt)
        self.assertEqual(rendered_train_prompt, rendered_eval_prompt)
        self.assertEqual(train_user_prompt.count(MATH_USER_INSTRUCTION), 1)
        self.assertEqual(
            build_math_user_prompt(train_user_prompt), train_user_prompt
        )

    def test_preformatted_dataset_prompt_does_not_duplicate_instruction(self):
        formatted = build_math_user_prompt("What is $5+5$?")
        messages = record_messages(
            {"prompt": [{"role": "user", "content": formatted}]},
            prompt_key="prompt",
        )

        self.assertEqual(messages[0]["content"], formatted)
        self.assertEqual(messages[0]["content"].count(MATH_USER_INSTRUCTION), 1)

    def test_missing_prompt_field_reports_available_columns(self):
        with self.assertRaisesRegex(
            KeyError, "prompt.*available fields: answer, problem"
        ):
            record_messages(
                {"problem": "1+1?", "answer": "2"}, prompt_key="prompt"
            )

    def test_dataset_prompt_schema_validation_reports_row_before_training(self):
        with self.assertRaisesRegex(
            KeyError, "Training row 1:.*available fields: answer"
        ):
            validate_prompt_records(
                [{"problem": "1+1?"}, {"answer": "2"}],
                {"prompt_key": "problem", "prefer_source_prompt": False},
            )

    def test_competition_math_eval_uses_answer_not_worked_solution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test-00000-of-00001.parquet"
            pd.DataFrame(
                [
                    {
                        "problem": "Find $2+2$.",
                        "solution": "Adding gives four.",
                        "answer": "4",
                        "subject": "Prealgebra",
                        "level": 1,
                        "unique_id": "test/prealgebra/1",
                    }
                ]
            ).to_parquet(path)

            rows, schema = load_benchmark(
                "Competition-MATH",
                {
                    "path": str(path),
                    "question_key": "problem",
                    "answer_key": "answer",
                    "id_key": "unique_id",
                },
            )

            self.assertEqual(rows[0], {"id": "test/prealgebra/1", "problem": "Find $2+2$.", "answer": "4"})
            self.assertEqual(schema["answer_key"], "answer")

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
            base["models"]["teacher_path"], "/workspace/storage-shared/models/Qwen3-8B"
        )
        self.assertEqual(
            base["models"]["student_path"],
            "/workspace/storage-shared/nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base",
        )
        self.assertEqual(
            base["data"]["path"],
            "/workspace/storage-shared/nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet",
        )
        self.assertIsNone(base["data"]["split"])
        self.assertEqual(base["data"]["prompt_key"], "problem")
        self.assertFalse(base["data"]["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(base["models"]["teacher_no_think"])
        self.assertEqual(
            base["evaluation"]["benchmarks"]["Competition-MATH"]["path"],
            "/workspace/storage-shared/nlp/minhpn19/data/competition_math/data/test-00000-of-00001.parquet",
        )
        self.assertEqual(
            tuple(base["training_evaluation"]["benchmark_names"]), BENCHMARK_ORDER
        )
        self.assertEqual(
            base["evaluation"]["benchmarks"]["Competition-MATH"]["answer_key"],
            "answer",
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
        self.assertEqual(base["distributed"]["strategy"], "fsdp")
        self.assertEqual(base["distributed"]["fsdp"]["sharding_strategy"], "FULL_SHARD")
        self.assertTrue(base["distributed"]["fsdp"]["use_orig_params"])
        self.assertFalse(base["distributed"]["fsdp"]["teacher_cpu_offload"])
        self.assertEqual(base["rollout"]["batch_size"], 64)
        self.assertEqual(base["rollout"]["num_responses"], 1)
        self.assertEqual(base["rollout"]["temperature"], 1.0)
        self.assertEqual(base["training"]["micro_batch_size_per_gpu"], 8)
        self.assertTrue(base["training"]["gradient_checkpointing"])
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
            "distributed",
            "logging",
        ):
            self.assertEqual(opd[section], ta[section], section)
            self.assertEqual(ta[section], rac[section], section)


if __name__ == "__main__":
    unittest.main()
