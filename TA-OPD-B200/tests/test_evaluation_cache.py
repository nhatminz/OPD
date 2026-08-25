from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from b200_experiment.evaluation_cache import (
    BASE_EVALUATION_CACHE_SCHEMA,
    base_evaluation_cache_key,
    evaluate_or_reuse_base,
)
from b200_experiment.math_prompts import EVAL_PROMPT_PROTOCOL_VERSION


class BaseEvaluationCacheTests(unittest.TestCase):
    def _fingerprint_fixture(self, root: Path):
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "tokenizer.json").write_text('{"vocab":{}}\n', encoding="utf-8")
        benchmark = root / "math.jsonl"
        benchmark.write_text(
            json.dumps({"problem": "1+1?", "answer": "2"}) + "\n",
            encoding="utf-8",
        )
        config = {
            "models": {"dtype": "bfloat16"},
            "data": {"chat_template_kwargs": {"enable_thinking": False}},
            "evaluation": {"benchmarks": {"MATH-500": {"path": str(benchmark)}}},
        }
        settings = {
            "backend": "vllm",
            "temperature": 1.0,
            "top_p": 0.95,
            "num_responses": 16,
            "max_new_tokens": 32,
            "limit": None,
            "benchmark_names": ["MATH-500"],
            "vllm": {"performance_mode": "throughput"},
        }
        return model, config, settings

    def test_cache_fingerprint_tracks_prompt_protocol_and_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, config, settings = self._fingerprint_fixture(root)
            first = base_evaluation_cache_key(config, settings, model)
            unchanged = base_evaluation_cache_key(config, settings, model)
            changed_settings = dict(settings, temperature=0.5)
            sampling_changed = base_evaluation_cache_key(
                config, changed_settings, model
            )
            prompt_copy = root / "math_prompts_modified.py"
            prompt_copy.write_text("DIFFERENT PROMPT PROTOCOL\n", encoding="utf-8")
            prompt_changed = base_evaluation_cache_key(
                config, settings, model, math_prompts_path=prompt_copy
            )

            self.assertEqual(first, unchanged)
            self.assertNotEqual(first, sampling_changed)
            self.assertNotEqual(first, prompt_changed)
            self.assertEqual(BASE_EVALUATION_CACHE_SCHEMA, 2)
            self.assertEqual(EVAL_PROMPT_PROTOCOL_VERSION, "verl_eopd_math_v1")

    def test_identical_base_eval_is_generated_once_and_paths_are_rewritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            benchmarks = {}
            for name in ("MATH-500", "AIME24", "AIME25"):
                path = root / f"{name}.jsonl"
                path.write_text(
                    json.dumps({"problem": "1+1?", "answer": "2"}) + "\n",
                    encoding="utf-8",
                )
                benchmarks[name] = {"path": str(path)}
            config = {
                "models": {"dtype": "bfloat16"},
                "data": {"chat_template_kwargs": {"enable_thinking": True}},
                "evaluation": {"benchmarks": benchmarks},
                "training_evaluation": {},
            }
            settings = {
                "backend": "vllm",
                "temperature": 1.0,
                "top_p": 0.95,
                "num_responses": 16,
                "max_new_tokens": 32,
                "limit": None,
                "benchmark_names": list(benchmarks),
                "vllm": {"performance_mode": "throughput"},
            }
            cache = root / "cache"
            calls = []

            def evaluator(destination: Path):
                def run():
                    calls.append(destination)
                    destination.mkdir(parents=True, exist_ok=True)
                    suite = {
                        "model": "Base student",
                        "model_path": str(model),
                        "parameters": {
                            "backend": "vllm",
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "num_responses": 16,
                            "max_new_tokens": 32,
                            "limit": None,
                        },
                        "benchmarks": {},
                    }
                    for name in benchmarks:
                        prediction = destination / f"{name}.jsonl.gz"
                        with gzip.open(prediction, "wt", encoding="utf-8") as handle:
                            handle.write("{}\n")
                        suite["benchmarks"][name] = {
                            "correct": 1,
                            "total": 16,
                            "accuracy": 1 / 16,
                            "predictions": str(prediction),
                        }
                    detailed_outputs = destination / "model_outputs_detailed.jsonl.gz"
                    with gzip.open(detailed_outputs, "wt", encoding="utf-8") as handle:
                        handle.write('{"response":"2"}\n')
                    suite["detailed_outputs"] = str(detailed_outputs)
                    (destination / "summary.json").write_text(
                        json.dumps(suite), encoding="utf-8"
                    )
                    return suite

                return run

            first = root / "opd" / "step-000000"
            suite, status = evaluate_or_reuse_base(
                config=config,
                runtime_settings=settings,
                model_path=model,
                model_name="Base student",
                destination=first,
                evaluator=evaluator(first),
                cache_dir=cache,
            )
            self.assertEqual(status, "generated")
            self.assertEqual(len(calls), 1)
            self.assertEqual(Path(suite["model_path"]), model)

            second = root / "rac" / "step-000000"
            suite, status = evaluate_or_reuse_base(
                config=config,
                runtime_settings=settings,
                model_path=model,
                model_name="Base student",
                destination=second,
                evaluator=evaluator(second),
                cache_dir=cache,
            )
            self.assertEqual(status, "shared")
            self.assertEqual(len(calls), 1)
            for result in suite["benchmarks"].values():
                self.assertEqual(Path(result["predictions"]).parent, second)
                self.assertTrue(Path(result["predictions"]).is_file())
            self.assertEqual(Path(suite["detailed_outputs"]).parent, second)
            self.assertTrue(Path(suite["detailed_outputs"]).is_file())

            _suite, status = evaluate_or_reuse_base(
                config=config,
                runtime_settings=settings,
                model_path=model,
                model_name="Base student",
                destination=first,
                evaluator=evaluator(first),
                cache_dir=cache,
            )
            self.assertEqual(status, "local")
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
