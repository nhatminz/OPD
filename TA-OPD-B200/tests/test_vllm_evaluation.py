from __future__ import annotations

import gzip
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from b200_experiment.vllm_evaluation import (
    _resolve_gpu_memory_utilization,
    evaluate_vllm_suite,
)


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _LLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.generate_calls = []
        self.__class__.instances.append(self)

    def generate(self, prompts, sampling, use_tqdm):
        self.generate_calls.append((prompts, sampling, use_tqdm))
        return [
            types.SimpleNamespace(
                outputs=[
                    types.SimpleNamespace(text="1")
                    for _ in range(sampling.kwargs.get("n", 1))
                ]
            )
            for _ in prompts
        ]


class VllmEvaluationTests(unittest.TestCase):
    def test_one_response_reports_accuracy_without_avg16_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark_path = root / "math.jsonl"
            benchmark_path.write_text(
                json.dumps({"problem": "1?", "answer": "1"}) + "\n",
                encoding="utf-8",
            )
            config = {
                "models": {"dtype": "bfloat16"},
                "data": {"chat_template_kwargs": {}},
                "evaluation": {
                    "benchmarks": {"MATH-500": {"path": str(benchmark_path)}}
                },
            }
            settings = {
                "max_new_tokens": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "num_responses": 1,
                "benchmark_names": ["MATH-500"],
                "vllm": {"gpu_memory_utilization": 0.4},
            }
            fake_vllm = types.SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
            _LLM.instances.clear()
            with (
                patch.dict(sys.modules, {"vllm": fake_vllm}),
                patch(
                    "b200_experiment.vllm_evaluation.AutoTokenizer.from_pretrained",
                    return_value=_Tokenizer(),
                ),
            ):
                suite = evaluate_vllm_suite(
                    "student", root, config, root / "results", settings
                )

            result = suite["benchmarks"]["MATH-500"]
            self.assertEqual(suite["parameters"]["metric"], "accuracy")
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["avg_at_n"], 1.0)
            self.assertNotIn("avg_at_16", result)
            detailed_path = Path(suite["detailed_outputs"])
            self.assertEqual(detailed_path.name, "model_outputs_detailed.jsonl.gz")
            with gzip.open(detailed_path, "rt", encoding="utf-8") as handle:
                detailed_rows = [json.loads(line) for line in handle]
            self.assertEqual(len(detailed_rows), 1)
            detailed = detailed_rows[0]
            self.assertEqual(detailed["model_name"], "student")
            self.assertEqual(detailed["backend"], "vllm")
            self.assertEqual(detailed["benchmark"], "MATH-500")
            self.assertEqual(
                detailed["rendered_prompt"],
                "1? Let's think step by step and output the final answer within \\boxed{}.",
            )
            self.assertEqual(detailed["samples_per_problem"], 1)
            self.assertEqual(detailed["outputs"][0]["response"], "1")
            self.assertTrue(detailed["outputs"][0]["correct"])

    def test_auto_memory_uses_all_free_vram_except_headroom(self):
        gib = 2**30
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(160 * gib, 180 * gib)),
        ):
            utilization = _resolve_gpu_memory_utilization(
                {"gpu_memory_utilization": "auto", "gpu_headroom_gib": 4}
            )
        self.assertAlmostEqual(utilization, 156 / 180)

    def test_all_benchmarks_share_one_generate_progress_bar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmarks = {}
            for name in ("MATH-500", "AIME24", "AIME25"):
                path = root / f"{name}.jsonl"
                path.write_text(
                    json.dumps({"problem": f"{name}: 1?", "answer": "1"}) + "\n",
                    encoding="utf-8",
                )
                benchmarks[name] = {"path": str(path)}
            config = {
                "models": {"dtype": "bfloat16"},
                "data": {"chat_template_kwargs": {"enable_thinking": True}},
                "evaluation": {"benchmarks": benchmarks},
            }
            settings = {
                "max_new_tokens": 32,
                "temperature": 1.0,
                "top_p": 0.95,
                "num_responses": 16,
                "benchmark_names": list(benchmarks),
                "vllm": {
                    "max_num_seqs": 64,
                    "max_num_batched_tokens": 4096,
                    "gpu_memory_utilization": 0.4,
                    "enable_chunked_prefill": True,
                    "async_scheduling": True,
                    "performance_mode": "throughput",
                },
            }
            fake_vllm = types.SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
            _LLM.instances.clear()
            with (
                patch.dict(sys.modules, {"vllm": fake_vllm}),
                patch(
                    "b200_experiment.vllm_evaluation.AutoTokenizer.from_pretrained",
                    return_value=_Tokenizer(),
                ),
            ):
                suite = evaluate_vllm_suite(
                    "student", root, config, root / "results", settings
                )

            self.assertEqual(len(_LLM.instances), 1)
            calls = _LLM.instances[0].generate_calls
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0][0]), 3)
            self.assertTrue(calls[0][2])
            self.assertEqual(_LLM.instances[0].kwargs["max_num_seqs"], 64)
            self.assertEqual(_LLM.instances[0].kwargs["max_num_batched_tokens"], 4096)
            self.assertTrue(_LLM.instances[0].kwargs["enable_chunked_prefill"])
            self.assertTrue(_LLM.instances[0].kwargs["async_scheduling"])
            self.assertEqual(_LLM.instances[0].kwargs["performance_mode"], "throughput")
            self.assertEqual(calls[0][1].kwargs["temperature"], 1.0)
            self.assertEqual(calls[0][1].kwargs["top_p"], 0.95)
            self.assertEqual(calls[0][1].kwargs["n"], 16)
            self.assertTrue(suite["parameters"]["do_sample"])
            self.assertEqual(
                [suite["benchmarks"][name]["accuracy"] for name in benchmarks],
                [1.0, 1.0, 1.0],
            )
            self.assertEqual(suite["benchmarks"]["MATH-500"]["total"], 16)
            self.assertEqual(suite["benchmarks"]["MATH-500"]["problems"], 1)


if __name__ == "__main__":
    unittest.main()
