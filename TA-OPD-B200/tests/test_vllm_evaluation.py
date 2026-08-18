from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from b200_experiment.vllm_evaluation import evaluate_vllm_suite


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
            types.SimpleNamespace(outputs=[types.SimpleNamespace(text="1")])
            for _ in prompts
        ]


class VllmEvaluationTests(unittest.TestCase):
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
                "benchmark_names": list(benchmarks),
                "vllm": {"max_num_seqs": 64, "gpu_memory_utilization": 0.4},
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
            self.assertEqual(
                [suite["benchmarks"][name]["accuracy"] for name in benchmarks],
                [1.0, 1.0, 1.0],
            )


if __name__ == "__main__":
    unittest.main()
