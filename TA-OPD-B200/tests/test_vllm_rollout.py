from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from b200_experiment.vllm_rollout import (
    VLLMRolloutEngine,
    rollout_batch_from_token_ids,
)


def _config(max_model_len: int = 1024):
    return {
        "models": {
            "student_path": "/workspace/storage-shared/models/Qwen3-1.7B",
            "dtype": "bfloat16",
        },
        "data": {"max_prompt_tokens": 512},
        "rollout": {
            "batch_size": 64,
            "max_new_tokens": 256,
            "vllm": {
                "gpu_memory_utilization": 0.25,
                "max_num_seqs": 64,
                "max_model_len": max_model_len,
                "max_concurrent_requests": 64,
                "enable_prefix_caching": True,
            },
        },
    }


class VLLMRolloutTests(unittest.TestCase):
    def test_rollout_requests_log_probs_only_for_enabled_sanity_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = VLLMRolloutEngine(_config(), Path(temporary))
            response = MagicMock()
            response.json.return_value = {
                "choices": [{"token_ids": [31], "logprobs": None}]
            }
            with patch(
                "b200_experiment.vllm_rollout.requests.post", return_value=response
            ) as post:
                engine._generate_one(
                    0,
                    [11, 12],
                    max_new_tokens=2,
                    temperature=1.0,
                    top_p=1.0,
                    eos_token_ids=[2],
                    seed=42,
                    return_log_probs=False,
                )
                without_sanity = post.call_args.kwargs["json"]
                engine._generate_one(
                    0,
                    [11, 12],
                    max_new_tokens=2,
                    temperature=1.0,
                    top_p=1.0,
                    eos_token_ids=[2],
                    seed=42,
                    return_log_probs=True,
                )
                with_sanity = post.call_args.kwargs["json"]
            engine.close()
        self.assertNotIn("logprobs", without_sanity)
        self.assertEqual(with_sanity["logprobs"], 1)

    def test_each_torchrun_worker_isolates_its_vllm_child_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = VLLMRolloutEngine(
                _config(), Path(temporary), local_rank=1, world_size=3
            )
            with patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": "2,4,6",
                    "RANK": "1",
                    "LOCAL_RANK": "1",
                    "WORLD_SIZE": "3",
                    "MASTER_PORT": "12345",
                },
                clear=False,
            ):
                environment = engine._server_environment()
            engine.close()
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "4")
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT"):
            self.assertNotIn(name, environment)

    def test_token_ids_preserve_left_padded_prompt_and_response_mask(self):
        prompt_ids = torch.tensor([[0, 11, 12], [21, 22, 23]])
        prompt_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        rollout = rollout_batch_from_token_ids(
            prompt_ids,
            prompt_mask,
            [[31, 32], [41]],
            pad_token_id=0,
        )
        self.assertEqual(rollout.prompt_width, 3)
        self.assertTrue(
            torch.equal(
                rollout.input_ids,
                torch.tensor([[0, 11, 12, 31, 32], [21, 22, 23, 41, 0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                rollout.attention_mask,
                torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                rollout.valid_mask,
                torch.tensor([[True, True], [True, False]]),
            )
        )
        self.assertEqual(rollout.rollout_log_probs.dtype, torch.float32)
        self.assertTrue(torch.isnan(rollout.rollout_log_probs).all())

    def test_server_log_probs_are_aligned_for_optional_sync_sanity(self):
        rollout = rollout_batch_from_token_ids(
            torch.tensor([[11, 12]]),
            torch.ones(1, 2, dtype=torch.long),
            [[31, 32]],
            pad_token_id=0,
            response_log_probs=[[-0.1, -0.2]],
        )
        self.assertTrue(
            torch.allclose(rollout.rollout_log_probs, torch.tensor([[-0.1, -0.2]]))
        )

    def test_server_command_enables_ipc_dummy_load_and_sleep(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = VLLMRolloutEngine(_config(), Path(temporary))
            with patch(
                "b200_experiment.vllm_rollout.shutil.which",
                return_value="/venv/bin/vllm",
            ):
                command = engine._server_command()
            engine.close()
        joined = " ".join(command)
        self.assertEqual(command[:3], ["/venv/bin/vllm", "serve", command[2]])
        self.assertIn("--weight-transfer-config", command)
        self.assertIn('{"backend": "ipc"}', command)
        self.assertIn("--load-format dummy", joined)
        self.assertIn("--enable-sleep-mode", command)
        self.assertIn("--enforce-eager", command)
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--enable-chunked-prefill", command)
        self.assertIn("--async-scheduling", command)
        self.assertIn("--performance-mode throughput", joined)
        self.assertIn("--gpu-memory-utilization 0.25", joined)

    def test_server_rejects_context_smaller_than_prompt_plus_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = VLLMRolloutEngine(_config(max_model_len=767), Path(temporary))
            with patch(
                "b200_experiment.vllm_rollout.shutil.which",
                return_value="/venv/bin/vllm",
            ):
                with self.assertRaisesRegex(ValueError, "prompt\\+response"):
                    engine._server_command()
            engine.close()

    def test_colocated_memory_utilization_must_be_numeric(self):
        config = _config()
        config["rollout"]["vllm"]["gpu_memory_utilization"] = "auto"
        with tempfile.TemporaryDirectory() as temporary:
            engine = VLLMRolloutEngine(config, Path(temporary))
            with patch(
                "b200_experiment.vllm_rollout.shutil.which",
                return_value="/venv/bin/vllm",
            ):
                with self.assertRaisesRegex(ValueError, "must be a number"):
                    engine._server_command()
            engine.close()


if __name__ == "__main__":
    unittest.main()
