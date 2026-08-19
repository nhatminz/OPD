from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from b200_experiment.resume import (
    resolve_resume_checkpoint,
    restore_optimizer,
    validate_append_history,
    validate_resume_config,
)


def _controlled_config(method: str = "ta") -> dict:
    return {
        "experiment": {"method": method, "seed": 1234},
        "models": {"student_path": "/base", "teacher_path": "/teacher"},
        "data": {"path": "/data", "split": "all"},
        "rollout": {
            "backend": "vllm",
            "batch_size": 64,
            "max_new_tokens": 256,
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": 42,
        },
        "selector": {
            "top_k": 16,
            "branch_m": 2,
            "rac_delta_mode": "full_vocab",
        },
        "token_budget": {"rho": 0.10},
        "training": {
            "learning_rate": 1e-5,
            "adam_betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "ppo_clip_low": 0.2,
            "ppo_clip_high": 0.28,
        },
    }


class ResumeTests(unittest.TestCase):
    def test_legacy_single_gpu_optimizer_checkpoint_restores_at_step_100(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-000100"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            source_model = torch.nn.Linear(3, 2)
            source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-5)
            source_model(torch.ones(1, 3)).sum().backward()
            source_optimizer.step()
            torch.save(
                {"step": 100, "optimizer": source_optimizer.state_dict()},
                checkpoint / "optimizer.pt",
            )

            target_model = torch.nn.Linear(3, 2)
            target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=9e-4)
            resolved = resolve_resume_checkpoint(checkpoint)
            state = restore_optimizer(target_optimizer, resolved, torch.device("cpu"))
            self.assertEqual(state.step, 100)
            self.assertEqual(target_optimizer.param_groups[0]["lr"], 1e-5)
            self.assertTrue(target_optimizer.state)

    def test_resume_refuses_logs_ahead_of_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metrics.jsonl").write_text(
                json.dumps({"step": 101}) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "step 101"):
                validate_append_history(root, 100)

    def test_resume_accepts_history_and_selector_logs_through_step_100(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metrics.jsonl").write_text(
                json.dumps({"step": 99}) + "\n" + json.dumps({"step": 100}) + "\n",
                encoding="utf-8",
            )
            (root / "eval_history.jsonl").write_text(
                json.dumps({"step": 0}) + "\n" + json.dumps({"step": 100}) + "\n",
                encoding="utf-8",
            )
            selector_root = root / "selector_scores"
            selector_root.mkdir()
            with gzip.open(
                selector_root / "selected_steps_000051_000100.jsonl.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps({"training_step": 100}) + "\n")
            result = validate_append_history(root, 100)
            self.assertEqual(result["metrics_last_step"], 100)
            self.assertEqual(result["evaluation_last_step"], 100)

    def test_resume_config_prevents_switching_ta_checkpoint_to_rac(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = output / "checkpoint-000100"
            checkpoint.mkdir()
            (output / "resolved_config.yaml").write_text(
                yaml.safe_dump(_controlled_config("ta")), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "experiment.method"):
                validate_resume_config(checkpoint, _controlled_config("rac"))


if __name__ == "__main__":
    unittest.main()
