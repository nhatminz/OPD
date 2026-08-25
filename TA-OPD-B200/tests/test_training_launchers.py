from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = (
    "train_opd_b200.sh",
    "train_ta_b200.sh",
    "train_rac_b200.sh",
)


class TrainingLauncherTests(unittest.TestCase):
    def test_training_launchers_do_not_require_preflight(self):
        for name in TRAIN_SCRIPTS:
            content = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertIn("USER CONFIG", content)
                self.assertNotIn("require_b200_validation", content)
                for variable in (
                    "STUDENT_MODEL",
                    "TEACHER_MODEL",
                    "TRAIN_DATA",
                    "PROMPT_KEY",
                    "GLOBAL_BATCH_SIZE",
                    "MICRO_BATCH_SIZE_PER_GPU",
                    "NUM_RESPONSES",
                    "NUM_EPOCHS",
                    "MAX_STEPS",
                    "LR",
                    "MAX_PROMPT_LEN",
                    "MAX_RESPONSE_LEN",
                    "TOP_K",
                    "SAVE_INTERVAL",
                    "EVAL_INTERVAL",
                ):
                    self.assertIn(f"export {variable}=", content)

    def test_common_config_accepts_new_model_and_data_aliases(self):
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHON_BIN": sys.executable,
                "STORAGE_ROOT": "/storage",
                "STUDENT_MODEL": "/models/student",
                "TEACHER_MODEL": "/models/teacher",
                "TRAIN_DATA": "/datasets/train.parquet",
                "PROMPT_KEY": "question_text",
                "CUDA_VISIBLE_DEVICES": "0",
                "TMPDIR": str(REPO_ROOT.parent),
            }
        )
        for legacy in (
            "STUDENT_MODEL_PATH",
            "TEACHER_MODEL_PATH",
            "TRAIN_DATA_PATH",
            "TRAIN_PROMPT_KEY",
        ):
            environment.pop(legacy, None)
        completed = subprocess.run(
            [
                "bash",
                "-c",
                "source scripts/common_b200.sh; "
                "printf '%s|%s|%s|%s' \"$STUDENT_MODEL_PATH\" "
                "\"$TEACHER_MODEL_PATH\" \"$TRAIN_DATA_PATH\" "
                "\"$TRAIN_PROMPT_KEY\"",
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.stdout,
            "/models/student|/models/teacher|/datasets/train.parquet|question_text",
        )


if __name__ == "__main__":
    unittest.main()
