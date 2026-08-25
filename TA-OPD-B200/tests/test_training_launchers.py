from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = (
    "train_opd_b200.sh",
    "train_ta_b200.sh",
    "train_rac_b200.sh",
)


class TrainingLauncherTests(unittest.TestCase):
    def test_opd_then_rac_workflow_has_exact_sequential_stages(self):
        path = REPO_ROOT / "scripts/train_opd_reeval_then_rac_reeval_b200.sh"
        content = path.read_text(encoding="utf-8")
        stages = (
            'bash "${SCRIPT_DIR}/train_opd_b200.sh"',
            'bash "${SCRIPT_DIR}/reeval_method_checkpoints_b200.sh" opd',
            'bash "${SCRIPT_DIR}/train_rac_b200.sh"',
            'bash "${SCRIPT_DIR}/reeval_method_checkpoints_b200.sh" rac',
        )
        positions = [content.index(stage) for stage in stages]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("export TRAIN_EVAL_ENABLED=false", content)
        self.assertIn('export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"', content)
        self.assertIn('export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"', content)
        self.assertIn('export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"', content)
        self.assertIn("models/Qwen3-8B", content)
        self.assertIn("Qwen3-1.7B-Base", content)
        self.assertIn("competition_math", content)

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
                    "PPO_MINI_BATCH_SIZE",
                    "MICRO_BATCH_SIZE_PER_GPU",
                    "NUM_RESPONSES",
                    "NUM_EPOCHS",
                    "MAX_STEPS",
                    "LR",
                    "MAX_PROMPT_LEN",
                    "OVERLONG_PROMPT_POLICY",
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

    def test_train_all_forwards_one_exact_shared_asset_selection(self):
        content = (REPO_ROOT / "scripts/train_all_b200.sh").read_text(
            encoding="utf-8"
        )
        self.assertLess(content.index("USER CONFIG"), content.index("source "))
        with tempfile.TemporaryDirectory(dir=REPO_ROOT.parent) as temporary:
            temporary_path = Path(temporary)
            capture = temporary_path / "children.txt"
            fake_bash = temporary_path / "bash"
            fake_bash.write_text(
                "#!/bin/sh\n"
                "printf '%s|%s|%s|%s|%s\\n' \"$1\" \"$STUDENT_MODEL\" "
                "\"$TEACHER_MODEL\" \"$TRAIN_DATA\" \"$PROMPT_KEY\" "
                ">> \"$CAPTURE_FILE\"\n",
                encoding="utf-8",
            )
            fake_bash.chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "PATH": f"{temporary}:{environment['PATH']}",
                    "CAPTURE_FILE": str(capture),
                    "PYTHON_BIN": sys.executable,
                    "CUDA_VISIBLE_DEVICES": "0",
                    "STUDENT_MODEL": "/new/student",
                    "TEACHER_MODEL": "/new/teacher",
                    "TRAIN_DATA": "/new/train.parquet",
                    "PROMPT_KEY": "question",
                    # Stale legacy values must not win over the USER CONFIG.
                    "STUDENT_MODEL_PATH": "/stale/student",
                    "TEACHER_MODEL_PATH": "/stale/teacher",
                    "TRAIN_DATA_PATH": "/stale/train.parquet",
                    "TRAIN_PROMPT_KEY": "stale_prompt",
                }
            )
            subprocess.run(
                ["/bin/bash", "scripts/train_all_b200.sh"],
                cwd=REPO_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            lines = capture.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(
            [Path(line.split("|", 1)[0]).name for line in lines],
            ["train_opd_b200.sh", "train_ta_b200.sh", "train_rac_b200.sh"],
        )
        for line in lines:
            self.assertEqual(
                line.split("|")[1:],
                [
                    "/new/student",
                    "/new/teacher",
                    "/new/train.parquet",
                    "question",
                ],
            )


if __name__ == "__main__":
    unittest.main()
