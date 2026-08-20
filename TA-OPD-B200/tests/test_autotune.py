from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from b200_experiment.autotune import _update_readme


class AutotuneTests(unittest.TestCase):
    def test_measured_batch_is_written_between_readme_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            readme = Path(temporary) / "README.md"
            readme.write_text(
                "before\n<!-- B200_AUTOTUNE_RESULT_START -->\nold\n"
                "<!-- B200_AUTOTUNE_RESULT_END -->\nafter\n",
                encoding="utf-8",
            )
            _update_readme(readme, 64, 0.81, "NVIDIA B200")
            content = readme.read_text(encoding="utf-8")
            self.assertIn("global batch/micro-batch **64/1**", content)
            self.assertIn("`0.810`", content)
            self.assertNotIn("old", content)


if __name__ == "__main__":
    unittest.main()
