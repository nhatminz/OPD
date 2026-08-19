from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch

from b200_experiment.selector_logging import SelectedTokenLogger


class FakeTokenizer:
    @staticmethod
    def convert_ids_to_tokens(ids):
        return [f"tok-{item}" for item in ids]


class SelectorLoggingTests(unittest.TestCase):
    def test_ta_selected_tokens_are_incrementally_gzipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True, False], [False, True]])
            diagnostics = {
                key: torch.arange(4, dtype=torch.float32).reshape(2, 2)
                for key in ("D", "C", "D_norm", "C_norm", "s_TA")
            }
            logger = SelectedTokenLogger(
                temporary, FakeTokenizer(), "ta", chunk_steps=2
            )
            count = logger.write(
                step=1,
                dataset_indices=[10, 11],
                sample_ids=["a", "b"],
                response_ids=torch.tensor([[5, 6], [7, 8]]),
                selected_mask=mask,
                diagnostics=diagnostics,
            )
            self.assertEqual(count, 2)
            path = next((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(rows[0]["token_text"], "tok-5")
            self.assertEqual(rows[1]["sample_id"], "b")
            self.assertIn("s_TA", rows[0])

    def test_distributed_ranks_use_independent_gzip_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True]])
            diagnostics = {
                key: torch.ones(1, 1) for key in ("D", "C", "D_norm", "C_norm", "s_TA")
            }
            for rank in (0, 1):
                logger = SelectedTokenLogger(
                    temporary,
                    FakeTokenizer(),
                    "ta",
                    rank=rank,
                    world_size=2,
                )
                logger.write(
                    step=1,
                    dataset_indices=[10 + rank],
                    sample_ids=[str(rank)],
                    response_ids=torch.tensor([[5 + rank]]),
                    selected_mask=mask,
                    diagnostics=diagnostics,
                    batch_index_offset=rank,
                )
            paths = sorted((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            self.assertEqual(len(paths), 2)
            self.assertIn("rank-00000", paths[0].name)
            self.assertIn("rank-00001", paths[1].name)


if __name__ == "__main__":
    unittest.main()
