from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import torch


class SelectedTokenLogger:
    """Incremental, crash-safe gzip JSONL logger containing selected positions only."""

    def __init__(
        self,
        output_dir: str | Path,
        tokenizer,
        method: str,
        chunk_steps: int = 50,
        enabled: bool = True,
    ):
        self.root = Path(output_dir) / "selector_scores"
        self.tokenizer = tokenizer
        self.method = method
        self.chunk_steps = max(1, int(chunk_steps))
        self.enabled = bool(enabled)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "gzip JSONL; concatenated gzip members are valid",
                "scope": "selected/accepted response positions only",
                "method": method,
                "chunk_steps": self.chunk_steps,
                "common_fields": [
                    "training_step",
                    "sample_id",
                    "dataset_index",
                    "batch_index",
                    "response_position",
                    "token_id",
                    "token_text",
                ],
                "score_fields": ["D", "C", "D_norm", "C_norm", "s_TA"]
                if method == "ta"
                else ["Delta", "A", "F", "B", "s_RAC"],
            }
            with (self.root / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
                handle.write("\n")

    def _path(self, step: int) -> Path:
        first = ((step - 1) // self.chunk_steps) * self.chunk_steps + 1
        last = first + self.chunk_steps - 1
        return self.root / f"selected_steps_{first:06d}_{last:06d}.jsonl.gz"

    def write(
        self,
        *,
        step: int,
        dataset_indices: list[int],
        sample_ids: list[str],
        response_ids: torch.Tensor,
        selected_mask: torch.Tensor,
        diagnostics: dict[str, Any],
    ) -> int:
        if not self.enabled:
            return 0
        coordinates = selected_mask.nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            return 0
        coords_cpu = coordinates.detach().cpu()
        token_ids = response_ids[selected_mask].detach().cpu().tolist()
        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
        keys = (
            ("D", "C", "D_norm", "C_norm", "s_TA")
            if self.method == "ta"
            else ("Delta", "A", "F", "B", "s_RAC")
        )
        values = {
            key: diagnostics[key][selected_mask].detach().float().cpu().tolist()
            for key in keys
        }
        path = self._path(step)
        with gzip.open(path, "at", encoding="utf-8", compresslevel=6) as handle:
            for offset, (batch_index_tensor, position_tensor) in enumerate(coords_cpu):
                batch_index, position = int(batch_index_tensor), int(position_tensor)
                row = {
                    "training_step": step,
                    "sample_id": sample_ids[batch_index],
                    "dataset_index": int(dataset_indices[batch_index]),
                    "batch_index": batch_index,
                    "response_position": position,
                    "token_id": int(token_ids[offset]),
                    "token_text": str(token_texts[offset]),
                }
                row.update({key: float(values[key][offset]) for key in keys})
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        return len(token_ids)
