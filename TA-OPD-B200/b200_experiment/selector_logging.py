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
        rank: int = 0,
        world_size: int = 1,
    ):
        self.root = Path(output_dir) / "selector_scores"
        self.tokenizer = tokenizer
        self.method = method
        self.chunk_steps = max(1, int(chunk_steps))
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "gzip JSONL; concatenated gzip members are valid",
                "scope": "selected/accepted response positions only",
                "method": method,
                "chunk_steps": self.chunk_steps,
                "distributed_world_size": self.world_size,
                "distributed_file_pattern": (
                    "selected_steps_*_rank-*.jsonl.gz"
                    if self.world_size > 1
                    else "selected_steps_*.jsonl.gz"
                ),
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
                else ["g", "alignment", "R", "M", "V", "z", "w"],
            }
            if self.rank == 0:
                with (self.root / "manifest.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(manifest, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")

    def _path(self, step: int) -> Path:
        first = ((step - 1) // self.chunk_steps) * self.chunk_steps + 1
        last = first + self.chunk_steps - 1
        suffix = f"_rank-{self.rank:05d}" if self.world_size > 1 else ""
        return self.root / (f"selected_steps_{first:06d}_{last:06d}{suffix}.jsonl.gz")

    def write(
        self,
        *,
        step: int,
        dataset_indices: list[int],
        sample_ids: list[str],
        response_ids: torch.Tensor,
        selected_mask: torch.Tensor,
        diagnostics: dict[str, Any],
        batch_index_offset: int = 0,
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
            else ("g", "alignment", "R", "M", "V", "z", "w")
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
                    "batch_index": int(batch_index_offset) + batch_index,
                    "response_position": position,
                    "token_id": int(token_ids[offset]),
                    "token_text": str(token_texts[offset]),
                }
                row.update({key: float(values[key][offset]) for key in keys})
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        return len(token_ids)


class TokenScoreStatsLogger:
    """Compact all-valid-token histograms, quantiles, and bounded samples."""

    def __init__(
        self,
        output_dir: str | Path,
        method: str,
        interval: int = 50,
        bins: int = 64,
        raw_sample_size: int = 2048,
        enabled: bool = True,
    ):
        self.root = Path(output_dir) / "token_score_stats"
        self.method = method
        self.interval = max(1, int(interval))
        self.bins = max(2, int(bins))
        self.raw_sample_size = max(0, int(raw_sample_size))
        self.enabled = bool(enabled)
        self.ranges = (
            {"D": (0.0, 10.0), "C": (0.0, 1.0), "s_TA": (0.0, 1.0)}
            if method == "ta"
            else {
                key: (0.0, 1.0)
                for key in ("g", "alignment", "V", "z", "w")
            }
        )
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "one JSON file per logged optimizer step",
                "scope": "all valid response positions in the global rollout batch",
                "method": method,
                "logged_steps": "step 1, every configured interval, and final step",
                "bins": self.bins,
                "histogram_ranges": self.ranges,
                "raw_sample_size_max": self.raw_sample_size,
                "note": "D values outside [0,10] are counted in underflow/overflow; normalized quantities use [0,1].",
            }
            (self.root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    def should_log(self, step: int, final_step: int) -> bool:
        return self.enabled and (
            step == 1 or step == final_step or step % self.interval == 0
        )

    def write(
        self, step: int, final_step: int, diagnostics: dict[str, Any]
    ) -> Path | None:
        if not self.should_log(step, final_step):
            return None
        payload: dict[str, Any] = {
            "step": int(step),
            "method": self.method,
            "scope": "global_valid_response_tokens",
            "scores": {},
        }
        quantile_levels = torch.tensor(
            [0.05, 0.25, 0.50, 0.75, 0.95],
            device=next(
                value.device for value in diagnostics.values() if torch.is_tensor(value)
            ),
        )
        for key, (low, high) in self.ranges.items():
            values = diagnostics[key].detach().float().reshape(-1)
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                continue
            clipped = values.clamp(low, high)
            counts = torch.histc(clipped, bins=self.bins, min=low, max=high)
            edges = torch.linspace(
                low, high, self.bins + 1, device=values.device, dtype=torch.float32
            )
            quantiles = torch.quantile(values, quantile_levels)
            sample_count = min(values.numel(), self.raw_sample_size)
            if sample_count:
                indices = torch.linspace(
                    0,
                    values.numel() - 1,
                    sample_count,
                    device=values.device,
                ).long()
                sample = values.index_select(0, indices)
            else:
                sample = values.new_empty((0,))
            payload["scores"][key] = {
                "count": int(values.numel()),
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "quantiles": {
                    name: float(value)
                    for name, value in zip(
                        ("q05", "q25", "q50", "q75", "q95"), quantiles
                    )
                },
                "histogram": {
                    "edges": edges.cpu().tolist(),
                    "counts": counts.long().cpu().tolist(),
                    "underflow": int((values < low).sum()),
                    "overflow": int((values > high).sum()),
                },
                "sample": sample.cpu().tolist(),
            }
        destination = self.root / f"step-{step:06d}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination
