from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def discover_data_files(path: str | Path, split: str | None = None) -> list[Path]:
    path = Path(path).resolve()
    if path.is_file():
        if path.suffix.lower() not in {".parquet", ".json", ".jsonl"}:
            raise ValueError(
                f"Unsupported data file {path}; expected parquet, JSON, or JSONL"
            )
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Training data path does not exist: {path}")
    search_root = path
    if split:
        split_root = path / split
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Requested dataset split {split!r} does not exist under {path}"
            )
        search_root = split_root
    parquet = sorted(
        item for item in search_root.rglob("*.parquet") if ".cache" not in item.parts
    )
    jsonl = sorted(
        item for item in search_root.rglob("*.jsonl") if ".cache" not in item.parts
    )
    json_files = sorted(
        item for item in search_root.rglob("*.json") if ".cache" not in item.parts
    )
    files = parquet or jsonl or json_files
    if not files:
        raise FileNotFoundError(f"No parquet/JSON/JSONL files found under {path}")
    return files


def _read_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        return [
            {key: _plain(value) for key, value in row.items()}
            for row in frame.to_dict("records")
        ]
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "train", "test", "records", "examples"):
            rows = payload.get(key)
            if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                return rows
        return [payload]
    raise ValueError(f"Unsupported JSON dataset structure in {path}")


def read_records(
    path: str | Path, split: str | None = None
) -> tuple[list[dict[str, Any]], list[Path]]:
    files = discover_data_files(path, split=split)
    records: list[dict[str, Any]] = []
    for file_path in files:
        records.extend(_read_file(file_path))
    return records, files


def record_messages(
    record: dict[str, Any],
    prompt_key: str = "prompt",
    prefer_source_prompt: bool = False,
):
    if prefer_source_prompt and record.get("source_prompt"):
        messages = _plain(record["source_prompt"])
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            return messages
    prompt = record.get(prompt_key)
    if isinstance(prompt, list):
        return prompt
    if not isinstance(prompt, str):
        raise TypeError(
            f"Expected string/list in column {prompt_key!r}, got {type(prompt).__name__}"
        )
    return [{"role": "user", "content": prompt}]


def stable_sample_id(record: dict[str, Any], dataset_index: int) -> str:
    for key in ("id", "index", "uuid", "sample_id", "unique_id"):
        if key in record and record[key] is not None:
            return str(record[key])
    extra = record.get("extra_info")
    if isinstance(extra, dict):
        for key in ("id", "index", "uuid", "sample_id", "unique_id"):
            if key in extra and extra[key] is not None:
                return str(extra[key])
    return str(dataset_index)


def tokenize_prompts(
    records, tokenizer, data_config: dict[str, Any], device: torch.device
):
    template_kwargs = dict(data_config.get("chat_template_kwargs", {}))
    prompts = [
        tokenizer.apply_chat_template(
            record_messages(
                record,
                prompt_key=data_config.get("prompt_key", "prompt"),
                prefer_source_prompt=bool(
                    data_config.get("prefer_source_prompt", False)
                ),
            ),
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        for record in records
    ]
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=int(data_config.get("max_prompt_tokens", 512)),
        return_tensors="pt",
    )
    tokenizer.padding_side = previous_side
    return {key: value.to(device) for key, value in encoded.items()}, prompts


def expand_prompt_batch(
    encoded: dict[str, torch.Tensor],
    dataset_indices: list[int],
    num_responses: int,
) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    """Interleave each prompt n times, matching thunlp/OPD rollout.n behavior."""
    n = int(num_responses)
    if n <= 0:
        raise ValueError("num_responses must be positive")
    if any(value.shape[0] != len(dataset_indices) for value in encoded.values()):
        raise ValueError("Encoded prompt batch and dataset indices do not align")
    expanded = {
        key: value.repeat_interleave(n, dim=0) for key, value in encoded.items()
    }
    expanded_indices = [index for index in dataset_indices for _ in range(n)]
    response_indices = list(range(n)) * len(dataset_indices)
    return expanded, expanded_indices, response_indices


def epoch_batch_indices(
    num_records: int, batch_size: int, step: int, seed: int
) -> list[int]:
    """Deterministic full-dataset shuffle; the last batch is never padded/repeated."""
    if num_records <= 0 or batch_size <= 0:
        raise ValueError("num_records and batch_size must be positive")
    steps_per_epoch = math.ceil(num_records / batch_size)
    epoch = step // steps_per_epoch
    step_in_epoch = step % steps_per_epoch
    order = list(range(num_records))
    random.Random(seed + epoch).shuffle(order)
    begin = step_in_epoch * batch_size
    return order[begin : min(begin + batch_size, num_records)]


def build_response_valid_mask(
    response_ids: torch.Tensor, eos_token_ids: int | list[int], pad_token_id: int
):
    """Include the first EOS and exclude padding and all tokens after EOS."""
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    active = torch.ones(
        response_ids.shape[0], dtype=torch.bool, device=response_ids.device
    )
    columns = []
    for step in range(response_ids.shape[1]):
        token = response_ids[:, step]
        valid = active.clone()
        columns.append(valid)
        is_eos = torch.zeros_like(active)
        for eos_id in eos_token_ids:
            is_eos |= token.eq(int(eos_id))
        active &= ~is_eos
        active &= ~token.eq(int(pad_token_id))
    if not columns:
        return torch.zeros_like(response_ids, dtype=torch.bool)
    return torch.stack(columns, dim=1)
