from __future__ import annotations

import csv
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from .models import choose_attention_implementation, model_dtype_kwargs

transformers_logging.disable_progress_bar()

BENCHMARK_ORDER = ("MATH-500", "AIME24", "AIME25")
MODEL_ORDER = ("Base", "TA-OPD", "RAC")
QUESTION_ALIASES = ("problem", "question", "prompt", "input", "query")
ANSWER_ALIASES = (
    "answer",
    "ground_truth",
    "groundtruth",
    "reference_answer",
    "target",
    "final_answer",
    "solution",
)
ID_ALIASES = ("id", "index", "sample_id", "uuid")


def _plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        return [
            {key: _plain(value) for key, value in row.items()}
            for row in pd.read_parquet(path).to_dict("records")
        ]
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError(f"Unsupported evaluation file: {path}")


def _resolve_benchmark_file(name: str, path: str | Path) -> Path:
    path = Path(path).resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"{name} path does not exist: {path}")
    preferred = [
        path / "test.jsonl",
        path / "test-00000-of-00001.parquet",
        path / "test.parquet",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    candidates = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.parquet"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one unambiguous data file for {name} under {path}; found {candidates}"
        )
    return candidates[0]


def _resolve_column(
    columns: list[str], explicit: str | None, aliases: tuple[str, ...], kind: str
) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(
                f"Configured {kind} column {explicit!r} is absent; columns={columns}"
            )
        return explicit
    normalized = {_normalized_key(column): column for column in columns}
    for alias in aliases:
        if _normalized_key(alias) in normalized:
            return normalized[_normalized_key(alias)]
    raise ValueError(
        f"Cannot infer {kind} column from {columns}; set {kind}_key in the config"
    )


def load_benchmark(
    name: str, spec: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    file_path = _resolve_benchmark_file(name, spec["path"])
    rows = _read_rows(file_path)
    if not rows:
        raise ValueError(f"{name} is empty: {file_path}")
    columns = list(rows[0])
    question_key = _resolve_column(
        columns, spec.get("question_key"), QUESTION_ALIASES, "question"
    )
    answer_key = _resolve_column(
        columns, spec.get("answer_key"), ANSWER_ALIASES, "answer"
    )
    try:
        id_key = _resolve_column(columns, spec.get("id_key"), ID_ALIASES, "id")
    except ValueError:
        id_key = None
    normalized = []
    for index, row in enumerate(rows):
        question, answer = row.get(question_key), row.get(answer_key)
        if (
            question is None
            or answer is None
            or not str(question).strip()
            or not str(answer).strip()
        ):
            raise ValueError(f"{name} row {index} has an empty question/answer")
        normalized.append(
            {
                "id": str(row.get(id_key, index) if id_key else index),
                "problem": str(question),
                "answer": str(answer),
            }
        )
    schema = {
        "benchmark": name,
        "file": str(file_path),
        "format": file_path.suffix.lower().lstrip("."),
        "rows": len(rows),
        "columns": columns,
        "question_key": question_key,
        "answer_key": answer_key,
        "id_key": id_key,
        "examples": [
            {"id": row["id"], "problem": row["problem"][:500], "answer": row["answer"]}
            for row in normalized[:3]
        ],
    }
    return normalized, schema


def inspect_evaluation_data(config: dict[str, Any]) -> dict[str, Any]:
    specs = config["evaluation"]["benchmarks"]
    report = {}
    for name in BENCHMARK_ORDER:
        _, report[name] = load_benchmark(name, specs[name])
    return report


def _load_eval_model(path: str | Path, config: dict[str, Any], device: torch.device):
    path = Path(path).resolve()
    attention = choose_attention_implementation(
        config["models"].get("attention_implementation", "auto")
    )
    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            is_trainable=False,
            attn_implementation=attention,
            **model_dtype_kwargs(torch.bfloat16),
        ).to(device)
        base_path = json.loads(adapter_config.read_text(encoding="utf-8"))[
            "base_model_name_or_path"
        ]
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            attn_implementation=attention,
            low_cpu_mem_usage=True,
            **model_dtype_kwargs(torch.bfloat16),
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def _grade(response: str, answer: str) -> bool:
    try:
        from math_verify import parse, verify

        return bool(verify(parse(str(answer)), parse(str(response))))
    except Exception:

        def normalized(value: str) -> str:
            boxed = re.findall(r"\\boxed\{([^{}]+)\}", value)
            value = boxed[-1] if boxed else value
            return re.sub(r"[\s,$]", "", value).lower().strip(".")

        return normalized(response) == normalized(answer)


@torch.inference_mode()
def evaluate_loaded_suite(
    model,
    tokenizer,
    model_name: str,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    runtime_settings: dict[str, Any] | None = None,
):
    """Evaluate an already-loaded model without changing its training/RNG state."""
    device = next(model.parameters()).device
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = {**config["evaluation"], **(runtime_settings or {})}
    batch_size = int(evaluation.get("batch_size", 16))
    max_new_tokens = int(evaluation.get("max_new_tokens", 2048))
    limit = evaluation.get("limit")
    benchmark_names = tuple(evaluation.get("benchmark_names", BENCHMARK_ORDER))
    unknown = set(benchmark_names) - set(BENCHMARK_ORDER)
    if unknown:
        raise ValueError(f"Unknown training-evaluation benchmarks: {sorted(unknown)}")
    loaded = {}
    total_batches = 0
    for benchmark in benchmark_names:
        records, schema = load_benchmark(
            benchmark, config["evaluation"]["benchmarks"][benchmark]
        )
        if limit is not None:
            records = records[: int(limit)]
        loaded[benchmark] = (records, schema)
        total_batches += math.ceil(len(records) / batch_size)
    suite = {
        "model": model_name,
        "benchmarks": {},
        "parameters": {
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "limit": limit,
            "do_sample": False,
            "temperature": 0.0,
        },
    }
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )
    previous_training = model.training
    previous_use_cache = model.config.use_cache
    model.eval()
    progress = tqdm(
        total=total_batches,
        desc=f"Eval {model_name}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    try:
        for benchmark in benchmark_names:
            records, schema = loaded[benchmark]
            correct, total = 0, 0
            prediction_path = output_dir / (
                f"{benchmark.lower().replace('-', '_')}_predictions.jsonl.gz"
            )
            with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
                for begin in range(0, len(records), batch_size):
                    progress.set_postfix_str(
                        f"{benchmark} accuracy={correct / max(total, 1):.3f}"
                    )
                    batch = records[begin : begin + batch_size]
                    messages = [
                        [
                            {
                                "role": "user",
                                "content": (
                                    "Solve the problem step by step. End with only the final answer inside \\boxed{}.\n\n"
                                    + row["problem"]
                                ),
                            }
                        ]
                        for row in batch
                    ]
                    template_kwargs = dict(
                        config["data"].get("chat_template_kwargs", {})
                    )
                    prompts = [
                        tokenizer.apply_chat_template(
                            item,
                            tokenize=False,
                            add_generation_prompt=True,
                            **template_kwargs,
                        )
                        for item in messages
                    ]
                    encoded = tokenizer(
                        prompts,
                        padding=True,
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).to(device)
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        temperature=None,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
                    responses = tokenizer.batch_decode(
                        generated[:, encoded.input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )
                    for row, response in zip(batch, responses):
                        passed = _grade(response, row["answer"])
                        correct += int(passed)
                        total += 1
                        handle.write(
                            json.dumps(
                                {**row, "response": response, "correct": passed},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    progress.update(1)
            suite["benchmarks"][benchmark] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / max(total, 1),
                "predictions": str(prediction_path),
                "schema": schema,
            }
    finally:
        progress.close()
        model.config.use_cache = previous_use_cache
        model.train(previous_training)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(suite, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return suite


@torch.inference_mode()
def evaluate_suite(
    model_name: str,
    model_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
):
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Model name must be one of {MODEL_ORDER}, got {model_name!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA")
    device = torch.device("cuda", 0)
    model, tokenizer = _load_eval_model(model_path, config, device)
    try:
        suite = evaluate_loaded_suite(model, tokenizer, model_name, config, output_dir)
        suite["model_path"] = str(Path(model_path).resolve())
        with (Path(output_dir).resolve() / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(suite, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return suite
    finally:
        del model
        torch.cuda.empty_cache()


def aggregate_evaluations(
    model_result_dirs: dict[str, str | Path], output_dir: str | Path
):
    if tuple(model_result_dirs) != MODEL_ORDER:
        raise ValueError(f"Aggregation requires exactly this order: {MODEL_ORDER}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, details = [], {}
    for model_name in MODEL_ORDER:
        source = Path(model_result_dirs[model_name]).resolve() / "summary.json"
        result = json.loads(source.read_text(encoding="utf-8"))
        details[model_name] = result
        row = {"Method": model_name}
        for benchmark in BENCHMARK_ORDER:
            row[benchmark] = result["benchmarks"][benchmark]["accuracy"]
        rows.append(row)
    payload = {"schema_version": 1, "rows": rows, "details": details}
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("Method", *BENCHMARK_ORDER))
        writer.writeheader()
        writer.writerows(rows)
    return rows
