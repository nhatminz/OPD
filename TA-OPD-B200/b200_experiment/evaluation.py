from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from .models import choose_attention_implementation, model_dtype_kwargs

transformers_logging.disable_progress_bar()

BENCHMARK_ORDER = ("Competition-MATH", "MATH-500", "AIME24", "AIME25")
MODEL_ORDER = ("Base", "OPD", "TA-OPD", "RAC")


def configured_benchmark_names(
    config: dict[str, Any], requested: list[str] | tuple[str, ...] | None = None
) -> tuple[str, ...]:
    """Return a validated canonical benchmark subset for this resolved config."""
    specs = config.get("evaluation", {}).get("benchmarks", {})
    names = (
        tuple(requested)
        if requested is not None
        else tuple(name for name in BENCHMARK_ORDER if name in specs)
    )
    if not names:
        raise ValueError("Evaluation must configure at least one benchmark")
    if len(names) != len(set(names)):
        raise ValueError(f"Evaluation benchmark names contain duplicates: {names}")
    unknown = set(names) - set(BENCHMARK_ORDER)
    if unknown:
        raise ValueError(f"Unknown evaluation benchmarks: {sorted(unknown)}")
    missing = [name for name in names if name not in specs]
    if missing:
        raise ValueError(f"Missing evaluation benchmark configuration for: {missing}")
    canonical = tuple(name for name in BENCHMARK_ORDER if name in names)
    if names != canonical:
        raise ValueError(
            f"Evaluation benchmarks must follow canonical order {BENCHMARK_ORDER}; "
            f"got {names}"
        )
    return names


def evaluation_metric_name(samples_per_problem: int) -> str:
    samples = int(samples_per_problem)
    if samples <= 0:
        raise ValueError("Evaluation samples per problem must be positive")
    return "accuracy" if samples == 1 else f"avg@{samples}"


def detailed_model_output_record(
    *,
    model_name: str,
    model_path: str | Path | None,
    backend: str,
    benchmark: str,
    row: dict[str, str],
    rendered_prompt: str,
    responses: list[str],
    correctness: list[bool],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Build one audit-friendly row containing every output for one problem."""
    if len(responses) != len(correctness):
        raise ValueError("Detailed responses and correctness flags must align")
    samples = len(responses)
    return {
        "schema_version": 1,
        "model_name": model_name,
        "model_path": str(Path(model_path).resolve()) if model_path else None,
        "backend": backend,
        "benchmark": benchmark,
        "problem_id": row["id"],
        "problem": row["problem"],
        "reference_answer": row["answer"],
        "rendered_prompt": rendered_prompt,
        "samples_per_problem": samples,
        "problem_score": sum(map(int, correctness)) / max(samples, 1),
        "generation": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_new_tokens": int(max_new_tokens),
            "seed": int(seed),
        },
        "outputs": [
            {
                "sample_index": sample_index,
                "response": response,
                "correct": bool(is_correct),
                "response_characters": len(response),
            }
            for sample_index, (response, is_correct) in enumerate(
                zip(responses, correctness)
            )
        ],
    }


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
ID_ALIASES = ("id", "index", "sample_id", "unique_id", "uuid")


def _plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        return [
            {key: _plain(value) for key, value in row.items()}
            for row in pd.read_parquet(path).to_dict("records")
        ]
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "test", "records", "examples"):
                rows = payload.get(key)
                if isinstance(rows, list) and all(
                    isinstance(row, dict) for row in rows
                ):
                    return rows
            return [payload]
        raise ValueError(f"Unsupported JSON evaluation structure in {path}")
    raise ValueError(f"Unsupported evaluation file: {path}")


def _resolve_benchmark_file(name: str, path: str | Path) -> Path:
    path = Path(path).resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"{name} path does not exist: {path}")
    preferred = [
        path / "test.jsonl",
        path / "test.json",
        path / "test-00000-of-00001.parquet",
        path / "test.parquet",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    candidates = (
        sorted(path.glob("*.jsonl"))
        + sorted(path.glob("*.json"))
        + sorted(path.glob("*.parquet"))
    )
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
    for name in configured_benchmark_names(config):
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
    temperature = float(evaluation.get("temperature", 0.7))
    top_p = float(evaluation.get("top_p", 0.95))
    samples_per_problem = int(evaluation.get("num_responses", 16))
    metric_name = evaluation_metric_name(samples_per_problem)
    seed = int(evaluation.get("seed", 1234))
    if temperature <= 0:
        raise ValueError("Sampled evaluation requires positive temperature")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("Evaluation top_p must be in (0, 1]")
    if samples_per_problem <= 0:
        raise ValueError("Evaluation num_responses must be positive")
    do_sample = True
    limit = evaluation.get("limit")
    benchmark_names = configured_benchmark_names(
        config, evaluation.get("benchmark_names")
    )
    loaded = {}
    total_samples = 0
    for benchmark in benchmark_names:
        records, schema = load_benchmark(
            benchmark, config["evaluation"]["benchmarks"][benchmark]
        )
        if limit is not None:
            records = records[: int(limit)]
        loaded[benchmark] = (records, schema)
        total_samples += len(records)
    suite = {
        "model": model_name,
        "benchmarks": {},
        "parameters": {
            "backend": evaluation.get("backend", "hf"),
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "limit": limit,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "num_responses": samples_per_problem,
            "metric": metric_name,
            "seed": seed,
        },
    }
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )
    previous_training = model.training
    previous_use_cache = model.config.use_cache
    model.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    detailed_output_path = output_dir / "model_outputs_detailed.jsonl.gz"
    detailed_handle = gzip.open(detailed_output_path, "wt", encoding="utf-8")
    progress = tqdm(
        total=total_samples,
        desc=f"Eval {model_name}",
        unit="sample",
        dynamic_ncols=True,
        leave=True,
        disable=False,
        mininterval=0.5,
    )
    try:
        for benchmark in benchmark_names:
            records, schema = loaded[benchmark]
            correct_generations = graded_generations = problems = 0
            problem_score_sum = 0.0
            prediction_path = output_dir / (
                f"{benchmark.lower().replace('-', '_')}_predictions.jsonl.gz"
            )
            with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
                for begin in range(0, len(records), batch_size):
                    progress.set_postfix_str(
                        f"{benchmark} {metric_name}="
                        f"{correct_generations / max(graded_generations, 1):.3f}"
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
                    generation_kwargs = {
                        "do_sample": do_sample,
                        "num_return_sequences": samples_per_problem,
                        "max_new_tokens": max_new_tokens,
                        "eos_token_id": tokenizer.eos_token_id,
                        "pad_token_id": tokenizer.pad_token_id,
                        "use_cache": True,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                    generated = model.generate(
                        **encoded,
                        **generation_kwargs,
                    )
                    flat_responses = tokenizer.batch_decode(
                        generated[:, encoded.input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )
                    expected_responses = len(batch) * samples_per_problem
                    if len(flat_responses) != expected_responses:
                        raise RuntimeError(
                            f"HF generated {len(flat_responses)} responses, expected "
                            f"{expected_responses}"
                        )
                    for row_index, row in enumerate(batch):
                        begin_response = row_index * samples_per_problem
                        responses = flat_responses[
                            begin_response : begin_response + samples_per_problem
                        ]
                        correctness = [
                            _grade(response, row["answer"]) for response in responses
                        ]
                        correct_generations += sum(map(int, correctness))
                        graded_generations += samples_per_problem
                        problems += 1
                        problem_score_sum += sum(correctness) / samples_per_problem
                        detailed_handle.write(
                            json.dumps(
                                detailed_model_output_record(
                                    model_name=model_name,
                                    model_path=getattr(
                                        getattr(model, "config", None),
                                        "_name_or_path",
                                        None,
                                    ),
                                    backend=str(evaluation.get("backend", "hf")),
                                    benchmark=benchmark,
                                    row=row,
                                    rendered_prompt=prompts[row_index],
                                    responses=responses,
                                    correctness=correctness,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_new_tokens=max_new_tokens,
                                    seed=seed,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        handle.write(
                            json.dumps(
                                {
                                    **row,
                                    "responses": responses,
                                    "correct": correctness,
                                    "problem_score": sum(correctness)
                                    / samples_per_problem,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    progress.set_postfix_str(
                        f"{benchmark} {metric_name}="
                        f"{correct_generations / max(graded_generations, 1):.3f}"
                    )
                    progress.update(len(batch))
            avg_at_n = problem_score_sum / max(problems, 1)
            benchmark_result = {
                "correct": correct_generations,
                "total": graded_generations,
                "problems": problems,
                "samples_per_problem": samples_per_problem,
                "avg_at_n": avg_at_n,
                "accuracy": avg_at_n,
                "predictions": str(prediction_path),
                "schema": schema,
            }
            if samples_per_problem == 16:
                benchmark_result["avg_at_16"] = avg_at_n
            suite["benchmarks"][benchmark] = benchmark_result
    finally:
        detailed_handle.close()
        progress.close()
        model.config.use_cache = previous_use_cache
        model.train(previous_training)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    suite["detailed_outputs"] = str(detailed_output_path.resolve())
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
    evaluation = config["evaluation"]
    backend = str(evaluation.get("backend", "hf")).lower()
    if backend == "vllm":
        from .vllm_evaluation import evaluate_vllm_suite

        return evaluate_vllm_suite(
            model_name,
            model_path,
            config,
            output_dir,
            {
                "backend": "vllm",
                "temperature": evaluation.get("temperature", 0.7),
                "top_p": evaluation.get("top_p", 0.95),
                "num_responses": evaluation.get("num_responses", 16),
                "max_new_tokens": evaluation.get("max_new_tokens", 2048),
                "limit": evaluation.get("limit"),
                "benchmark_names": list(configured_benchmark_names(config)),
                "vllm": evaluation.get("vllm", {}),
            },
        )
    if backend != "hf":
        raise ValueError("evaluation.backend must be 'vllm' or 'hf'")
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
    requested_order = tuple(model_result_dirs)
    expected_order = tuple(item for item in MODEL_ORDER if item in model_result_dirs)
    if (
        not requested_order
        or requested_order[0] != "Base"
        or requested_order != expected_order
    ):
        raise ValueError(
            f"Aggregation requires Base followed by an ordered subset of {MODEL_ORDER[1:]}; "
            f"got {requested_order}"
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, details = [], {}
    benchmark_names: tuple[str, ...] | None = None
    for model_name in requested_order:
        source = Path(model_result_dirs[model_name]).resolve() / "summary.json"
        result = json.loads(source.read_text(encoding="utf-8"))
        details[model_name] = result
        current_names = tuple(result.get("benchmarks", {}))
        unknown = set(current_names) - set(BENCHMARK_ORDER)
        canonical = tuple(name for name in BENCHMARK_ORDER if name in current_names)
        if not current_names or unknown or current_names != canonical:
            raise ValueError(
                f"{model_name} has invalid benchmark order {current_names}; "
                f"supported order is {BENCHMARK_ORDER}"
            )
        if benchmark_names is None:
            benchmark_names = current_names
        elif current_names != benchmark_names:
            raise ValueError(
                "Cannot aggregate evaluations with different benchmark sets: "
                f"{benchmark_names} vs {current_names} for {model_name}"
            )
        row = {"Method": model_name}
        for benchmark in benchmark_names:
            row[benchmark] = result["benchmarks"][benchmark]["accuracy"]
        rows.append(row)
    assert benchmark_names is not None
    payload = {
        "schema_version": 1,
        "benchmarks": list(benchmark_names),
        "rows": rows,
        "details": details,
    }
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("Method", *benchmark_names))
        writer.writeheader()
        writer.writerows(rows)
    return rows
