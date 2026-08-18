from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import load_config
from .evaluation import BENCHMARK_ORDER, _grade, load_benchmark


def _resolve_gpu_memory_utilization(vllm_settings: dict[str, Any]) -> float:
    requested = vllm_settings.get("gpu_memory_utilization", "auto")
    if str(requested).lower() != "auto":
        utilization = float(requested)
        if not 0.0 < utilization <= 1.0:
            raise ValueError("vLLM gpu_memory_utilization must be in (0, 1]")
        return utilization
    if not torch.cuda.is_available():
        raise RuntimeError("Automatic vLLM memory sizing requires CUDA")
    headroom_gib = float(vllm_settings.get("gpu_headroom_gib", 4))
    if headroom_gib < 0:
        raise ValueError("vLLM gpu_headroom_gib must be non-negative")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    usable_bytes = free_bytes - int(headroom_gib * 2**30)
    if usable_bytes <= 0:
        raise RuntimeError(
            f"Only {free_bytes / 2**30:.1f} GiB VRAM is free, below the "
            f"configured {headroom_gib:.1f} GiB headroom"
        )
    # Keep a tiny driver/workspace margin even when this is the only process.
    return min(0.98, usable_bytes / total_bytes)


def _prompt(tokenizer, problem: str, config: dict[str, Any]) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the problem step by step. End with only the final answer "
                "inside \\boxed{}.\n\n" + problem
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **dict(config["data"].get("chat_template_kwargs", {})),
    )


def evaluate_vllm_suite(
    model_name: str,
    model_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    # Import here so unit tests and the Hugging Face fallback do not require an
    # initialized vLLM/CUDA runtime.
    from vllm import LLM, SamplingParams

    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_new_tokens = int(runtime_settings.get("max_new_tokens", 2048))
    limit = runtime_settings.get("limit")
    benchmark_names = tuple(runtime_settings.get("benchmark_names", BENCHMARK_ORDER))
    unknown = set(benchmark_names) - set(BENCHMARK_ORDER)
    if unknown:
        raise ValueError(f"Unknown training-evaluation benchmarks: {sorted(unknown)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    loaded: dict[str, tuple[list[dict[str, str]], dict[str, Any]]] = {}
    prompts: list[str] = []
    prompt_rows: list[tuple[str, dict[str, str]]] = []
    for benchmark in benchmark_names:
        records, schema = load_benchmark(
            benchmark, config["evaluation"]["benchmarks"][benchmark]
        )
        if limit is not None:
            records = records[: int(limit)]
        loaded[benchmark] = (records, schema)
        for row in records:
            prompts.append(_prompt(tokenizer, row["problem"], config))
            prompt_rows.append((benchmark, row))

    vllm_settings = dict(runtime_settings.get("vllm", {}))
    engine_kwargs: dict[str, Any] = {
        "model": str(model_path),
        "tokenizer": str(model_path),
        "dtype": config["models"].get("dtype", "bfloat16"),
        "tensor_parallel_size": int(vllm_settings.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": _resolve_gpu_memory_utilization(vllm_settings),
        "max_num_seqs": int(vllm_settings.get("max_num_seqs", 256)),
        "enable_prefix_caching": bool(vllm_settings.get("enable_prefix_caching", True)),
        "disable_log_stats": True,
        "seed": int(vllm_settings.get("seed", 1234)),
        "trust_remote_code": False,
        "generation_config": "vllm",
    }
    max_model_len = vllm_settings.get("max_model_len", 4096)
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = int(max_model_len)

    tqdm.write(
        f"Eval {model_name}: loading vLLM engine for {len(prompts)} full-dataset samples..."
    )
    engine = LLM(**engine_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    # One call produces one tqdm progress bar for all configured benchmarks.
    generated = engine.generate(prompts, sampling, use_tqdm=True) if prompts else []
    if len(generated) != len(prompt_rows):
        raise RuntimeError(
            f"vLLM returned {len(generated)} outputs for {len(prompt_rows)} prompts"
        )

    suite: dict[str, Any] = {
        "model": model_name,
        "model_path": str(model_path),
        "benchmarks": {},
        "parameters": {
            "backend": "vllm",
            "max_new_tokens": max_new_tokens,
            "limit": limit,
            "do_sample": False,
            "temperature": 0.0,
            **{key: value for key, value in engine_kwargs.items() if key != "model"},
        },
    }
    grouped: dict[str, list[tuple[dict[str, str], str]]] = {
        name: [] for name in benchmark_names
    }
    for (benchmark, row), request_output in zip(prompt_rows, generated):
        response = request_output.outputs[0].text
        grouped[benchmark].append((row, response))

    grade_progress = tqdm(
        total=len(prompt_rows),
        desc=f"Grade {model_name}",
        unit="sample",
        dynamic_ncols=True,
        leave=True,
        disable=False,
        mininterval=0.2,
    )
    for benchmark in benchmark_names:
        records, schema = loaded[benchmark]
        prediction_path = output_dir / (
            f"{benchmark.lower().replace('-', '_')}_predictions.jsonl.gz"
        )
        correct = graded = 0
        with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
            for row, response in grouped[benchmark]:
                passed = _grade(response, row["answer"])
                correct += int(passed)
                graded += 1
                handle.write(
                    json.dumps(
                        {**row, "response": response, "correct": passed},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                grade_progress.set_postfix_str(
                    f"{benchmark} accuracy={correct / max(graded, 1):.3f}",
                    refresh=False,
                )
                grade_progress.update(1)
        suite["benchmarks"][benchmark] = {
            "correct": correct,
            "total": len(records),
            "accuracy": correct / max(len(records), 1),
            "predictions": str(prediction_path),
            "schema": schema,
        }
    grade_progress.close()

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(suite, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM periodic evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--settings-json", required=True)
    return parser


def main() -> int:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    args = build_parser().parse_args()
    evaluate_vllm_suite(
        args.name,
        args.model,
        load_config(args.config),
        args.output,
        json.loads(args.settings_json),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
