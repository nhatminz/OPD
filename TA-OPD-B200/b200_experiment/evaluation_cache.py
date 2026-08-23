from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .evaluation import _resolve_benchmark_file


BASE_EVALUATION_CACHE_SCHEMA = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_signature(model_path: Path) -> list[dict[str, Any]]:
    """Fingerprint a local snapshot without reading multi-GB weight shards."""
    signature = []
    for path in sorted(item for item in model_path.rglob("*") if item.is_file()):
        stat = path.stat()
        row: dict[str, Any] = {
            "path": str(path.relative_to(model_path)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        # Hash the small files which control architecture, tokenizer, and chat
        # rendering.  Size/mtime is sufficient for immutable weight snapshots.
        if stat.st_size <= 4 * 1024 * 1024:
            row["sha256"] = _sha256_file(path)
        signature.append(row)
    return signature


def base_evaluation_cache_key(
    config: dict[str, Any],
    runtime_settings: dict[str, Any],
    model_path: str | Path,
) -> str:
    """Return a method-independent key for an untouched-base evaluation."""
    model_path = Path(model_path).expanduser().resolve()
    benchmark_names = tuple(runtime_settings.get("benchmark_names", ()))
    benchmark_signatures = []
    for name in benchmark_names:
        spec = config["evaluation"]["benchmarks"][name]
        path = _resolve_benchmark_file(name, spec["path"])
        stat = path.stat()
        benchmark_signatures.append(
            {
                "name": name,
                "spec": spec,
                "path": str(path),
                "size": stat.st_size,
                "sha256": _sha256_file(path),
            }
        )
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = "unavailable"
    payload = {
        "schema": BASE_EVALUATION_CACHE_SCHEMA,
        "evaluator_sha256": _sha256_file(
            Path(__file__).resolve().with_name("vllm_evaluation.py")
        ),
        "vllm_version": vllm_version,
        "model_path": str(model_path),
        "model_signature": _model_signature(model_path),
        "model_dtype": config.get("models", {}).get("dtype"),
        "chat_template_kwargs": config.get("data", {}).get(
            "chat_template_kwargs", {}
        ),
        "runtime_settings": runtime_settings,
        "benchmarks": benchmark_signatures,
    }
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_base_cache_root(
    config: dict[str, Any], cache_dir: str | Path | None = None
) -> Path:
    configured = cache_dir
    if configured is None:
        configured = config.get("training_evaluation", {}).get("base_cache_dir")
    repo_root = Path(__file__).resolve().parents[1]
    if configured in (None, ""):
        return (repo_root / "outputs" / ".base_eval_cache").resolve()
    path = Path(configured).expanduser()
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return float(actual) == expected
        except (TypeError, ValueError):
            return False
    return actual == expected


def load_compatible_evaluation(
    directory: str | Path,
    model_path: str | Path,
    runtime_settings: dict[str, Any],
    *,
    expected_cache_key: str | None = None,
) -> dict[str, Any] | None:
    directory = Path(directory).resolve()
    summary_path = directory / "summary.json"
    if expected_cache_key is not None:
        manifest_path = directory / "cache_manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            manifest.get("schema") != BASE_EVALUATION_CACHE_SCHEMA
            or manifest.get("cache_key") != expected_cache_key
        ):
            return None
    if not summary_path.is_file():
        return None
    try:
        suite = json.loads(summary_path.read_text(encoding="utf-8"))
        if Path(suite["model_path"]).resolve() != Path(model_path).resolve():
            return None
        parameters = suite["parameters"]
        expected_parameters = {
            "backend": runtime_settings.get("backend", "vllm"),
            "max_new_tokens": int(runtime_settings["max_new_tokens"]),
            "limit": runtime_settings.get("limit"),
            "temperature": float(runtime_settings["temperature"]),
            "top_p": float(runtime_settings["top_p"]),
            "num_responses": int(runtime_settings["num_responses"]),
        }
        if any(
            not _same_value(parameters.get(key), value)
            for key, value in expected_parameters.items()
        ):
            return None
        benchmark_names = tuple(runtime_settings.get("benchmark_names", ()))
        if tuple(suite["benchmarks"]) != benchmark_names:
            return None
        for result in suite["benchmarks"].values():
            prediction = directory / Path(result["predictions"]).name
            if not prediction.is_file():
                return None
            result["predictions"] = str(prediction.resolve())
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    return suite


def materialize_evaluation(
    source: str | Path,
    destination: str | Path,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Copy cached artifacts and rewrite every embedded destination path."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    suite = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    for result in suite["benchmarks"].values():
        source_prediction = source / Path(result["predictions"]).name
        destination_prediction = destination / source_prediction.name
        temporary = destination / f".{source_prediction.name}.{uuid.uuid4().hex}.tmp"
        shutil.copy2(source_prediction, temporary)
        os.replace(temporary, destination_prediction)
        result["predictions"] = str(destination_prediction.resolve())
    suite["model"] = model_name
    summary_temporary = destination / f".summary.{uuid.uuid4().hex}.tmp"
    summary_temporary.write_text(
        json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(summary_temporary, destination / "summary.json")
    return suite


@contextmanager
def _cache_lock(cache_root: Path, cache_key: str) -> Iterator[None]:
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / f".{cache_key}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _publish_cache(
    source: Path,
    cache_entry: Path,
    cache_key: str,
    model_name: str,
) -> None:
    cache_entry.mkdir(parents=True, exist_ok=True)
    (cache_entry / "cache_manifest.json").unlink(missing_ok=True)
    (cache_entry / "summary.json").unlink(missing_ok=True)
    materialize_evaluation(source, cache_entry, model_name=model_name)
    manifest_temporary = cache_entry / f".manifest.{uuid.uuid4().hex}.tmp"
    manifest_temporary.write_text(
        json.dumps(
            {"schema": BASE_EVALUATION_CACHE_SCHEMA, "cache_key": cache_key},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_temporary, cache_entry / "cache_manifest.json")


def evaluate_or_reuse_base(
    *,
    config: dict[str, Any],
    runtime_settings: dict[str, Any],
    model_path: str | Path,
    model_name: str,
    destination: str | Path,
    evaluator: Callable[[], dict[str, Any]],
    cache_dir: str | Path | None = None,
    reuse_destination: bool = True,
) -> tuple[dict[str, Any], str]:
    """Evaluate once, then reuse the identical base result across methods.

    The status is ``local`` for a valid result already in ``destination``,
    ``shared`` for a shared-cache hit, or ``generated`` for a real vLLM run.
    """
    destination = Path(destination).resolve()
    cache_root = resolve_base_cache_root(config, cache_dir)
    cache_key = base_evaluation_cache_key(config, runtime_settings, model_path)
    cache_entry = cache_root / cache_key
    with _cache_lock(cache_root, cache_key):
        if reuse_destination:
            local = load_compatible_evaluation(
                destination, model_path, runtime_settings
            )
            if local is not None:
                if (
                    load_compatible_evaluation(
                        cache_entry,
                        model_path,
                        runtime_settings,
                        expected_cache_key=cache_key,
                    )
                    is None
                ):
                    _publish_cache(destination, cache_entry, cache_key, model_name)
                return local, "local"

        cached = load_compatible_evaluation(
            cache_entry,
            model_path,
            runtime_settings,
            expected_cache_key=cache_key,
        )
        if cached is not None:
            return (
                materialize_evaluation(
                    cache_entry, destination, model_name=model_name
                ),
                "shared",
            )

        suite = evaluator()
        verified = load_compatible_evaluation(
            destination, model_path, runtime_settings
        )
        if verified is None:
            raise RuntimeError(
                "Base evaluator completed without a compatible summary/prediction set"
            )
        _publish_cache(destination, cache_entry, cache_key, model_name)
        return suite, "generated"
