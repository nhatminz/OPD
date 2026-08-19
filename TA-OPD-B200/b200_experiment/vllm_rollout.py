from __future__ import annotations

import atexit
import json
import os
import signal
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import torch
from tqdm.auto import tqdm

from .scoring import RolloutBatch


def rollout_batch_from_token_ids(
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    response_token_ids: list[list[int]],
    pad_token_id: int,
) -> RolloutBatch:
    if len(response_token_ids) != prompt_ids.shape[0]:
        raise ValueError("vLLM response count does not match the prompt batch")
    width = max((len(tokens) for tokens in response_token_ids), default=0)
    if width <= 0:
        raise RuntimeError("vLLM produced no rollout tokens")
    device = prompt_ids.device
    responses = torch.full(
        (prompt_ids.shape[0], width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    valid = torch.zeros_like(responses, dtype=torch.bool)
    for row, tokens in enumerate(response_token_ids):
        if tokens:
            values = torch.tensor(tokens, dtype=torch.long, device=device)
            responses[row, : values.numel()] = values
            valid[row, : values.numel()] = True
    attention = torch.cat((prompt_attention_mask, valid.long()), dim=1)
    return RolloutBatch(
        input_ids=torch.cat((prompt_ids, responses), dim=1),
        attention_mask=attention,
        response_ids=responses,
        valid_mask=valid,
        rollout_log_probs=torch.zeros_like(responses, dtype=torch.float32),
        prompt_width=prompt_ids.shape[1],
    )


class VLLMRolloutEngine:
    """Persistent same-GPU vLLM server with per-step CUDA-IPC weight sync."""

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: Path,
        *,
        local_rank: int = 0,
        world_size: int = 1,
        port: int | None = None,
    ):
        self.config = config
        self.settings = dict(config["rollout"].get("vllm", {}))
        self.output_dir = output_dir
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.model_path = Path(config["models"]["student_path"]).resolve()
        self.served_model_name = "opd-rollout-student"
        self.process: subprocess.Popen | None = None
        self.log_handle = None
        self.sleeping = False
        self.closed = False
        self.last_metrics: dict[str, float] = {}
        self.port = int(port) if port is not None else self._free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        atexit.register(self.close)

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind(("127.0.0.1", 0))
            return int(handle.getsockname()[1])

    def _gpu_memory_utilization(self) -> float:
        try:
            utilization = float(self.settings.get("gpu_memory_utilization", 0.25))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "rollout.vllm.gpu_memory_utilization must be a number, not 'auto'"
            ) from error
        if not 0.0 < utilization <= 1.0:
            raise ValueError(
                "rollout.vllm.gpu_memory_utilization must be in the interval (0, 1]"
            )
        return utilization

    def _server_command(self) -> list[str]:
        executable = shutil.which("vllm")
        if executable is None:
            raise RuntimeError(
                "Cannot find the vllm executable in the active venv; install requirements.txt"
            )
        max_model_len = int(self.settings.get("max_model_len", 1024))
        required_length = int(self.config["data"].get("max_prompt_tokens", 512)) + int(
            self.config["rollout"].get("max_new_tokens", 256)
        )
        if max_model_len < required_length:
            raise ValueError(
                f"rollout.vllm.max_model_len={max_model_len} is below prompt+response "
                f"length {required_length}"
            )
        command = [
            executable,
            "serve",
            str(self.model_path),
            "--served-model-name",
            self.served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--dtype",
            str(self.config["models"].get("dtype", "bfloat16")),
            "--tensor-parallel-size",
            "1",
            "--gpu-memory-utilization",
            str(self._gpu_memory_utilization()),
            "--max-num-seqs",
            str(
                self.settings.get("max_num_seqs", self.config["rollout"]["batch_size"])
            ),
            "--max-model-len",
            str(max_model_len),
            "--load-format",
            "dummy",
            "--weight-transfer-config",
            json.dumps({"backend": "ipc"}),
            "--enable-sleep-mode",
            "--enforce-eager",
            "--disable-log-stats",
            "--generation-config",
            "vllm",
        ]
        if bool(self.settings.get("enable_prefix_caching", True)):
            command.append("--enable-prefix-caching")
        return command

    def _server_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["VLLM_SERVER_DEV_MODE"] = "1"
        environment["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        environment["VLLM_LOGGING_LEVEL"] = environment.get(
            "VLLM_LOGGING_LEVEL", "WARNING"
        )
        for name in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RUN_ID",
        ):
            environment.pop(name, None)
        # torchrun workers see every listed GPU. Restrict this child to the
        # worker's matching physical device so each rank owns one colocated
        # vLLM server and CUDA-IPC transfers target the correct B200.
        visible = environment.get("CUDA_VISIBLE_DEVICES", "")
        visible_devices = [item.strip() for item in visible.split(",") if item.strip()]
        if visible_devices:
            if self.local_rank >= len(visible_devices):
                raise RuntimeError(
                    f"LOCAL_RANK={self.local_rank} exceeds CUDA_VISIBLE_DEVICES={visible!r}"
                )
            environment["CUDA_VISIBLE_DEVICES"] = visible_devices[self.local_rank]
        else:
            environment["CUDA_VISIBLE_DEVICES"] = str(self.local_rank)
        return environment

    def start(self) -> None:
        if self.closed:
            raise RuntimeError("Cannot restart a closed vLLM rollout engine")
        if self.process is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_name = (
            f"vllm_rollout_server.rank-{self.local_rank:05d}.log"
            if self.world_size > 1
            else "vllm_rollout_server.log"
        )
        log_path = self.output_dir / log_name
        self.log_handle = log_path.open("a", encoding="utf-8")
        environment = self._server_environment()
        try:
            self.process = subprocess.Popen(
                self._server_command(),
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + float(
                self.settings.get("startup_timeout", 600)
            )
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self._raise_server_failure(log_path)
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=2)
                    if response.ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError(
                    f"vLLM rollout server did not become ready; see {log_path}"
                )
            self._post("/init_weight_transfer_engine", json={"init_info": {}})
            # Discard dummy weights and KV cache before loading the two HF models.
            self._sleep()
        except BaseException:
            self.close()
            raise

    def _raise_server_failure(self, log_path: Path) -> None:
        self.log_handle.flush()
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError("vLLM rollout server exited:\n" + "\n".join(tail))

    def _post(self, path: str, **kwargs):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("vLLM rollout server is not running")
        response = requests.post(
            f"{self.base_url}{path}",
            timeout=float(self.settings.get("control_timeout", 600)),
            **kwargs,
        )
        response.raise_for_status()
        return response

    def _sleep(self) -> None:
        if not self.sleeping:
            self._post("/sleep", params={"level": 2})
            self.sleeping = True

    def _wake_weights(self) -> None:
        if self.sleeping:
            self._post("/wake_up", params=[("tags", "weights")])

    def _wake_kv_cache(self) -> None:
        if self.sleeping:
            self._post("/wake_up", params=[("tags", "kv_cache")])
            self.sleeping = False

    def _sync_weights(self, model) -> None:
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        from vllm.distributed.weight_transfer.ipc_engine import (
            IPCTrainerSendWeightsArgs,
            IPCWeightTransferEngine,
        )

        trainer_args = IPCTrainerSendWeightsArgs(mode="http", url=self.base_url)
        IPCWeightTransferEngine.trainer_send_weights(
            iterator=model.named_parameters(), trainer_args=trainer_args
        )

    def _release_torch_cache_if_needed(self) -> tuple[bool, float]:
        """Return cached blocks only when the colocated engine cannot wake safely."""
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        headroom_gib = float(self.settings.get("wake_headroom_gib", 2))
        if headroom_gib < 0:
            raise ValueError("rollout.vllm.wake_headroom_gib cannot be negative")
        required_bytes = int(
            total_bytes * self._gpu_memory_utilization() + headroom_gib * 2**30
        )
        if free_bytes >= required_bytes:
            return False, 0.0
        started = time.perf_counter()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        free_after, _ = torch.cuda.mem_get_info()
        if free_after < required_bytes:
            raise RuntimeError(
                "Not enough free VRAM to wake the colocated vLLM rollout engine: "
                f"need about {required_bytes / 2**30:.1f} GiB, have "
                f"{free_after / 2**30:.1f} GiB. Lower "
                "ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION or TRAIN_BATCH_SIZE."
            )
        return True, elapsed

    def _generate_one(
        self,
        index: int,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_ids: list[int],
        seed: int,
    ) -> tuple[int, list[int]]:
        payload = {
            "model": self.served_model_name,
            "prompt": prompt_token_ids,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "seed": int(seed) + index,
            "stop_token_ids": eos_token_ids,
            "return_token_ids": True,
            "skip_special_tokens": False,
        }
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=float(self.settings.get("generation_timeout", 3600)),
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        tokens = choice.get("token_ids")
        if tokens is None:
            raise RuntimeError(
                "vLLM response omitted token_ids; this project requires vLLM >=0.17.1"
            )
        return index, [int(token) for token in tokens]

    def generate(
        self,
        model,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_ids: int | list[int],
        pad_token_id: int,
        seed: int,
        sample_seed_offset: int = 0,
    ) -> RolloutBatch:
        if self.process is None:
            raise RuntimeError("vLLM rollout server was not started")
        eos_ids = (
            [int(eos_token_ids)]
            if isinstance(eos_token_ids, int)
            else [int(item) for item in eos_token_ids]
        )
        prompts = [
            row[mask.bool()].detach().cpu().tolist()
            for row, mask in zip(prompt_ids, prompt_attention_mask)
        ]
        started = time.perf_counter()
        cache_released, cache_release_seconds = self._release_torch_cache_if_needed()
        self._wake_weights()
        torch.cuda.synchronize()
        sync_started = time.perf_counter()
        self._sync_weights(model)
        torch.cuda.synchronize()
        sync_seconds = time.perf_counter() - sync_started
        self._wake_kv_cache()
        generate_started = time.perf_counter()
        responses: list[list[int] | None] = [None] * len(prompts)
        workers = min(
            len(prompts), int(self.settings.get("max_concurrent_requests", 128))
        )
        try:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = [
                    executor.submit(
                        self._generate_one,
                        index,
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        eos_token_ids=eos_ids,
                        seed=seed + int(sample_seed_offset),
                    )
                    for index, prompt in enumerate(prompts)
                ]
                progress = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="vLLM rollout",
                    unit="sample",
                    dynamic_ncols=True,
                    leave=False,
                    disable=False,
                )
                for future in progress:
                    index, tokens = future.result()
                    responses[index] = tokens
        finally:
            sleep_started = time.perf_counter()
            self._sleep()
            sleep_seconds = time.perf_counter() - sleep_started
        generate_seconds = time.perf_counter() - generate_started - sleep_seconds
        if any(tokens is None for tokens in responses):
            raise RuntimeError("vLLM did not return every rollout response")
        self.last_metrics = {
            "weight_sync_time": sync_seconds,
            "generation_time": generate_seconds,
            "sleep_time": sleep_seconds,
            "torch_cache_released": float(cache_released),
            "torch_cache_release_time": cache_release_seconds,
            "total_time": time.perf_counter() - started,
        }
        return rollout_batch_from_token_ids(
            prompt_ids,
            prompt_attention_mask,
            [tokens for tokens in responses if tokens is not None],
            pad_token_id,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
