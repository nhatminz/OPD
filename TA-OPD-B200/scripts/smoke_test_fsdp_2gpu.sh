#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
source "${SCRIPT_DIR}/common_b200.sh"

export TRAIN_NPROC_PER_NODE=2
if (( $(visible_gpu_count) < 2 )); then
  echo "FSDP smoke test requires two visible GPUs" >&2
  exit 1
fi

METHOD="${METHOD:-opd}"
case "${METHOD}" in
  opd) METHOD_CONFIG="${OPD_CONFIG}" ;;
  ta) METHOD_CONFIG="${TA_CONFIG}" ;;
  rac) METHOD_CONFIG="${RAC_CONFIG}" ;;
  *) echo "METHOD must be opd, ta, or rac" >&2; exit 1 ;;
esac

SMOKE_OUTPUT="${SMOKE_OUTPUT:-${REPO_DIR}/outputs/fsdp_smoke_${METHOD}_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "${SMOKE_OUTPUT}" ]]; then
  echo "Refusing to overwrite smoke output: ${SMOKE_OUTPUT}" >&2
  exit 1
fi

cd "${REPO_DIR}"
run_training_cli train \
  --config "${METHOD_CONFIG}" \
  "${ASSET_CONFIG_ARGS[@]}" \
  --set "experiment.output_dir=${SMOKE_OUTPUT}" \
  --set "distributed.strategy=fsdp" \
  --set "rollout.backend=vllm" \
  --set "rollout.batch_size=4" \
  --set "rollout.num_responses=1" \
  --set "data.max_prompt_tokens=128" \
  --set "rollout.max_new_tokens=64" \
  --set "rollout.vllm.max_model_len=256" \
  --set "rollout.vllm.max_num_seqs=2" \
  --set "rollout.vllm.max_concurrent_requests=2" \
  --set "rollout.vllm.gpu_memory_utilization=0.20" \
  --set "rollout.vllm.logprob_sanity.enabled=true" \
  --set "rollout.vllm.logprob_sanity.max_tokens_per_rank=32" \
  --set "rollout.vllm.logprob_sanity.tolerance=0.10" \
  --set "rollout.vllm.logprob_sanity.fail_on_mismatch=true" \
  --set "selector.score_micro_batch_size=1" \
  --set "training.micro_batch_size_per_gpu=1" \
  --set "training.max_steps=2" \
  --set "training.save_checkpoints=true" \
  --set "training.save_interval=2" \
  --set "training.save_optimizer=true" \
  --set "training_evaluation.enabled=false" \
  --set "logging.selected_tokens_enabled=false" \
  --set "logging.token_score_stats_enabled=false" \
  --set "logging.tensorboard.enabled=true"

"${PYTHON_BIN}" - "${SMOKE_OUTPUT}" "${METHOD}" <<'PY'
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(sys.argv[1]).resolve()
method = sys.argv[2]
rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line]
if [row["step"] for row in rows] != [1, 2]:
    raise SystemExit(f"Expected smoke steps [1, 2], got {[row['step'] for row in rows]}")
last = rows[-1]
if last["distributed_strategy"] != "fsdp" or last["distributed_world_size"] != 2:
    raise SystemExit("Smoke run did not use two-rank FSDP")
if last["global_prompt_batch_size"] != 4 or last["micro_batch_size_per_gpu"] != 1:
    raise SystemExit("Smoke batch layout is incorrect")
if not last["vllm_logprob_sanity"].get("passed"):
    raise SystemExit(f"vLLM/HF log-prob validation failed: {last['vllm_logprob_sanity']}")
checkpoint = root / "final"
for name in ("config.json", "optimizer.pt"):
    if not (checkpoint / name).is_file():
        raise SystemExit(f"Missing FSDP checkpoint artifact: {checkpoint / name}")
if not list(checkpoint.glob("*.safetensors")):
    raise SystemExit("FSDP export did not produce safetensors weights")
events = list((root / "tensorboard").glob("events.out.tfevents.*"))
if not events:
    raise SystemExit("TensorBoard event file is missing")
accumulator = EventAccumulator(str(root / "tensorboard"))
accumulator.Reload()
tags = set(accumulator.Tags().get("scalars", []))
base = {
    "train/loss", "train/grad_norm", "train/learning_rate",
    "opd/advantage_abs_mean", "opd/teacher_student_logprob_gap",
    "opd/clip_fraction", "rollout/response_length_mean",
    "rollout/eos_fraction", "system/step_time", "system/rollout_time",
    "system/tokens_per_second", "system/peak_vram_gb",
    "debug/vllm_hf_logprob_mae",
}
extra = {"ta/selected_token_fraction"} if method == "ta" else set()
extra |= {"rac/effective_token_fraction"} if method == "rac" else set()
if tags != base | extra:
    raise SystemExit(f"Unexpected TensorBoard tags: {sorted(tags)}")
print(f"Validated two-GPU FSDP smoke output: {root}")
PY

RELOAD_CHECKPOINT="${SMOKE_OUTPUT}/final"
if [[ "${SMOKE_TEST_RESUME:-true}" == "true" ]]; then
  RESUME_OUTPUT="${SMOKE_OUTPUT}_resume"
  if [[ -e "${RESUME_OUTPUT}" ]]; then
    echo "Refusing to overwrite smoke resume output: ${RESUME_OUTPUT}" >&2
    exit 1
  fi
  run_training_cli train \
    --config "${METHOD_CONFIG}" \
    "${ASSET_CONFIG_ARGS[@]}" \
    --set "experiment.output_dir=${RESUME_OUTPUT}" \
    --set "distributed.strategy=fsdp" \
    --set "rollout.backend=vllm" \
    --set "rollout.batch_size=4" \
    --set "rollout.num_responses=1" \
    --set "data.max_prompt_tokens=128" \
    --set "rollout.max_new_tokens=64" \
    --set "rollout.vllm.max_model_len=256" \
    --set "rollout.vllm.max_num_seqs=2" \
    --set "rollout.vllm.max_concurrent_requests=2" \
    --set "rollout.vllm.gpu_memory_utilization=0.20" \
    --set "rollout.vllm.logprob_sanity.enabled=true" \
    --set "rollout.vllm.logprob_sanity.max_tokens_per_rank=32" \
    --set "rollout.vllm.logprob_sanity.tolerance=0.10" \
    --set "rollout.vllm.logprob_sanity.fail_on_mismatch=true" \
    --set "selector.score_micro_batch_size=1" \
    --set "training.micro_batch_size_per_gpu=1" \
    --set "training.max_steps=3" \
    --set "training.resume_from_checkpoint=${SMOKE_OUTPUT}/final" \
    --set "training.save_checkpoints=true" \
    --set "training.save_interval=3" \
    --set "training.save_optimizer=true" \
    --set "training_evaluation.enabled=false" \
    --set "logging.selected_tokens_enabled=false" \
    --set "logging.token_score_stats_enabled=false" \
    --set "logging.tensorboard.enabled=true"
  "${PYTHON_BIN}" - "${RESUME_OUTPUT}" "${SMOKE_OUTPUT}/final" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
source = str(Path(sys.argv[2]).resolve())
rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line]
if len(rows) != 1 or rows[0]["step"] != 3 or rows[0]["resumed_from"] != source:
    raise SystemExit(f"FSDP true-resume validation failed: {rows}")
if not (root / "final" / "optimizer.pt").is_file():
    raise SystemExit("Resumed FSDP optimizer checkpoint is missing")
print(f"Validated FSDP optimizer resume at step 3: {root}")
PY
  RELOAD_CHECKPOINT="${RESUME_OUTPUT}/final"
fi

if [[ "${SMOKE_RELOAD_CHECKPOINT:-true}" == "true" ]]; then
  "${PYTHON_BIN}" - "${RELOAD_CHECKPOINT}" <<'PY'
import sys
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    sys.argv[1], device_map="cpu", low_cpu_mem_usage=True
)
if not model.state_dict():
    raise SystemExit("Reloaded checkpoint has no parameters")
print("Reloaded the exported Hugging Face checkpoint successfully")
PY
fi

echo "FSDP smoke test passed: ${SMOKE_OUTPUT}"
