#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONUNBUFFERED=1

BASE_CONFIG="${REPO_DIR}/configs/qwen3_b200_base.yaml"
TA_CONFIG="${REPO_DIR}/configs/qwen3_b200_ta.yaml"
RAC_CONFIG="${REPO_DIR}/configs/qwen3_b200_rac.yaml"
AUTOTUNE_CONFIG="${AUTOTUNE_CONFIG:-${REPO_DIR}/configs/qwen3_b200_autotuned.yaml}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${REPO_DIR}/results/preflight.json}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    return 1
  fi
}

require_b200_validation() {
  require_file "${PREFLIGHT_REPORT}" || {
    echo "Run: bash scripts/smoke_test_b200.sh" >&2
    return 1
  }
  "${PYTHON_BIN}" - "${PREFLIGHT_REPORT}" <<'PY'
import json
import sys
from pathlib import Path

preflight = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if preflight.get("status") != "passed":
    raise SystemExit("B200 preflight did not pass")
PY
  if batch_autotune_enabled; then
    require_file "${AUTOTUNE_CONFIG}" || {
      echo "Run: USE_BATCH_AUTOTUNE=true bash scripts/smoke_test_b200.sh" >&2
      return 1
    }
    "${PYTHON_BIN}" - "${AUTOTUNE_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path

autotune = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not autotune.get("autotune", {}).get("validated"):
    raise SystemExit("B200 batch config is not validated")
batch = autotune["autotune"]["selected_batch_size"]
if batch != autotune["rollout"]["batch_size"] or batch != autotune["training"]["micro_batch_size"]:
    raise SystemExit("Autotune batch/micro-batch values are inconsistent")
print(f"Autotuned B200 batch size: {batch} (gradient accumulation: 1)")
PY
  else
    "${PYTHON_BIN}" - "${TRAIN_BATCH_SIZE:-64}" <<'PY'
import sys

batch = int(sys.argv[1])
if batch <= 0:
    raise SystemExit("TRAIN_BATCH_SIZE must be a positive integer")
print(f"Fixed training batch size: {batch} (micro-batch: {batch}, gradient accumulation: 1)")
PY
  fi
}

batch_autotune_enabled() {
  case "${USE_BATCH_AUTOTUNE:-false}" in
    true|TRUE|1|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

build_training_args() {
  local output_dir="$1"
  COMMON_TRAIN_ARGS=(
    --set "experiment.output_dir=${output_dir}"
    --set "training.epochs=${EPOCHS:-1}"
    --set "training.learning_rate=${LEARNING_RATE:-1.0e-5}"
    --set "training.save_interval=${SAVE_INTERVAL:-100}"
    --set "rollout.backend=${ROLLOUT_BACKEND:-vllm}"
    --set "rollout.max_new_tokens=${MAX_NEW_TOKENS:-256}"
    --set "rollout.vllm.gpu_memory_utilization=${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
    --set "rollout.vllm.max_num_seqs=${ROLLOUT_VLLM_MAX_NUM_SEQS:-${TRAIN_BATCH_SIZE:-64}}"
    --set "rollout.vllm.max_model_len=${ROLLOUT_VLLM_MAX_MODEL_LEN:-1024}"
    --set "rollout.vllm.max_concurrent_requests=${ROLLOUT_VLLM_MAX_CONCURRENT_REQUESTS:-${TRAIN_BATCH_SIZE:-64}}"
    --set "rollout.vllm.wake_headroom_gib=${ROLLOUT_VLLM_WAKE_HEADROOM_GIB:-2}"
    --set "token_budget.rho=${RHO:-0.10}"
    --set "selector.top_k=${TOP_K:-16}"
    --set "training_evaluation.enabled=${TRAIN_EVAL_ENABLED:-true}"
    --set "training_evaluation.backend=${TRAIN_EVAL_BACKEND:-vllm}"
    --set "training_evaluation.target_evaluations=${TRAIN_EVAL_TARGET:-16}"
    --set "training_evaluation.limit=null"
    --set "training_evaluation.batch_size=${TRAIN_EVAL_BATCH_SIZE:-16}"
    --set "training_evaluation.max_new_tokens=${TRAIN_EVAL_MAX_NEW_TOKENS:-2048}"
    --set "training_evaluation.vllm.tensor_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    --set "training_evaluation.vllm.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-auto}"
    --set "training_evaluation.vllm.gpu_headroom_gib=${VLLM_GPU_HEADROOM_GIB:-4}"
    --set "training_evaluation.vllm.max_num_seqs=${VLLM_MAX_NUM_SEQS:-256}"
    --set "training_evaluation.vllm.max_model_len=${VLLM_MAX_MODEL_LEN:-4096}"
  )
  if batch_autotune_enabled; then
    COMMON_TRAIN_ARGS=(--overlay "${AUTOTUNE_CONFIG}" "${COMMON_TRAIN_ARGS[@]}")
  else
    COMMON_TRAIN_ARGS+=(
      --set "rollout.batch_size=${TRAIN_BATCH_SIZE:-64}"
      --set "training.micro_batch_size=${TRAIN_BATCH_SIZE:-64}"
    )
  fi
  if [[ -n "${TRAIN_EVAL_INTERVAL:-}" ]]; then
    COMMON_TRAIN_ARGS+=(
      --set "training_evaluation.target_evaluations=null"
      --set "training_evaluation.interval_steps=${TRAIN_EVAL_INTERVAL}"
    )
  fi
  if [[ -n "${MAX_STEPS:-}" ]]; then
    COMMON_TRAIN_ARGS+=(--set "training.max_steps=${MAX_STEPS}")
  fi
}

plot_training_progress_if_ready() {
  local ta_output="$1"
  local rac_output="$2"
  if [[ -f "${ta_output}/eval_history.jsonl" && -f "${rac_output}/eval_history.jsonl" ]]; then
    "${PYTHON_BIN}" -m b200_experiment.cli plot-training-progress \
      --results "${RESULTS_DIR:-${REPO_DIR}/results}" \
      --ta-output "${ta_output}" \
      --rac-output "${rac_output}" \
      --smoothing-window "${SMOOTHING_WINDOW:-10}"
  else
    echo "Training-progress plot is pending until both TA and RAC histories exist."
  fi
}
