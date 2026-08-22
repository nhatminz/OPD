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
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"

BASE_CONFIG="${REPO_DIR}/configs/qwen3_b200_base.yaml"
OPD_CONFIG="${REPO_DIR}/configs/qwen3_b200_opd.yaml"
TA_CONFIG="${REPO_DIR}/configs/qwen3_b200_ta.yaml"
RAC_CONFIG="${REPO_DIR}/configs/qwen3_b200_rac.yaml"
AUTOTUNE_CONFIG="${AUTOTUNE_CONFIG:-${REPO_DIR}/configs/qwen3_b200_autotuned.yaml}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${REPO_DIR}/results/preflight.json}"

resolve_run_paths() {
  RUN_NAME="${RUN_NAME:-run01}"
  local opd_name="${OPD_RUN_NAME:-${RUN_NAME}}"
  local ta_name="${TA_RUN_NAME:-${RUN_NAME}}"
  local rac_name="${RAC_RUN_NAME:-${RUN_NAME}}"
  if [[ -n "${COMPARISON_NAME:-}" ]]; then
    COMPARISON_NAME="${COMPARISON_NAME}"
  elif [[ -n "${OPD_RUN_NAME:-}" ]]; then
    COMPARISON_NAME="${opd_name}_vs_${ta_name}_vs_${rac_name}"
  elif [[ -n "${TA_RUN_NAME:-}" || -n "${RAC_RUN_NAME:-}" ]]; then
    COMPARISON_NAME="${ta_name}_vs_${rac_name}"
  else
    COMPARISON_NAME="${RUN_NAME}"
  fi
  for name in "${RUN_NAME}" "${opd_name}" "${ta_name}" "${rac_name}" "${COMPARISON_NAME}"; do
    if ! [[ "${name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
      echo "Run names may contain only letters, numbers, dot, underscore, and dash: ${name}" >&2
      return 1
    fi
  done
  OPD_RUN_NAME="${opd_name}"
  TA_RUN_NAME="${ta_name}"
  RAC_RUN_NAME="${rac_name}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs}"
  OPD_RUN_OUTPUT="${OPD_OUTPUT_DIR:-${OUTPUT_ROOT}/${OPD_RUN_NAME}/opd}"
  TA_RUN_OUTPUT="${TA_OUTPUT_DIR:-${OUTPUT_ROOT}/${TA_RUN_NAME}/ta_opd}"
  RAC_RUN_OUTPUT="${RAC_OUTPUT_DIR:-${OUTPUT_ROOT}/${RAC_RUN_NAME}/rac_opd}"
  RUN_RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results/${COMPARISON_NAME}}"
}

visible_gpu_count() {
  local visible="${CUDA_VISIBLE_DEVICES:-0}"
  local devices=()
  IFS=',' read -r -a devices <<< "${visible}"
  if [[ "${#devices[@]}" -le 0 ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain at least one GPU" >&2
    return 1
  fi
  printf '%s\n' "${#devices[@]}"
}

run_training_cli() {
  local visible_count
  visible_count="$(visible_gpu_count)"
  local processes="${TRAIN_NPROC_PER_NODE:-${visible_count}}"
  if ! [[ "${processes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_NPROC_PER_NODE must be a positive integer, got ${processes}" >&2
    return 1
  fi
  if (( processes > visible_count )); then
    echo "Requested ${processes} workers but only ${visible_count} GPUs are visible" >&2
    return 1
  fi
  if (( processes == 1 )); then
    "${PYTHON_BIN}" -m b200_experiment.cli "$@"
  else
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${processes}" \
      -m b200_experiment.cli "$@"
  fi
}

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
micro = autotune["training"]["micro_batch_size"]
if batch != autotune["rollout"]["batch_size"] or micro <= 0:
    raise SystemExit("Autotune rollout/micro-batch values are inconsistent")
print(f"Autotuned B200 global batch size: {batch} (global micro-batch: {micro})")
PY
  else
    "${PYTHON_BIN}" - "${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-16}}}" "${MICRO_BATCH_SIZE:-1}" <<'PY'
import sys

batch, micro = map(int, sys.argv[1:])
if batch <= 0 or micro <= 0:
    raise SystemExit("GLOBAL_BATCH_SIZE and MICRO_BATCH_SIZE must be positive integers")
print(f"Fixed global training batch size: {batch} (global micro-batch: {micro})")
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
  local prompt_batch="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-16}}}"
  local response_count="${NUM_RESPONSES:-4}"
  local trajectory_batch=$((prompt_batch * response_count))
  COMMON_TRAIN_ARGS=(
    --set "experiment.output_dir=${output_dir}"
    --set "paths.storage_root=${STORAGE_ROOT}"
    --set "experiment.seed=${SEED:-${EXPERIMENT_SEED:-42}}"
    --set "data.max_prompt_tokens=${MAX_PROMPT_LENGTH:-${MAX_PROMPT_LEN:-1024}}"
    --set "training.epochs=${NUM_EPOCHS:-${EPOCHS:-1}}"
    --set "training.learning_rate=${LR:-${LEARNING_RATE:-1.0e-6}}"
    --set "training.save_interval=${SAVE_INTERVAL:-50}"
    --set "training.grad_accum_steps=${GRAD_ACCUM_STEPS:-auto}"
    --set "distributed.bucket_cap_mb=${DDP_BUCKET_CAP_MB:-100}"
    --set "rollout.backend=${ROLLOUT_BACKEND:-vllm}"
    --set "rollout.seed=${ROLLOUT_SEED:-42}"
    --set "rollout.num_responses=${response_count}"
    --set "rollout.max_new_tokens=${MAX_RESPONSE_LENGTH:-${MAX_RESPONSE_LEN:-${MAX_NEW_TOKENS:-7168}}}"
    --set "rollout.temperature=${ROLLOUT_TEMPERATURE:-1.0}"
    --set "rollout.top_p=${ROLLOUT_TOP_P:-1.0}"
    --set "rollout.vllm.gpu_memory_utilization=${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
    --set "rollout.vllm.max_num_seqs=${ROLLOUT_VLLM_MAX_NUM_SEQS:-${trajectory_batch}}"
    --set "rollout.vllm.max_model_len=${ROLLOUT_VLLM_MAX_MODEL_LEN:-9216}"
    --set "rollout.vllm.max_concurrent_requests=${ROLLOUT_VLLM_MAX_CONCURRENT_REQUESTS:-${trajectory_batch}}"
    --set "rollout.vllm.wake_headroom_gib=${ROLLOUT_VLLM_WAKE_HEADROOM_GIB:-2}"
    --set "rollout.vllm.logprob_sanity.enabled=${VLLM_LOGPROB_SANITY_ENABLED:-false}"
    --set "rollout.vllm.logprob_sanity.max_tokens_per_rank=${VLLM_LOGPROB_SANITY_TOKENS:-32}"
    --set "rollout.vllm.logprob_sanity.tolerance=${VLLM_LOGPROB_SANITY_TOLERANCE:-0.05}"
    --set "rollout.vllm.logprob_sanity.fail_on_mismatch=${VLLM_LOGPROB_SANITY_FAIL:-false}"
    --set "opd.teacher_temperature=${TEACHER_TEMPERATURE:-1.0}"
    --set "token_budget.rho=${TA_RHO:-${RHO:-0.10}}"
    --set "selector.top_k=${TOP_K:-16}"
    --set "selector.score_micro_batch_size=${SCORE_MICRO_BATCH_SIZE:-1}"
    --set "selector.score_chunk_steps=${SCORE_CHUNK_STEPS:-128}"
    --set "selector.ta_vocab_chunk_tokens=${TA_VOCAB_CHUNK_TOKENS:-2048}"
    --set "selector.rac_gamma=${RAC_GAMMA:-0.995}"
    --set "selector.rac_w_min=${RAC_W_MIN:-0.10}"
    --set "selector.rac_beta=${RAC_BETA:-2.0}"
    --set "selector.rac_scan_backend=${RAC_SCAN_BACKEND:-parallel}"
    --set "logging.token_score_interval=${TOKEN_SCORE_INTERVAL:-${EVAL_INTERVAL:-50}}"
    --set "logging.log_interval=${LOG_INTERVAL:-1}"
    --set "training_evaluation.enabled=${TRAIN_EVAL_ENABLED:-true}"
    --set "training_evaluation.backend=${TRAIN_EVAL_BACKEND:-vllm}"
    --set "training_evaluation.temperature=${TRAIN_EVAL_TEMPERATURE:-0.7}"
    --set "training_evaluation.top_p=${TRAIN_EVAL_TOP_P:-0.95}"
    --set "training_evaluation.num_responses=${TRAIN_EVAL_NUM_RESPONSES:-16}"
    --set "training_evaluation.target_evaluations=null"
    --set "training_evaluation.interval_steps=${TRAIN_EVAL_INTERVAL:-${EVAL_INTERVAL:-50}}"
    --set "training_evaluation.limit=null"
    --set "training_evaluation.batch_size=${TRAIN_EVAL_BATCH_SIZE:-1}"
    --set "training_evaluation.max_new_tokens=${TRAIN_EVAL_MAX_NEW_TOKENS:-7168}"
    --set "training_evaluation.vllm.tensor_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    --set "training_evaluation.vllm.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-auto}"
    --set "training_evaluation.vllm.gpu_headroom_gib=${VLLM_GPU_HEADROOM_GIB:-4}"
    --set "training_evaluation.vllm.max_num_seqs=${VLLM_MAX_NUM_SEQS:-256}"
    --set "training_evaluation.vllm.max_model_len=${VLLM_MAX_MODEL_LEN:-9216}"
  )
  if batch_autotune_enabled; then
    COMMON_TRAIN_ARGS=(--overlay "${AUTOTUNE_CONFIG}" "${COMMON_TRAIN_ARGS[@]}")
  else
    COMMON_TRAIN_ARGS+=(
      --set "rollout.batch_size=${prompt_batch}"
      --set "training.micro_batch_size=${MICRO_BATCH_SIZE:-1}"
    )
  fi
  if [[ -n "${TRAIN_EVAL_TARGET:-}" ]]; then
    COMMON_TRAIN_ARGS+=(
      --set "training_evaluation.target_evaluations=${TRAIN_EVAL_TARGET}"
      --set "training_evaluation.interval_steps=null"
    )
  fi
  if [[ -n "${MAX_STEPS:-}" && "${MAX_STEPS}" != "-1" ]]; then
    COMMON_TRAIN_ARGS+=(--set "training.max_steps=${MAX_STEPS}")
  fi
  if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    COMMON_TRAIN_ARGS+=(
      --set "training.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}"
      --set "training.resume_allow_config_mismatch=${RESUME_ALLOW_CONFIG_MISMATCH:-false}"
    )
  fi
}
