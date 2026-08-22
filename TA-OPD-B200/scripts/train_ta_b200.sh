#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= TA-OPD PARAMETERS: EDIT HERE =================
# A fresh invocation gets a local-time run name automatically. You can still
# pass RUN_NAME=... explicitly, especially when resuming an existing run.
export RUN_NAME="${RUN_NAME:-ta_qwen3_4b_to_1p7b_$(date +%Y%m%d_%H%M%S)}"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SEED="${SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export GLOBAL_BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-16}}}"
export BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
export NUM_RESPONSES="${NUM_RESPONSES:-4}"
export MICRO_BATCH_SIZE="${MICRO:-${MICRO_BATCH_SIZE:-1}}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-auto}"
export NUM_EPOCHS="${NUM_EPOCHS:-${EPOCHS:-1}}"
export MAX_STEPS="${MAX_STEPS:--1}"               # -1 = consume all configured epochs
export LR="${LR:-${LEARNING_RATE:-1.0e-6}}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LENGTH:-${MAX_PROMPT_LEN:-1024}}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LENGTH:-${MAX_RESPONSE_LEN:-${MAX_NEW_TOKENS:-7168}}}"
export TOP_K="${TOP_K:-16}"
export SCORE_CHUNK_STEPS="${SCORE_CHUNK_STEPS:-128}"
export TA_VOCAB_CHUNK_TOKENS="${TA_VOCAB_CHUNK_TOKENS:-2048}"
export TA_RHO="${TA_RHO:-${RHO:-0.10}}"
# Kept visible for exact config parity; TA does not consume Bellman weights.
export RAC_GAMMA="${RAC_GAMMA:-0.995}"
export RAC_W_MIN="${RAC_W_MIN:-0.10}"
export RAC_BETA="${RAC_BETA:-2.0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"

# Keep enabled when accuracy-over-step plots are needed after training.
export TRAIN_EVAL_ENABLED="${TRAIN_EVAL_ENABLED:-true}"
export TRAIN_EVAL_BACKEND="${TRAIN_EVAL_BACKEND:-vllm}"
export TRAIN_EVAL_TEMPERATURE="${TRAIN_EVAL_TEMPERATURE:-0.7}"
export TRAIN_EVAL_TOP_P="${TRAIN_EVAL_TOP_P:-0.95}"
export TRAIN_EVAL_NUM_RESPONSES="${TRAIN_EVAL_NUM_RESPONSES:-16}"
export TRAIN_EVAL_MAX_NEW_TOKENS="${TRAIN_EVAL_MAX_NEW_TOKENS:-7168}"

# Engineering/DDP defaults. GLOBAL_BATCH_SIZE is independent of GPU count.
export DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-100}"
export USE_BATCH_AUTOTUNE="${USE_BATCH_AUTOTUNE:-false}"
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
# 1024 prompt tokens + 7168 generated tokens fit within this context.
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-9216}"
export ROLLOUT_VLLM_WAKE_HEADROOM_GIB="${ROLLOUT_VLLM_WAKE_HEADROOM_GIB:-2}"

# Fresh train: leave empty. Resume: normally pass this on the launch command.
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${RESUME:-}}"
# ================================================================

source "${SCRIPT_DIR}/common_b200.sh"

require_b200_validation
resolve_run_paths
if [[ -n "${RESUME_FROM_CHECKPOINT}" && "${RESUME_FROM_CHECKPOINT}" != "auto" && -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$(dirname -- "${RESUME_FROM_CHECKPOINT}")"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${TA_RUN_OUTPUT}}"
fi
build_training_args "${OUTPUT_DIR}"
echo "TA-OPD run: ${RUN_NAME}"
echo "TA-OPD output: ${OUTPUT_DIR}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
else
  echo "Training mode: fresh"
fi
cd "${REPO_DIR}"
run_training_cli train \
  --config "${TA_CONFIG}" \
  "${COMMON_TRAIN_ARGS[@]}" "$@"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "TA-OPD resume completed in: ${OUTPUT_DIR}"
else
  echo "TA-OPD completed. Keep this identifier: TA_RUN_NAME=${RUN_NAME}"
fi
