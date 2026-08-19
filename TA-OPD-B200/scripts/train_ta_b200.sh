#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= TA-OPD PARAMETERS: EDIT HERE =================
# A fresh invocation gets a local-time run name automatically. You can still
# pass RUN_NAME=... explicitly, especially when resuming an existing run.
export RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EXPERIMENT_SEED="${EXPERIMENT_SEED:-1234}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"   # global, independent of GPU count
export EPOCHS="${EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-}"                 # total target; empty = infer from epochs
export LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
export RHO="${RHO:-0.10}"
export TOP_K="${TOP_K:-16}"
# Kept shared with RAC for config parity/resume validation; TA does not probe branches.
export BRANCH_M="${BRANCH_M:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"

# Keep enabled when accuracy_over_steps.png is needed after both runs finish.
export TRAIN_EVAL_ENABLED="${TRAIN_EVAL_ENABLED:-true}"
export TRAIN_EVAL_TARGET="${TRAIN_EVAL_TARGET:-16}"
export TRAIN_EVAL_BACKEND="${TRAIN_EVAL_BACKEND:-vllm}"
export TRAIN_EVAL_MAX_NEW_TOKENS="${TRAIN_EVAL_MAX_NEW_TOKENS:-2048}"

# Engineering/DDP defaults. TRAIN_BATCH_SIZE remains the global batch.
export DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-100}"
export USE_BATCH_AUTOTUNE="${USE_BATCH_AUTOTUNE:-false}"
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
# 512 prompt tokens + 2048 generated tokens fit within this context.
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-4096}"
export ROLLOUT_VLLM_WAKE_HEADROOM_GIB="${ROLLOUT_VLLM_WAKE_HEADROOM_GIB:-2}"

# Fresh train: leave empty. Resume: normally pass this on the launch command.
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
# ================================================================

source "${SCRIPT_DIR}/common_b200.sh"

require_b200_validation
resolve_run_paths
if [[ -n "${RESUME_FROM_CHECKPOINT}" && -z "${OUTPUT_DIR:-}" ]]; then
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
