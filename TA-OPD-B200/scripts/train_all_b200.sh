#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= BASIC PARAMETERS: EDIT HERE =================
# Environment values supplied when launching the script still take precedence.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Shared rollout/micro-batch for both TA and RAC (gradient accumulation = 1).
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
# Set true only when you explicitly want smoke_test_b200.sh to search candidates.
export USE_BATCH_AUTOTUNE="${USE_BATCH_AUTOTUNE:-false}"
export EPOCHS="${EPOCHS:-1}"
export LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
export RHO="${RHO:-0.10}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
# Training rollout backend. vllm syncs the live student before every batch;
# use hf only as a compatibility/debug fallback.
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"  # vllm or hf
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
export ROLLOUT_VLLM_MAX_NUM_SEQS="${ROLLOUT_VLLM_MAX_NUM_SEQS:-${TRAIN_BATCH_SIZE}}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-1024}"
export ROLLOUT_VLLM_MAX_CONCURRENT_REQUESTS="${ROLLOUT_VLLM_MAX_CONCURRENT_REQUESTS:-${TRAIN_BATCH_SIZE}}"
export ROLLOUT_VLLM_WAKE_HEADROOM_GIB="${ROLLOUT_VLLM_WAKE_HEADROOM_GIB:-2}"
export TOP_K="${TOP_K:-16}"
export BRANCH_M="${BRANCH_M:-2}"
export RAC_BRANCH_CHUNK_SIZE="${RAC_BRANCH_CHUNK_SIZE:-256}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
# Optional hard cap; leave empty to train the configured number of epochs.
export MAX_STEPS="${MAX_STEPS:-}"

# Exactly about 16 evenly-spaced evaluations, including step 0 and the last step.
export TRAIN_EVAL_ENABLED="${TRAIN_EVAL_ENABLED:-true}"
export TRAIN_EVAL_TARGET="${TRAIN_EVAL_TARGET:-16}"
export TRAIN_EVAL_BACKEND="${TRAIN_EVAL_BACKEND:-vllm}"  # vllm or hf
export TRAIN_EVAL_MAX_NEW_TOKENS="${TRAIN_EVAL_MAX_NEW_TOKENS:-2048}"
export TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-16}"  # hf backend only

# Periodic/final evaluation vLLM settings (separate from training rollout).
# For tensor parallelism, expose the same number of GPUs above.
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-auto}"
export VLLM_GPU_HEADROOM_GIB="${VLLM_GPU_HEADROOM_GIB:-4}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
# To use a fixed interval instead, uncomment: export TRAIN_EVAL_INTERVAL=100
# ================================================================

source "${SCRIPT_DIR}/common_b200.sh"

TA_RUN_OUTPUT="${TA_OUTPUT_DIR:-${REPO_DIR}/outputs/ta_opd}"
RAC_RUN_OUTPUT="${RAC_OUTPUT_DIR:-${REPO_DIR}/outputs/rac_opd}"
OUTPUT_DIR="${TA_RUN_OUTPUT}" bash "${SCRIPT_DIR}/train_ta_b200.sh"
OUTPUT_DIR="${RAC_RUN_OUTPUT}" bash "${SCRIPT_DIR}/train_rac_b200.sh"
TA_OUTPUT_DIR="${TA_RUN_OUTPUT}" RAC_OUTPUT_DIR="${RAC_RUN_OUTPUT}" \
  bash "${SCRIPT_DIR}/plot_training_progress.sh"
