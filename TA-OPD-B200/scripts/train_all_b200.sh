#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= BASIC PARAMETERS: EDIT HERE =================
# Environment values supplied when launching the script still take precedence.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export TA_RUN_NAME="${TA_RUN_NAME:-ta_qwen3_4b_to_1p7b_${RUN_TIMESTAMP}}"
export RAC_RUN_NAME="${RAC_RUN_NAME:-rac_bellman_qwen3_4b_to_1p7b_${RUN_TIMESTAMP}}"
export RUN_NAME="${RUN_NAME:-comparison_${RUN_TIMESTAMP}}"
# Shared GLOBAL rollout/micro-batch for both methods. Changing the visible GPU
# count changes only the per-GPU shard; it does not change optimizer semantics.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-8}}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-100}"
# Set true only when you explicitly want smoke_test_b200.sh to search candidates.
export USE_BATCH_AUTOTUNE="${USE_BATCH_AUTOTUNE:-false}"
export NUM_EPOCHS="${NUM_EPOCHS:-${EPOCHS:-1}}"
export LR="${LR:-${LEARNING_RATE:-1.0e-6}}"
export TA_RHO="${TA_RHO:-${RHO:-0.10}}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-${MAX_NEW_TOKENS:-8192}}"
# Training rollout backend. vllm syncs the live student before every batch;
# use hf only as a compatibility/debug fallback.
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"  # vllm or hf
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
export ROLLOUT_VLLM_MAX_NUM_SEQS="${ROLLOUT_VLLM_MAX_NUM_SEQS:-${GLOBAL_BATCH_SIZE}}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-12288}"
export ROLLOUT_VLLM_MAX_CONCURRENT_REQUESTS="${ROLLOUT_VLLM_MAX_CONCURRENT_REQUESTS:-${GLOBAL_BATCH_SIZE}}"
export ROLLOUT_VLLM_WAKE_HEADROOM_GIB="${ROLLOUT_VLLM_WAKE_HEADROOM_GIB:-2}"
export TOP_K="${TOP_K:-16}"
export RAC_GAMMA="${RAC_GAMMA:-0.995}"
export RAC_W_MIN="${RAC_W_MIN:-0.10}"
export RAC_BETA="${RAC_BETA:-2.0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
# Optional hard cap; leave empty to train the configured number of epochs.
export MAX_STEPS="${MAX_STEPS:-}"

# Exactly about 16 evenly-spaced evaluations, including step 0 and the last step.
export TRAIN_EVAL_ENABLED="${TRAIN_EVAL_ENABLED:-true}"
export TRAIN_EVAL_BACKEND="${TRAIN_EVAL_BACKEND:-vllm}"  # vllm or hf
export TRAIN_EVAL_MAX_NEW_TOKENS="${TRAIN_EVAL_MAX_NEW_TOKENS:-8192}"
export TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-16}"  # hf backend only

# Periodic/final evaluation vLLM settings (separate from training rollout).
# For tensor parallelism, expose the same number of GPUs above.
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-auto}"
export VLLM_GPU_HEADROOM_GIB="${VLLM_GPU_HEADROOM_GIB:-4}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
# To use a fixed interval instead, uncomment: export TRAIN_EVAL_INTERVAL=100
# ================================================================

source "${SCRIPT_DIR}/common_b200.sh"

resolve_run_paths
RUN_NAME="${TA_RUN_NAME}" OUTPUT_DIR="${TA_RUN_OUTPUT}" \
  RESUME_FROM_CHECKPOINT="${TA_RESUME_FROM_CHECKPOINT:-}" \
  bash "${SCRIPT_DIR}/train_ta_b200.sh"
RUN_NAME="${RAC_RUN_NAME}" OUTPUT_DIR="${RAC_RUN_OUTPUT}" \
  RESUME_FROM_CHECKPOINT="${RAC_RESUME_FROM_CHECKPOINT:-}" \
  bash "${SCRIPT_DIR}/train_rac_b200.sh"
echo "Both runs finished. Plot explicitly with:"
echo "RUN_NAME=${RUN_NAME} bash scripts/plot_training_progress.sh"
