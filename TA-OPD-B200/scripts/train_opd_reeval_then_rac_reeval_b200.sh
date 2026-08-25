#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ================= USER CONFIG ================================
# Model/data selection shared exactly by OPD and RAC.
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export TEACHER_MODEL="${TEACHER_MODEL:-models/Qwen3-8B}"
export STUDENT_MODEL="${STUDENT_MODEL:-nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base}"
export TRAIN_DATASET="competition_math"
export TRAIN_DATA="${TRAIN_DATA:-nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet}"
export TRAIN_DATA_SPLIT="null"
export PROMPT_KEY="problem"

# Training batch/length settings. PPO mini-batch and per-GPU micro-batch are
# deliberately separate. Override MICRO_BATCH_SIZE_PER_GPU only for memory.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
export NUM_RESPONSES="${NUM_RESPONSES:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
export MAX_RESPONSE_LEN="${MAX_NEW_TOKENS}"
export MAX_RESPONSE_LENGTH="${MAX_NEW_TOKENS}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-9216}"
export OVERLONG_PROMPT_POLICY="${OVERLONG_PROMPT_POLICY:-filter}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export LR="${LR:-1.0e-6}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

# Training never runs periodic evaluation in this workflow. Evaluation starts
# only after each method has finished and covers every saved checkpoint.
export TRAIN_EVAL_ENABLED=false
export REEVAL_TEMPERATURE="${REEVAL_TEMPERATURE:-0.7}"
export REEVAL_TOP_P="${REEVAL_TOP_P:-0.95}"
export REEVAL_NUM_RESPONSES="${REEVAL_NUM_RESPONSES:-16}"
export REEVAL_MAX_NEW_TOKENS="${REEVAL_MAX_NEW_TOKENS:-8192}"
export REEVAL_VLLM_MAX_MODEL_LEN="${REEVAL_VLLM_MAX_MODEL_LEN:-9216}"
export REEVAL_SKIP_BASE="${REEVAL_SKIP_BASE:-false}"

# Two training GPUs by default; both values remain overridable at launch.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-2}"

SEQUENCE_TIMESTAMP="$(date +%Y%m%d_%H%M%S_%N)"
export OPD_RUN_NAME="${OPD_RUN_NAME:-opd_qwen3_8b_to_1p7b_base_compmath_${SEQUENCE_TIMESTAMP}}"
export RAC_RUN_NAME="${RAC_RUN_NAME:-rac_qwen3_8b_to_1p7b_base_compmath_${SEQUENCE_TIMESTAMP}}"
# ==============================================================

cd "${REPO_DIR}"

echo "[1/4] Training pure OPD without periodic evaluation: ${OPD_RUN_NAME}"
RUN_NAME="${OPD_RUN_NAME}" \
  TRAIN_EVAL_ENABLED=false \
  bash "${SCRIPT_DIR}/train_opd_b200.sh"

echo "[2/4] Re-evaluating every saved OPD checkpoint: ${OPD_RUN_NAME}"
bash "${SCRIPT_DIR}/reeval_method_checkpoints_b200.sh" opd "${OPD_RUN_NAME}"

echo "[3/4] Training RAC without periodic evaluation: ${RAC_RUN_NAME}"
RUN_NAME="${RAC_RUN_NAME}" \
  TRAIN_EVAL_ENABLED=false \
  bash "${SCRIPT_DIR}/train_rac_b200.sh"

echo "[4/4] Re-evaluating every saved RAC checkpoint: ${RAC_RUN_NAME}"
bash "${SCRIPT_DIR}/reeval_method_checkpoints_b200.sh" rac "${RAC_RUN_NAME}"

echo "Sequential OPD/RAC training and checkpoint re-evaluation completed."
echo "OPD_RUN_NAME=${OPD_RUN_NAME}"
echo "RAC_RUN_NAME=${RAC_RUN_NAME}"
