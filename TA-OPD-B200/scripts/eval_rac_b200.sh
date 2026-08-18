#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

MODEL_PATH="${RAC_CHECKPOINT:-${REPO_DIR}/outputs/rac_opd/final}"
require_file "${MODEL_PATH}/config.json"
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli evaluate \
  --config "${RAC_CONFIG}" --name "RAC" --model "${MODEL_PATH}" \
  --output "${REPO_DIR}/results/eval/rac" \
  --set "evaluation.backend=${EVAL_BACKEND:-vllm}" \
  --set "evaluation.batch_size=${EVAL_BATCH_SIZE:-16}" \
  --set "evaluation.max_new_tokens=${EVAL_MAX_NEW_TOKENS:-2048}" \
  --set "evaluation.vllm.tensor_parallel_size=${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --set "evaluation.vllm.gpu_memory_utilization=${EVAL_VLLM_GPU_MEMORY_UTILIZATION:-auto}" \
  --set "evaluation.vllm.gpu_headroom_gib=${EVAL_VLLM_GPU_HEADROOM_GIB:-4}" \
  --set "evaluation.vllm.max_num_seqs=${EVAL_VLLM_MAX_NUM_SEQS:-256}" \
  --set "evaluation.vllm.max_model_len=${EVAL_VLLM_MAX_MODEL_LEN:-4096}" \
  --set "evaluation.limit=null" "$@"
