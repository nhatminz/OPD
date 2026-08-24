#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

resolve_run_paths
MODEL_PATH="${TA_CHECKPOINT:-${TA_RUN_OUTPUT}/final}"
EVAL_OUTPUT="${TA_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/ta_opd}"
require_file "${MODEL_PATH}/config.json"
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli evaluate \
  --config "${TA_CONFIG}" --name "TA-OPD" --model "${MODEL_PATH}" \
  "${ASSET_CONFIG_ARGS[@]}" \
  --output "${EVAL_OUTPUT}" \
  --set "paths.storage_root=${STORAGE_ROOT}" \
  --set "evaluation.backend=${EVAL_BACKEND:-vllm}" \
  --set "evaluation.temperature=${EVAL_TEMPERATURE:-0.7}" \
  --set "evaluation.top_p=${EVAL_TOP_P:-0.95}" \
  --set "evaluation.num_responses=${EVAL_NUM_RESPONSES:-16}" \
  --set "evaluation.batch_size=${EVAL_BATCH_SIZE:-1}" \
  --set "evaluation.max_new_tokens=${EVAL_MAX_NEW_TOKENS:-7168}" \
  --set "evaluation.vllm.tensor_parallel_size=${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --set "evaluation.vllm.gpu_memory_utilization=${EVAL_VLLM_GPU_MEMORY_UTILIZATION:-auto}" \
  --set "evaluation.vllm.gpu_headroom_gib=${EVAL_VLLM_GPU_HEADROOM_GIB:-4}" \
  --set "evaluation.vllm.max_num_seqs=${EVAL_VLLM_MAX_NUM_SEQS:-256}" \
  --set "evaluation.vllm.max_model_len=${EVAL_VLLM_MAX_MODEL_LEN:-9216}" \
  --set "evaluation.limit=null" "$@"
