#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/eval_checkpoint_b200.sh METHOD CHECKPOINT [OUTPUT_DIR] [extra CLI args]

METHOD may be: opd, ta-opd (or ta), rac.

Examples:
  bash scripts/eval_checkpoint_b200.sh opd outputs/run01/opd/checkpoint-000050
  bash scripts/eval_checkpoint_b200.sh ta-opd /path/checkpoint-000100 results/eval/ta_step100
  EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 bash scripts/eval_checkpoint_b200.sh rac /path/checkpoint-000150
EOF
}

if (( $# < 2 )); then
  usage
  exit 2
fi

METHOD_INPUT="$1"
CHECKPOINT_INPUT="$2"
shift 2

case "${METHOD_INPUT,,}" in
  opd|pure-opd|pure_opd)
    METHOD_SLUG="opd"
    MODEL_NAME="OPD"
    METHOD_CONFIG="${OPD_CONFIG}"
    ;;
  ta|ta-opd|ta_opd)
    METHOD_SLUG="ta_opd"
    MODEL_NAME="TA-OPD"
    METHOD_CONFIG="${TA_CONFIG}"
    ;;
  rac|bellman-rac|bellman_rac)
    METHOD_SLUG="rac"
    MODEL_NAME="RAC"
    METHOD_CONFIG="${RAC_CONFIG}"
    ;;
  *)
    echo "Unknown method: ${METHOD_INPUT}" >&2
    usage
    exit 2
    ;;
esac

if [[ ! -d "${CHECKPOINT_INPUT}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_INPUT}" >&2
  exit 1
fi
CHECKPOINT_PATH="$(cd "${CHECKPOINT_INPUT}" && pwd)"
require_file "${CHECKPOINT_PATH}/config.json"
require_file "${METHOD_CONFIG}"

if (( $# > 0 )) && [[ "$1" != --* ]]; then
  EVAL_OUTPUT="$1"
  shift
else
  CHECKPOINT_NAME="$(basename "${CHECKPOINT_PATH}")"
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  EVAL_OUTPUT="${REPO_DIR}/results/checkpoint_eval/${METHOD_SLUG}_${CHECKPOINT_NAME}_${TIMESTAMP}"
fi

if [[ "${EVAL_OUTPUT}" != /* ]]; then
  EVAL_OUTPUT="${REPO_DIR}/${EVAL_OUTPUT}"
fi

echo "Method: ${MODEL_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Evaluation output: ${EVAL_OUTPUT}"
echo "Protocol: backend=${EVAL_BACKEND:-vllm}, temperature=${EVAL_TEMPERATURE:-1.0}, responses=${EVAL_NUM_RESPONSES:-16}"

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli evaluate \
  --config "${METHOD_CONFIG}" \
  "${ASSET_CONFIG_ARGS[@]}" \
  --name "${MODEL_NAME}" \
  --model "${CHECKPOINT_PATH}" \
  --output "${EVAL_OUTPUT}" \
  --set "paths.storage_root=${STORAGE_ROOT}" \
  --set "evaluation.backend=${EVAL_BACKEND:-vllm}" \
  --set "evaluation.temperature=${EVAL_TEMPERATURE:-1.0}" \
  --set "evaluation.top_p=${EVAL_TOP_P:-0.95}" \
  --set "evaluation.num_responses=${EVAL_NUM_RESPONSES:-16}" \
  --set "evaluation.batch_size=${EVAL_BATCH_SIZE:-1}" \
  --set "evaluation.max_new_tokens=${EVAL_MAX_NEW_TOKENS:-7168}" \
  --set "evaluation.vllm.tensor_parallel_size=${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --set "evaluation.vllm.gpu_memory_utilization=${EVAL_VLLM_GPU_MEMORY_UTILIZATION:-auto}" \
  --set "evaluation.vllm.gpu_headroom_gib=${EVAL_VLLM_GPU_HEADROOM_GIB:-4}" \
  --set "evaluation.vllm.max_num_seqs=${EVAL_VLLM_MAX_NUM_SEQS:-256}" \
  --set "evaluation.vllm.max_model_len=${EVAL_VLLM_MAX_MODEL_LEN:-9216}" \
  --set "evaluation.vllm.enable_chunked_prefill=${EVAL_VLLM_ENABLE_CHUNKED_PREFILL:-true}" \
  --set "evaluation.vllm.performance_mode=${EVAL_VLLM_PERFORMANCE_MODE:-throughput}" \
  --set "evaluation.vllm.async_scheduling=${EVAL_VLLM_ASYNC_SCHEDULING:-true}" \
  --set "evaluation.limit=${EVAL_LIMIT:-null}" \
  "$@"
