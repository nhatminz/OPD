#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

MODEL_PATH="${BASE_CHECKPOINT:-/workspace/storage-shared/models/Qwen3-1.7B}"
require_file "${MODEL_PATH}/config.json"
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli evaluate \
  --config "${BASE_CONFIG}" --name "Base" --model "${MODEL_PATH}" \
  --output "${REPO_DIR}/results/eval/base" \
  --set "evaluation.batch_size=${EVAL_BATCH_SIZE:-16}" \
  --set "evaluation.max_new_tokens=${EVAL_MAX_NEW_TOKENS:-2048}" \
  --set "evaluation.limit=${EVAL_LIMIT:-null}" "$@"
