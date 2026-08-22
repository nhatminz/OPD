#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

cd "${REPO_DIR}"
"${PYTHON_BIN}" -m unittest discover -s tests -v
"${PYTHON_BIN}" -m b200_experiment.cli preflight \
  --config "${BASE_CONFIG}" \
  --output "${PREFLIGHT_REPORT}"

if batch_autotune_enabled; then
  TUNE_ARGS=(
    --opd-config "${OPD_CONFIG}"
    --ta-config "${TA_CONFIG}"
    --rac-config "${RAC_CONFIG}"
    --output "${REPO_DIR}/outputs/autotune"
    --generated-config "${AUTOTUNE_CONFIG}"
  )
  if [[ -n "${BATCH_CANDIDATES:-}" ]]; then
    read -r -a CANDIDATES <<< "${BATCH_CANDIDATES}"
    TUNE_ARGS+=(--candidates "${CANDIDATES[@]}")
  fi
  "${PYTHON_BIN}" -m b200_experiment.cli autotune-batch "${TUNE_ARGS[@]}"
else
  echo "Skipping batch search; fixed prompt BATCH_SIZE=${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-16}}}, NUM_RESPONSES=${NUM_RESPONSES:-4}."
fi
require_b200_validation

echo "Smoke/preflight passed. Full training was not started."
