#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

cd "${REPO_DIR}"
"${PYTHON_BIN}" -m unittest discover -s tests -v
"${PYTHON_BIN}" -m b200_experiment.cli preflight \
  --config "${BASE_CONFIG}" \
  --output "${PREFLIGHT_REPORT}"

TUNE_ARGS=(
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
require_b200_validation

echo "Smoke/autotune passed. Full training was not started."
