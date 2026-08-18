#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli plot-training-progress \
  --results "${RESULTS_DIR:-${REPO_DIR}/results}" \
  --ta-output "${TA_OUTPUT_DIR:-${REPO_DIR}/outputs/ta_opd}" \
  --rac-output "${RAC_OUTPUT_DIR:-${REPO_DIR}/outputs/rac_opd}" \
  --smoothing-window "${SMOOTHING_WINDOW:-10}" "$@"
