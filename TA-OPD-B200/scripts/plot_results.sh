#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

resolve_run_paths
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli plot \
  --results "${RUN_RESULTS_DIR}" \
  --ta-output "${TA_RUN_OUTPUT}" \
  --rac-output "${RAC_RUN_OUTPUT}" \
  --smoothing-window "${SMOOTHING_WINDOW:-10}" "$@"
