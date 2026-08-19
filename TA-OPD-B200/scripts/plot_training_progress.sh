#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

resolve_run_paths
require_file "${TA_RUN_OUTPUT}/metrics.jsonl"
require_file "${RAC_RUN_OUTPUT}/metrics.jsonl"
require_file "${TA_RUN_OUTPUT}/eval_history.jsonl"
require_file "${RAC_RUN_OUTPUT}/eval_history.jsonl"
echo "Plotting TA run ${TA_RUN_NAME} against RAC run ${RAC_RUN_NAME}"
echo "Comparison name: ${COMPARISON_NAME}"
echo "Results directory: ${RUN_RESULTS_DIR}"
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli plot-training-progress \
  --results "${RUN_RESULTS_DIR}" \
  --ta-output "${TA_RUN_OUTPUT}" \
  --rac-output "${RAC_RUN_OUTPUT}" \
  --smoothing-window "${SMOOTHING_WINDOW:-10}" "$@"
