#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

resolve_run_paths
PLOT_METHOD="${PLOT_METHOD:-both}"
NORMALIZED_PLOT_METHOD="${PLOT_METHOD,,}"
PLOT_ARGS=(--method "${NORMALIZED_PLOT_METHOD}")
case "${NORMALIZED_PLOT_METHOD}" in
  both)
    require_file "${TA_RUN_OUTPUT}/metrics.jsonl"
    require_file "${RAC_RUN_OUTPUT}/metrics.jsonl"
    require_file "${TA_RUN_OUTPUT}/eval_history.jsonl"
    require_file "${RAC_RUN_OUTPUT}/eval_history.jsonl"
    PLOT_ARGS+=(--ta-output "${TA_RUN_OUTPUT}" --rac-output "${RAC_RUN_OUTPUT}")
    echo "Plotting TA run ${TA_RUN_NAME} against RAC run ${RAC_RUN_NAME}"
    echo "Comparison name: ${COMPARISON_NAME}"
    ;;
  ta|ta-opd)
    require_file "${TA_RUN_OUTPUT}/eval_history.jsonl"
    RUN_RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results/${TA_RUN_NAME}}"
    PLOT_ARGS+=(--ta-output "${TA_RUN_OUTPUT}")
    echo "Plotting TA-OPD run ${TA_RUN_NAME} on all evaluation datasets"
    ;;
  rac|bellman-rac)
    require_file "${RAC_RUN_OUTPUT}/eval_history.jsonl"
    RUN_RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results/${RAC_RUN_NAME}}"
    PLOT_ARGS+=(--rac-output "${RAC_RUN_OUTPUT}")
    echo "Plotting Bellman-RAC run ${RAC_RUN_NAME} on all evaluation datasets"
    ;;
  *)
    echo "PLOT_METHOD must be one of: both, ta, ta-opd, rac, bellman-rac" >&2
    exit 2
    ;;
esac
echo "Results directory: ${RUN_RESULTS_DIR}"
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.cli plot-training-progress \
  --results "${RUN_RESULTS_DIR}" \
  "${PLOT_ARGS[@]}" \
  --smoothing-window "${SMOOTHING_WINDOW:-10}" "$@"
