#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_b200.sh"

resolve_run_paths
BASE_EVAL_OUTPUT="${BASE_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/base}"
TA_EVAL_OUTPUT="${TA_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/ta_opd}"
RAC_EVAL_OUTPUT="${RAC_EVAL_OUTPUT:-${RUN_RESULTS_DIR}/eval/rac}"
TA_CHECKPOINT="${TA_CHECKPOINT:-${TA_RUN_OUTPUT}/final}"
RAC_CHECKPOINT="${RAC_CHECKPOINT:-${RAC_RUN_OUTPUT}/final}"

echo "Evaluating TA run ${TA_RUN_NAME} against RAC run ${RAC_RUN_NAME}"
echo "Comparison name: ${COMPARISON_NAME}"
echo "TA checkpoint: ${TA_CHECKPOINT}"
echo "RAC checkpoint: ${RAC_CHECKPOINT}"
echo "Evaluation results: ${RUN_RESULTS_DIR}"
RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
  BASE_EVAL_OUTPUT="${BASE_EVAL_OUTPUT}" \
  bash "${SCRIPT_DIR}/eval_base_b200.sh"
RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
  TA_OUTPUT_DIR="${TA_RUN_OUTPUT}" TA_CHECKPOINT="${TA_CHECKPOINT}" \
  TA_EVAL_OUTPUT="${TA_EVAL_OUTPUT}" \
  bash "${SCRIPT_DIR}/eval_ta_b200.sh"
RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
  RAC_OUTPUT_DIR="${RAC_RUN_OUTPUT}" RAC_CHECKPOINT="${RAC_CHECKPOINT}" \
  RAC_EVAL_OUTPUT="${RAC_EVAL_OUTPUT}" \
  bash "${SCRIPT_DIR}/eval_rac_b200.sh"
"${PYTHON_BIN}" -m b200_experiment.cli aggregate-eval \
  --base-dir "${BASE_EVAL_OUTPUT}" \
  --ta-dir "${TA_EVAL_OUTPUT}" \
  --rac-dir "${RAC_EVAL_OUTPUT}" \
  --output "${RUN_RESULTS_DIR}"
RUN_NAME="${RUN_NAME}" RESULTS_DIR="${RUN_RESULTS_DIR}" \
  TA_OUTPUT_DIR="${TA_RUN_OUTPUT}" RAC_OUTPUT_DIR="${RAC_RUN_OUTPUT}" \
  bash "${SCRIPT_DIR}/plot_results.sh"
