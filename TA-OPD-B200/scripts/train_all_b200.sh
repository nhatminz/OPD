#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_b200.sh"

TA_RUN_OUTPUT="${TA_OUTPUT_DIR:-${REPO_DIR}/outputs/ta_opd}"
RAC_RUN_OUTPUT="${RAC_OUTPUT_DIR:-${REPO_DIR}/outputs/rac_opd}"
OUTPUT_DIR="${TA_RUN_OUTPUT}" bash "${SCRIPT_DIR}/train_ta_b200.sh"
OUTPUT_DIR="${RAC_RUN_OUTPUT}" bash "${SCRIPT_DIR}/train_rac_b200.sh"
TA_OUTPUT_DIR="${TA_RUN_OUTPUT}" RAC_OUTPUT_DIR="${RAC_RUN_OUTPUT}" \
  bash "${SCRIPT_DIR}/plot_training_progress.sh"
