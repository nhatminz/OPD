#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

require_b200_validation
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/ta_opd}"
build_training_args "${OUTPUT_DIR}"
cd "${REPO_DIR}"
run_training_cli train \
  --config "${TA_CONFIG}" \
  "${COMMON_TRAIN_ARGS[@]}" "$@"
plot_training_progress_if_ready \
  "${OUTPUT_DIR}" "${RAC_OUTPUT_DIR:-${REPO_DIR}/outputs/rac_opd}"
