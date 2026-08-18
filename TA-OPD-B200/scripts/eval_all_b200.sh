#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_b200.sh"

bash "${SCRIPT_DIR}/eval_base_b200.sh"
bash "${SCRIPT_DIR}/eval_ta_b200.sh"
bash "${SCRIPT_DIR}/eval_rac_b200.sh"
"${PYTHON_BIN}" -m b200_experiment.cli aggregate-eval \
  --base-dir "${REPO_DIR}/results/eval/base" \
  --ta-dir "${REPO_DIR}/results/eval/ta_opd" \
  --rac-dir "${REPO_DIR}/results/eval/rac" \
  --output "${REPO_DIR}/results"
bash "${SCRIPT_DIR}/plot_results.sh"
