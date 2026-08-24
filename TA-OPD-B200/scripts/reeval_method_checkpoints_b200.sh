#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/reeval_method_checkpoints_b200.sh METHOD [RUN_NAME]

METHOD may be: opd, ta-opd (or ta), rac.

RUN_NAME is optional when the corresponding OPD_RUN_NAME, TA_RUN_NAME, or
RAC_RUN_NAME environment variable is already set. To select an output directory
directly, omit RUN_NAME and set OPD_OUTPUT_DIR, TA_OUTPUT_DIR, or RAC_OUTPUT_DIR.

Examples:
  bash scripts/reeval_method_checkpoints_b200.sh opd my_opd_run
  REEVAL_DRY_RUN=true bash scripts/reeval_method_checkpoints_b200.sh ta my_ta_run
  RAC_OUTPUT_DIR=/absolute/path/to/rac_opd bash scripts/reeval_method_checkpoints_b200.sh rac
EOF
}

if (( $# < 1 || $# > 2 )); then
  usage
  exit 2
fi

METHOD_INPUT="$1"
RUN_NAME_INPUT="${2:-}"
case "${METHOD_INPUT,,}" in
  opd|pure-opd|pure_opd)
    METHOD="opd"
    if [[ -n "${RUN_NAME_INPUT}" ]]; then export OPD_RUN_NAME="${RUN_NAME_INPUT}"; fi
    ;;
  ta|ta-opd|ta_opd)
    METHOD="ta"
    if [[ -n "${RUN_NAME_INPUT}" ]]; then export TA_RUN_NAME="${RUN_NAME_INPUT}"; fi
    ;;
  rac|bellman-rac|bellman_rac)
    METHOD="rac"
    if [[ -n "${RUN_NAME_INPUT}" ]]; then export RAC_RUN_NAME="${RUN_NAME_INPUT}"; fi
    ;;
  *)
    echo "Unknown method: ${METHOD_INPUT}" >&2
    usage
    exit 2
    ;;
esac

export REEVAL_METHODS="${METHOD}"
exec bash "${SCRIPT_DIR}/reeval_all_checkpoints_b200.sh"
