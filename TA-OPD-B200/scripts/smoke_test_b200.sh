#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

cd "${REPO_DIR}"
case "${SMOKE_RUN_GPU_UNIT_TESTS:-false}" in
  true|TRUE|1|yes|YES)
    "${PYTHON_BIN}" -m unittest discover -s tests -v
    ;;
  *)
    # Generic preflight must not reserve/spawn extra GPUs from a live workload.
    # The explicit smoke_test_fsdp_multigpu.sh performs the real distributed test.
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -m unittest discover -s tests -v
    ;;
esac
"${PYTHON_BIN}" -m b200_experiment.cli preflight \
  --config "${BASE_CONFIG}" \
  "${ASSET_CONFIG_ARGS[@]}" \
  --output "${PREFLIGHT_REPORT}"

if batch_autotune_enabled; then
  TUNE_ARGS=(
    --opd-config "${OPD_CONFIG}"
    --ta-config "${TA_CONFIG}"
    --rac-config "${RAC_CONFIG}"
    --output "${AUTOTUNE_OUTPUT_ROOT}"
    --generated-config "${AUTOTUNE_CONFIG}"
    "${ASSET_CONFIG_ARGS[@]}"
  )
  if [[ -n "${BATCH_CANDIDATES:-}" ]]; then
    read -r -a CANDIDATES <<< "${BATCH_CANDIDATES}"
    TUNE_ARGS+=(--candidates "${CANDIDATES[@]}")
  fi
  "${PYTHON_BIN}" -m b200_experiment.cli autotune-batch "${TUNE_ARGS[@]}"
else
  echo "Skipping batch search; fixed prompt BATCH_SIZE=${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-64}}}, NUM_RESPONSES=${NUM_RESPONSES:-1}."
fi
require_b200_validation

echo "Smoke/preflight passed. Full training was not started."
