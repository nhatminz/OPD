#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

BASE_CONFIG="${REPO_DIR}/configs/qwen3_b200_base.yaml"
TA_CONFIG="${REPO_DIR}/configs/qwen3_b200_ta.yaml"
RAC_CONFIG="${REPO_DIR}/configs/qwen3_b200_rac.yaml"
AUTOTUNE_CONFIG="${AUTOTUNE_CONFIG:-${REPO_DIR}/configs/qwen3_b200_autotuned.yaml}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${REPO_DIR}/results/preflight.json}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    return 1
  fi
}

require_b200_validation() {
  require_file "${PREFLIGHT_REPORT}" || {
    echo "Run: bash scripts/smoke_test_b200.sh" >&2
    return 1
  }
  require_file "${AUTOTUNE_CONFIG}" || {
    echo "Run: bash scripts/smoke_test_b200.sh" >&2
    return 1
  }
  "${PYTHON_BIN}" - "${PREFLIGHT_REPORT}" "${AUTOTUNE_CONFIG}" <<'PY'
import json
import sys
import yaml
from pathlib import Path

preflight = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
autotune = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
if preflight.get("status") != "passed":
    raise SystemExit("B200 preflight did not pass")
if not autotune.get("autotune", {}).get("validated"):
    raise SystemExit("B200 batch config is not validated")
batch = autotune["autotune"]["selected_batch_size"]
if batch != autotune["rollout"]["batch_size"] or batch != autotune["training"]["micro_batch_size"]:
    raise SystemExit("Autotune batch/micro-batch values are inconsistent")
print(f"Validated B200 batch size: {batch} (gradient accumulation: 1)")
PY
}

build_training_args() {
  local output_dir="$1"
  COMMON_TRAIN_ARGS=(
    --overlay "${AUTOTUNE_CONFIG}"
    --set "experiment.output_dir=${output_dir}"
    --set "training.epochs=${EPOCHS:-1}"
    --set "training.learning_rate=${LEARNING_RATE:-1.0e-5}"
    --set "training.save_interval=${SAVE_INTERVAL:-100}"
    --set "rollout.max_new_tokens=${MAX_NEW_TOKENS:-256}"
    --set "token_budget.rho=${RHO:-0.10}"
    --set "selector.top_k=${TOP_K:-16}"
  )
  if [[ -n "${MAX_STEPS:-}" ]]; then
    COMMON_TRAIN_ARGS+=(--set "training.max_steps=${MAX_STEPS}")
  fi
}
