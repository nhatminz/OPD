#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_b200.sh"

resolve_run_paths
for path in "${OPD_RUN_OUTPUT}" "${TA_RUN_OUTPUT}" "${RAC_RUN_OUTPUT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Missing training output directory: ${path}" >&2
    exit 1
  fi
  require_file "${path}/resolved_config.yaml"
done

ARGS=(
  --opd-output "${OPD_RUN_OUTPUT}"
  --ta-output "${TA_RUN_OUTPUT}"
  --rac-output "${RAC_RUN_OUTPUT}"
  --temperature 1.0
  --max-new-tokens "${REEVAL_MAX_NEW_TOKENS:-8192}"
  --tensor-parallel-size "${REEVAL_VLLM_TENSOR_PARALLEL_SIZE:-${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}}"
  --gpu-memory-utilization "${REEVAL_VLLM_GPU_MEMORY_UTILIZATION:-${EVAL_VLLM_GPU_MEMORY_UTILIZATION:-auto}}"
  --gpu-headroom-gib "${REEVAL_VLLM_GPU_HEADROOM_GIB:-${EVAL_VLLM_GPU_HEADROOM_GIB:-4}}"
  --max-num-seqs "${REEVAL_VLLM_MAX_NUM_SEQS:-${EVAL_VLLM_MAX_NUM_SEQS:-256}}"
  --max-model-len "${REEVAL_VLLM_MAX_MODEL_LEN:-${EVAL_VLLM_MAX_MODEL_LEN:-12288}}"
  --seed "${REEVAL_SEED:-1234}"
)

case "${REEVAL_SKIP_BASE:-false}" in
  true|TRUE|1|yes|YES) ARGS+=(--skip-base) ;;
esac
case "${REEVAL_ALLOW_UNMATCHED_EVAL_DIRECTORIES:-false}" in
  true|TRUE|1|yes|YES) ARGS+=(--allow-unmatched-eval-directories) ;;
esac
case "${REEVAL_DRY_RUN:-false}" in
  true|TRUE|1|yes|YES)
    ARGS+=(--dry-run)
    IS_DRY_RUN=true
    ;;
  *) IS_DRY_RUN=false ;;
esac

echo "Re-evaluating every saved OPD, TA-OPD, and RAC checkpoint."
echo "Temperature: 1.0; backend: vLLM; seed: ${REEVAL_SEED:-1234}"
if [[ "${IS_DRY_RUN}" == "true" ]]; then
  echo "Dry run: files will only be validated and listed; nothing will be written."
else
  echo "Existing training_eval/step-*, eval_history.jsonl, and eval_metrics.csv will be replaced."
fi
echo "OPD output: ${OPD_RUN_OUTPUT}"
echo "TA output:  ${TA_RUN_OUTPUT}"
echo "RAC output: ${RAC_RUN_OUTPUT}"

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.checkpoint_evaluation "${ARGS[@]}"
