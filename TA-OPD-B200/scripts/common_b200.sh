#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"

# ================= MODEL / TRAIN-DATA PRESET: EDIT HERE =================
# Paths may be absolute or relative to STORAGE_ROOT. Environment overrides
# take precedence, so the same launchers work for any vocabulary-compatible
# teacher/student pair without editing YAML files.
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${TEACHER_PATH:-models/Qwen3-8B}}"
export STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${STUDENT_PATH:-nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base}}"
export TRAIN_DATASET="${TRAIN_DATASET:-competition_math}"  # competition_math, dapo_math, custom

case "${TRAIN_DATASET,,}" in
  competition_math|competition-math|math)
    PRESET_TRAIN_DATA_PATH="nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet"
    PRESET_TRAIN_DATA_SPLIT="null"
    PRESET_TRAIN_PROMPT_KEY="problem"
    PRESET_TRAIN_PREFER_SOURCE_PROMPT="false"
    ;;
  dapo_math|dapo-math|dapo)
    PRESET_TRAIN_DATA_PATH="nlp/minhpn19/data/DAPO-Math-17k-Processed"
    PRESET_TRAIN_DATA_SPLIT="all"
    PRESET_TRAIN_PROMPT_KEY="prompt"
    PRESET_TRAIN_PREFER_SOURCE_PROMPT="false"
    ;;
  custom)
    if [[ -z "${TRAIN_DATA_PATH:-}" || -z "${TRAIN_PROMPT_KEY:-}" ]]; then
      echo "TRAIN_DATASET=custom requires TRAIN_DATA_PATH and TRAIN_PROMPT_KEY" >&2
      return 2 2>/dev/null || exit 2
    fi
    PRESET_TRAIN_DATA_PATH="${TRAIN_DATA_PATH}"
    PRESET_TRAIN_DATA_SPLIT="${TRAIN_DATA_SPLIT:-null}"
    PRESET_TRAIN_PROMPT_KEY="${TRAIN_PROMPT_KEY}"
    PRESET_TRAIN_PREFER_SOURCE_PROMPT="${TRAIN_PREFER_SOURCE_PROMPT:-false}"
    ;;
  *)
    echo "Unknown TRAIN_DATASET=${TRAIN_DATASET}; use competition_math, dapo_math, or custom" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${PRESET_TRAIN_DATA_PATH}}"
export TRAIN_DATA_SPLIT="${TRAIN_DATA_SPLIT:-${PRESET_TRAIN_DATA_SPLIT}}"
export TRAIN_PROMPT_KEY="${TRAIN_PROMPT_KEY:-${PRESET_TRAIN_PROMPT_KEY}}"
export TRAIN_PREFER_SOURCE_PROMPT="${TRAIN_PREFER_SOURCE_PROMPT:-${PRESET_TRAIN_PREFER_SOURCE_PROMPT}}"
# This experiment intentionally has no teacher-tokenizer path. Qwen3's hard
# no-think switch is always applied while the student tokenizer renders the one
# shared prompt. Trainer validation rejects attempts to enable thinking.
# ========================================================================

BASE_CONFIG="${REPO_DIR}/configs/qwen3_b200_base.yaml"
OPD_CONFIG="${REPO_DIR}/configs/qwen3_b200_opd.yaml"
TA_CONFIG="${REPO_DIR}/configs/qwen3_b200_ta.yaml"
RAC_CONFIG="${REPO_DIR}/configs/qwen3_b200_rac.yaml"

storage_asset_path() {
  if [[ "$1" == /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "${STORAGE_ROOT%/}" "$1"
  fi
}

TEACHER_MODEL_ABS="$(storage_asset_path "${TEACHER_MODEL_PATH}")"
STUDENT_MODEL_ABS="$(storage_asset_path "${STUDENT_MODEL_PATH}")"
TRAIN_DATA_ABS="$(storage_asset_path "${TRAIN_DATA_PATH}")"

# Every model/data protocol gets an independent validation namespace.
ASSET_FINGERPRINT="$(
  "${PYTHON_BIN}" - \
    "${TEACHER_MODEL_ABS}" \
    "${STUDENT_MODEL_ABS}" \
    "${TRAIN_DATA_ABS}" \
    "${TRAIN_DATA_SPLIT}" \
    "${TRAIN_PROMPT_KEY}" \
    "${TRAIN_PREFER_SOURCE_PROMPT}" <<'PY'
import hashlib
import json
import sys

payload = json.dumps(sys.argv[1:], ensure_ascii=False, separators=(",", ":"))
print(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20])
PY
)"

# Preflight checks every visible GPU, while autotuning is also topology-
# dependent. Keep both outputs separate per requested worker count so a
# one-GPU and a four-GPU workload cannot consume or overwrite each other's
# validation state.
_B200_VISIBLE_DEVICES=()
IFS=',' read -r -a _B200_VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0}"
VALIDATION_WORLD_SIZE="${TRAIN_NPROC_PER_NODE:-${#_B200_VISIBLE_DEVICES[@]}}"
if ! [[ "${VALIDATION_WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_NPROC_PER_NODE must be a positive integer, got ${VALIDATION_WORLD_SIZE}" >&2
  return 2 2>/dev/null || exit 2
fi
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${REPO_DIR}/results/preflight/asset-${ASSET_FINGERPRINT}-${VALIDATION_WORLD_SIZE}gpu.json}"
AUTOTUNE_CONFIG="${AUTOTUNE_CONFIG:-${REPO_DIR}/configs/autotuned/asset-${ASSET_FINGERPRINT}-${VALIDATION_WORLD_SIZE}gpu.yaml}"
AUTOTUNE_OUTPUT_ROOT="${AUTOTUNE_OUTPUT_ROOT:-${REPO_DIR}/outputs/autotune/asset-${ASSET_FINGERPRINT}-${VALIDATION_WORLD_SIZE}gpu}"

ASSET_CONFIG_ARGS=(
  --set "models.teacher_path=${TEACHER_MODEL_PATH}"
  --set "models.student_path=${STUDENT_MODEL_PATH}"
  --set "models.teacher_no_think=true"
  --set "data.path=${TRAIN_DATA_PATH}"
  --set "data.split=${TRAIN_DATA_SPLIT}"
  --set "data.prompt_key=${TRAIN_PROMPT_KEY}"
  --set "data.prefer_source_prompt=${TRAIN_PREFER_SOURCE_PROMPT}"
  --set "data.chat_template_kwargs.enable_thinking=false"
)

print_asset_selection() {
  echo "Teacher model: ${TEACHER_MODEL_ABS}"
  echo "Student model/tokenizer: ${STUDENT_MODEL_ABS}"
  echo "Training dataset: preset=${TRAIN_DATASET}, path=${TRAIN_DATA_ABS}, split=${TRAIN_DATA_SPLIT}, prompt_key=${TRAIN_PROMPT_KEY}"
  echo "Teacher protocol: no-think, shared student token IDs, no teacher re-tokenization"
  echo "Asset fingerprint: ${ASSET_FINGERPRINT}"
  echo "Preflight report: ${PREFLIGHT_REPORT}"
  echo "Validation topology: ${VALIDATION_WORLD_SIZE} GPU(s)"
  echo "Autotune config: ${AUTOTUNE_CONFIG}"
}

resolve_run_paths() {
  RUN_NAME="${RUN_NAME:-run01}"
  local opd_name="${OPD_RUN_NAME:-${RUN_NAME}}"
  local ta_name="${TA_RUN_NAME:-${RUN_NAME}}"
  local rac_name="${RAC_RUN_NAME:-${RUN_NAME}}"
  if [[ -n "${COMPARISON_NAME:-}" ]]; then
    COMPARISON_NAME="${COMPARISON_NAME}"
  elif [[ -n "${OPD_RUN_NAME:-}" ]]; then
    COMPARISON_NAME="${opd_name}_vs_${ta_name}_vs_${rac_name}"
  elif [[ -n "${TA_RUN_NAME:-}" || -n "${RAC_RUN_NAME:-}" ]]; then
    COMPARISON_NAME="${ta_name}_vs_${rac_name}"
  else
    COMPARISON_NAME="${RUN_NAME}"
  fi
  for name in "${RUN_NAME}" "${opd_name}" "${ta_name}" "${rac_name}" "${COMPARISON_NAME}"; do
    if ! [[ "${name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
      echo "Run names may contain only letters, numbers, dot, underscore, and dash: ${name}" >&2
      return 1
    fi
  done
  OPD_RUN_NAME="${opd_name}"
  TA_RUN_NAME="${ta_name}"
  RAC_RUN_NAME="${rac_name}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs}"
  OPD_RUN_OUTPUT="${OPD_OUTPUT_DIR:-${OUTPUT_ROOT}/${OPD_RUN_NAME}/opd}"
  TA_RUN_OUTPUT="${TA_OUTPUT_DIR:-${OUTPUT_ROOT}/${TA_RUN_NAME}/ta_opd}"
  RAC_RUN_OUTPUT="${RAC_OUTPUT_DIR:-${OUTPUT_ROOT}/${RAC_RUN_NAME}/rac_opd}"
  RUN_RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results/${COMPARISON_NAME}}"
}

visible_gpu_count() {
  local visible="${CUDA_VISIBLE_DEVICES:-0}"
  local devices=()
  IFS=',' read -r -a devices <<< "${visible}"
  if [[ "${#devices[@]}" -le 0 ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain at least one GPU" >&2
    return 1
  fi
  printf '%s\n' "${#devices[@]}"
}

run_training_cli() {
  local visible_count
  visible_count="$(visible_gpu_count)"
  local processes="${TRAIN_NPROC_PER_NODE:-${visible_count}}"
  if ! [[ "${processes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_NPROC_PER_NODE must be a positive integer, got ${processes}" >&2
    return 1
  fi
  if (( processes > visible_count )); then
    echo "Requested ${processes} workers but only ${visible_count} GPUs are visible" >&2
    return 1
  fi
  if (( processes == 1 )); then
    "${PYTHON_BIN}" -m b200_experiment.cli "$@"
  else
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${processes}" \
      -m b200_experiment.cli "$@"
  fi
}

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
  "${PYTHON_BIN}" - \
    "${PREFLIGHT_REPORT}" \
    "${TEACHER_MODEL_ABS}" \
    "${STUDENT_MODEL_ABS}" \
    "${TRAIN_DATA_ABS}" \
    "${TRAIN_DATA_SPLIT}" <<'PY'
import json
import sys
from pathlib import Path

preflight = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if preflight.get("status") != "passed":
    raise SystemExit("B200 preflight did not pass")
expected_split = None if sys.argv[5] == "null" else sys.argv[5]
expected = {
    "teacher": str(Path(sys.argv[2]).resolve()),
    "student": str(Path(sys.argv[3]).resolve()),
    "training_data": str(Path(sys.argv[4]).resolve()),
    "split": expected_split,
}
actual = {
    "teacher": str(Path(preflight["models"]["teacher_path"]).resolve()),
    "student": str(Path(preflight["models"]["student_path"]).resolve()),
    "training_data": str(Path(preflight["training_data"]["path"]).resolve()),
    "split": preflight["training_data"].get("split"),
}
if actual != expected:
    raise SystemExit(
        "B200 preflight belongs to a different model/data selection; "
        "rerun bash scripts/smoke_test_b200.sh. "
        f"expected={expected}, actual={actual}"
    )
protocol = preflight["models"].get("tokenizer_protocol", {})
if protocol.get("tokenizer_source") != "student" or not protocol.get(
    "teacher_no_think"
):
    raise SystemExit("B200 preflight did not validate the no-think shared-token protocol")
PY
  if batch_autotune_enabled; then
    require_file "${AUTOTUNE_CONFIG}" || {
      echo "Run: USE_BATCH_AUTOTUNE=true bash scripts/smoke_test_b200.sh" >&2
      return 1
    }
    "${PYTHON_BIN}" - "${AUTOTUNE_CONFIG}" <<'PY'
import sys
import yaml
from pathlib import Path

autotune = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not autotune.get("autotune", {}).get("validated"):
    raise SystemExit("B200 batch config is not validated")
batch = autotune["autotune"]["selected_batch_size"]
micro = autotune["training"]["micro_batch_size_per_gpu"]
if batch != autotune["rollout"]["batch_size"] or micro <= 0:
    raise SystemExit("Autotune rollout/micro-batch values are inconsistent")
print(f"Autotuned B200 global batch size: {batch} (microbatch/GPU: {micro})")
PY
  else
    "${PYTHON_BIN}" - "${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-64}}}" "${MICRO_BATCH_SIZE_PER_GPU:-${MICRO_BATCH_SIZE:-8}}" "$(visible_gpu_count)" "${NUM_RESPONSES:-1}" <<'PY'
import sys

batch, micro, world_size, responses = map(int, sys.argv[1:])
if min(batch, micro, world_size, responses) <= 0:
    raise SystemExit("Batch, microbatch/GPU, world size and responses must be positive")
local_prompts = (batch + world_size - 1) // world_size
print(
    f"FSDP batch: global prompts={batch}, local prompts/GPU={local_prompts}, "
    f"n={responses}, local trajectories/GPU={local_prompts * responses}, "
    f"microbatch/GPU={micro}"
)
PY
  fi
}

batch_autotune_enabled() {
  case "${USE_BATCH_AUTOTUNE:-false}" in
    true|TRUE|1|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

build_training_args() {
  local output_dir="$1"
  local prompt_batch="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-64}}}"
  local response_count="${NUM_RESPONSES:-1}"
  local trajectory_batch=$((prompt_batch * response_count))
  local process_count="${TRAIN_NPROC_PER_NODE:-$(visible_gpu_count)}"
  local local_trajectory_batch=$(((trajectory_batch + process_count - 1) / process_count))
  COMMON_TRAIN_ARGS=(
    "${ASSET_CONFIG_ARGS[@]}"
    --set "experiment.output_dir=${output_dir}"
    --set "paths.storage_root=${STORAGE_ROOT}"
    --set "experiment.seed=${SEED:-${EXPERIMENT_SEED:-42}}"
    --set "data.max_prompt_tokens=${MAX_PROMPT_LENGTH:-${MAX_PROMPT_LEN:-1024}}"
    --set "training.epochs=${NUM_EPOCHS:-${EPOCHS:-1}}"
    --set "training.learning_rate=${LR:-${LEARNING_RATE:-1.0e-6}}"
    --set "training.save_interval=${SAVE_INTERVAL:-50}"
    --set "training.grad_accum_steps=${GRAD_ACCUM_STEPS:-auto}"
    --set "training.gradient_checkpointing=${GRADIENT_CHECKPOINTING:-true}"
    --set "distributed.strategy=${DISTRIBUTED_STRATEGY:-fsdp}"
    --set "distributed.fsdp.teacher_cpu_offload=${FSDP_TEACHER_CPU_OFFLOAD:-false}"
    --set "distributed.fsdp.use_no_sync=${FSDP_USE_NO_SYNC:-false}"
    --set "distributed.bucket_cap_mb=${DDP_BUCKET_CAP_MB:-100}"
    --set "rollout.backend=${ROLLOUT_BACKEND:-vllm}"
    --set "rollout.seed=${ROLLOUT_SEED:-42}"
    --set "rollout.num_responses=${response_count}"
    --set "rollout.max_new_tokens=${MAX_RESPONSE_LENGTH:-${MAX_RESPONSE_LEN:-${MAX_NEW_TOKENS:-7168}}}"
    --set "rollout.temperature=${ROLLOUT_TEMPERATURE:-1.0}"
    --set "rollout.top_p=${ROLLOUT_TOP_P:-1.0}"
    --set "rollout.vllm.gpu_memory_utilization=${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.40}"
    --set "rollout.vllm.max_num_seqs=${ROLLOUT_VLLM_MAX_NUM_SEQS:-${local_trajectory_batch}}"
    --set "rollout.vllm.max_model_len=${ROLLOUT_VLLM_MAX_MODEL_LEN:-9216}"
    --set "rollout.vllm.max_concurrent_requests=${ROLLOUT_VLLM_MAX_CONCURRENT_REQUESTS:-${local_trajectory_batch}}"
    --set "rollout.vllm.wake_headroom_gib=${ROLLOUT_VLLM_WAKE_HEADROOM_GIB:-2}"
    --set "rollout.vllm.enable_chunked_prefill=${ROLLOUT_VLLM_ENABLE_CHUNKED_PREFILL:-true}"
    --set "rollout.vllm.performance_mode=${ROLLOUT_VLLM_PERFORMANCE_MODE:-throughput}"
    --set "rollout.vllm.async_scheduling=${ROLLOUT_VLLM_ASYNC_SCHEDULING:-true}"
    --set "rollout.vllm.logprob_sanity.enabled=${VLLM_LOGPROB_SANITY_ENABLED:-false}"
    --set "rollout.vllm.logprob_sanity.max_tokens_per_rank=${VLLM_LOGPROB_SANITY_TOKENS:-32}"
    --set "rollout.vllm.logprob_sanity.tolerance=${VLLM_LOGPROB_SANITY_TOLERANCE:-0.05}"
    --set "rollout.vllm.logprob_sanity.fail_on_mismatch=${VLLM_LOGPROB_SANITY_FAIL:-false}"
    --set "opd.teacher_temperature=${TEACHER_TEMPERATURE:-1.0}"
    --set "token_budget.rho=${TA_RHO:-${RHO:-0.10}}"
    --set "selector.top_k=${TOP_K:-16}"
    --set "selector.score_micro_batch_size=${SCORE_MICRO_BATCH_SIZE:-8}"
    --set "selector.trim_padding=${TRIM_RESPONSE_PADDING:-true}"
    --set "selector.length_bucketed_scoring=${LENGTH_BUCKETED_SCORING:-true}"
    --set "selector.joint_cross_scoring=${JOINT_CROSS_SCORING:-true}"
    --set "selector.score_chunk_steps=${SCORE_CHUNK_STEPS:-128}"
    --set "selector.ta_vocab_chunk_tokens=${TA_VOCAB_CHUNK_TOKENS:-2048}"
    --set "selector.rac_gamma=${RAC_GAMMA:-0.995}"
    --set "selector.rac_w_min=${RAC_W_MIN:-0.10}"
    --set "selector.rac_beta=${RAC_BETA:-2.0}"
    --set "selector.rac_scan_backend=${RAC_SCAN_BACKEND:-parallel}"
    --set "logging.token_score_interval=${TOKEN_SCORE_INTERVAL:-${EVAL_INTERVAL:-50}}"
    --set "logging.log_interval=${LOG_INTERVAL:-1}"
    --set "logging.tensorboard.enabled=${TENSORBOARD_ENABLED:-true}"
    --set "logging.tensorboard.log_interval=${TENSORBOARD_LOG_INTERVAL:-1}"
    --set "training_evaluation.enabled=${TRAIN_EVAL_ENABLED:-true}"
    --set "training_evaluation.backend=${TRAIN_EVAL_BACKEND:-vllm}"
    --set "training_evaluation.temperature=${TRAIN_EVAL_TEMPERATURE:-0.7}"
    --set "training_evaluation.top_p=${TRAIN_EVAL_TOP_P:-0.95}"
    --set "training_evaluation.num_responses=${TRAIN_EVAL_NUM_RESPONSES:-16}"
    --set "training_evaluation.target_evaluations=null"
    --set "training_evaluation.interval_steps=${TRAIN_EVAL_INTERVAL:-${EVAL_INTERVAL:-50}}"
    --set "training_evaluation.limit=null"
    --set "training_evaluation.batch_size=${TRAIN_EVAL_BATCH_SIZE:-1}"
    --set "training_evaluation.max_new_tokens=${TRAIN_EVAL_MAX_NEW_TOKENS:-7168}"
    --set "training_evaluation.reuse_base_evaluation=${TRAIN_EVAL_REUSE_BASE:-true}"
    --set "training_evaluation.base_cache_dir=${TRAIN_EVAL_BASE_CACHE_DIR:-outputs/.base_eval_cache}"
    --set "training_evaluation.vllm.tensor_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    --set "training_evaluation.vllm.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-auto}"
    --set "training_evaluation.vllm.gpu_headroom_gib=${VLLM_GPU_HEADROOM_GIB:-4}"
    --set "training_evaluation.vllm.max_num_seqs=${VLLM_MAX_NUM_SEQS:-256}"
    --set "training_evaluation.vllm.max_model_len=${VLLM_MAX_MODEL_LEN:-9216}"
    --set "training_evaluation.vllm.enable_chunked_prefill=${VLLM_ENABLE_CHUNKED_PREFILL:-true}"
    --set "training_evaluation.vllm.performance_mode=${VLLM_PERFORMANCE_MODE:-throughput}"
    --set "training_evaluation.vllm.async_scheduling=${VLLM_ASYNC_SCHEDULING:-true}"
  )
  if batch_autotune_enabled; then
    COMMON_TRAIN_ARGS=(--overlay "${AUTOTUNE_CONFIG}" "${COMMON_TRAIN_ARGS[@]}")
  else
    COMMON_TRAIN_ARGS+=(
      --set "rollout.batch_size=${prompt_batch}"
      --set "training.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-${MICRO_BATCH_SIZE:-8}}"
      --set "training.length_bucketed_micro_batches=${LENGTH_BUCKETED_MICRO_BATCHES:-true}"
    )
  fi
  if [[ -n "${TRAIN_EVAL_TARGET:-}" ]]; then
    COMMON_TRAIN_ARGS+=(
      --set "training_evaluation.target_evaluations=${TRAIN_EVAL_TARGET}"
      --set "training_evaluation.interval_steps=null"
    )
  fi
  if [[ -n "${MAX_STEPS:-}" && "${MAX_STEPS}" != "-1" ]]; then
    COMMON_TRAIN_ARGS+=(--set "training.max_steps=${MAX_STEPS}")
  fi
  if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    COMMON_TRAIN_ARGS+=(
      --set "training.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}"
      --set "training.resume_allow_config_mismatch=${RESUME_ALLOW_CONFIG_MISMATCH:-false}"
    )
  fi
}
