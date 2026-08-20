# Full B200 runbook

## Fresh environment

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Không pin thủ công một CUDA wheel khác trước `requirements.txt`: image B200 cần PyTorch/vLLM phù hợp
driver thực tế. Nếu cluster cung cấp module/container PyTorch đã kiểm thử, dùng stack đó và cài phần
requirements còn lại theo chính sách cluster.

## Validate environment and schemas

```bash
export STORAGE_ROOT=/workspace/storage-shared
python scripts/check_b200_env.py
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_test_b200.sh
```

`smoke_test_b200.sh` chạy unit tests và preflight, đọc full training/eval schema nhưng không train full.
Báo cáo nằm ở `results/preflight.json`.

Có thể chỉ inspect schema/path qua preflight CLI:

```bash
CUDA_VISIBLE_DEVICES=0 python -m b200_experiment.cli preflight \
  --config configs/qwen3_b200_base.yaml \
  --set paths.storage_root="$STORAGE_ROOT" \
  --output results/preflight.json
```

## Controlled full TA and Bellman-RAC runs

Tạo một pair ID rồi giữ mọi shared knob giống hệt nhau:

```bash
PAIR=$(date +%Y%m%d_%H%M%S)
export TA_RUN_NAME="ta_qwen3_4b_to_1p7b_${PAIR}"
export RAC_RUN_NAME="rac_bellman_qwen3_4b_to_1p7b_${PAIR}"

export CUDA_VISIBLE_DEVICES=0,1,2
export GLOBAL_BATCH_SIZE=8
export MICRO_BATCH_SIZE=1
export LR=1e-6
export NUM_EPOCHS=1
export MAX_PROMPT_LEN=2048
export MAX_RESPONSE_LEN=8192
export TOP_K=16
export TA_RHO=0.10
export RAC_GAMMA=0.995
export RAC_W_MIN=0.10
export RAC_BETA=2.0
export EVAL_INTERVAL=50
export SAVE_INTERVAL=50
export SEED=42

RUN_NAME="$TA_RUN_NAME" bash scripts/train_ta_b200.sh
RUN_NAME="$RAC_RUN_NAME" bash scripts/train_rac_b200.sh
```

Không export `MAX_STEPS`, hoặc đặt `MAX_STEPS=-1`, để consume full configured epoch. `MAX_STEPS=N`
chỉ dành cho debug/explicit total target; nó không phải số step chạy thêm sau resume.

Một override hợp lệ, áp giống nhau cho hai method:

```bash
LR=5e-7 GLOBAL_BATCH_SIZE=4 EVAL_INTERVAL=40 \
RUN_NAME="$RAC_RUN_NAME" bash scripts/train_rac_b200.sh
```

Các OOM knob chính là `GLOBAL_BATCH_SIZE`, `MICRO_BATCH_SIZE`, `MAX_RESPONSE_LEN`,
`ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION`, `ROLLOUT_VLLM_MAX_NUM_SEQS` và
`ROLLOUT_VLLM_MAX_MODEL_LEN`. Chưa có peak-memory measurement trên B200 local, nên không coi default
là fit guarantee.

## Resume after interruption

Explicit checkpoint:

```bash
RUN_NAME="$TA_RUN_NAME" \
RESUME_FROM_CHECKPOINT="outputs/$TA_RUN_NAME/ta_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_ta_b200.sh

RUN_NAME="$RAC_RUN_NAME" \
RESUME_FROM_CHECKPOINT="outputs/$RAC_RUN_NAME/rac_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_rac_b200.sh
```

Automatic latest complete checkpoint:

```bash
RUN_NAME="$TA_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_ta_b200.sh
RUN_NAME="$RAC_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_rac_b200.sh
```

`RESUME=auto` đọc `latest.json`, rồi fallback sang checkpoint hoàn chỉnh mới nhất. Resume validator
từ chối đổi method/scientific config hoặc append đè log ở step mới hơn.

## Manual evaluation

Eval một RAC checkpoint trên full MATH-500/AIME24/AIME25:

```bash
RUN_NAME="$RAC_RUN_NAME" \
RAC_CHECKPOINT="outputs/$RAC_RUN_NAME/rac_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_rac_b200.sh
```

Eval Base, TA final và RAC final rồi aggregate:

```bash
TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_all_b200.sh
```

Evaluation mặc định là vLLM greedy (`n=1`, `temperature=0`) và lưu raw generation/correctness.
Fallback: `EVAL_BACKEND=hf`. Tensor parallel example:

```bash
TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
CUDA_VISIBLE_DEVICES=0,1 EVAL_VLLM_TENSOR_PARALLEL_SIZE=2 \
bash scripts/eval_all_b200.sh
```

## Plot one pair

```bash
TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  bash scripts/plot_training_progress.sh
```

Sau final aggregate:

```bash
TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  bash scripts/plot_results.sh
```

Mỗi launch tạo `results/<ta>_vs_<rac>/plots/plot_YYYYMMDD_HHMMSS/`. Override tên folder:

```bash
TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name paper_v1
```

## Output map

```text
outputs/<run>/<method>/
  resolved_config.yaml
  run_metadata.json
  metrics.jsonl
  train_metrics.csv
  eval_metrics.csv
  latest.json
  checkpoints or final/
  eval_history.jsonl
  training_eval/step-*/
  selector_scores/          # optional TA selected-token audit log (off by default)
  token_score_stats/        # compact all-valid-token histograms/quantiles/samples
```

Không kết luận TA hay Bellman-RAC tốt hơn cho tới khi hai controlled B200 run và full evaluation hoàn
tất.
