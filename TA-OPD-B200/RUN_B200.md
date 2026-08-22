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

## Controlled full OPD, TA-OPD và Bellman-RAC runs

Tạo một comparison ID rồi giữ mọi shared knob giống hệt nhau:

```bash
PAIR=$(date +%Y%m%d_%H%M%S)
export OPD_RUN_NAME="opd_qwen3_4b_to_1p7b_${PAIR}"
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

RUN_NAME="$OPD_RUN_NAME" bash scripts/train_opd_b200.sh
RUN_NAME="$TA_RUN_NAME"  bash scripts/train_ta_b200.sh
RUN_NAME="$RAC_RUN_NAME" bash scripts/train_rac_b200.sh
```

Ba YAML method chỉ khác `experiment.method` và `experiment.output_dir`. OPD thuần dùng uniform
weight `1` trên mọi valid response token; cả ba launcher đi qua cùng `common_b200.sh`, do đó dùng
cùng vLLM rollout, data order, model, seed, batch/micro-batch, LR, optimizer, checkpoint và lịch
eval mặc định step 0 / mỗi 50 step / final. Nên chạy tuần tự trên cùng GPU layout để tránh nhiễu
tài nguyên giữa các run.

Không export `MAX_STEPS`, hoặc đặt `MAX_STEPS=-1`, để consume full configured epoch. `MAX_STEPS=N`
chỉ dành cho debug/explicit total target; nó không phải số step chạy thêm sau resume.

Một override hợp lệ phải áp giống nhau cho cả ba method.

```bash
LR=5e-7 GLOBAL_BATCH_SIZE=4 EVAL_INTERVAL=40 \
RUN_NAME="$OPD_RUN_NAME" bash scripts/train_opd_b200.sh
LR=5e-7 GLOBAL_BATCH_SIZE=4 EVAL_INTERVAL=40 \
RUN_NAME="$TA_RUN_NAME" bash scripts/train_ta_b200.sh
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
RUN_NAME="$OPD_RUN_NAME" \
RESUME_FROM_CHECKPOINT="outputs/$OPD_RUN_NAME/opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_opd_b200.sh

RUN_NAME="$TA_RUN_NAME" \
RESUME_FROM_CHECKPOINT="outputs/$TA_RUN_NAME/ta_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_ta_b200.sh

RUN_NAME="$RAC_RUN_NAME" \
RESUME_FROM_CHECKPOINT="outputs/$RAC_RUN_NAME/rac_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_rac_b200.sh
```

Automatic latest complete checkpoint:

```bash
RUN_NAME="$OPD_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_opd_b200.sh
RUN_NAME="$TA_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_ta_b200.sh
RUN_NAME="$RAC_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_rac_b200.sh
```

`RESUME=auto` đọc `latest.json`, rồi fallback sang checkpoint hoàn chỉnh mới nhất. Resume validator
từ chối đổi method/scientific config. Nếu output đã có log/checkpoint mới hơn checkpoint được chọn,
resume sẽ tự rewind output về đúng step đó rồi ghi lại các step tiếp theo.

## Manual evaluation

Eval một RAC checkpoint trên full MATH-500/AIME24/AIME25:

```bash
RUN_NAME="$RAC_RUN_NAME" \
RAC_CHECKPOINT="outputs/$RAC_RUN_NAME/rac_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_rac_b200.sh
```

Eval Base, OPD, TA và RAC final rồi aggregate:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_all_b200.sh
```

Evaluation mặc định dùng vLLM sampling (`n=1`, `temperature=1.0`, seed `1234`) và lưu raw
generation/correctness.
Fallback: `EVAL_BACKEND=hf`. Tensor parallel example:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
CUDA_VISIBLE_DEVICES=0,1 EVAL_VLLM_TENSOR_PARALLEL_SIZE=2 \
bash scripts/eval_all_b200.sh
```

## Re-eval mọi checkpoint đã lưu ở temperature 1

Script dưới đây tìm `checkpoint-<step>` và `final/` trong cả ba output. Nó cũng eval lại base ở
step 0 để toàn bộ đường accuracy dùng cùng temperature. Mỗi `training_eval/step-*` tương ứng,
`eval_history.jsonl` và `eval_metrics.csv` cũ sẽ bị thay thế; raw prediction và `summary.json`
trong từng step cũng bị thay thế.

Kiểm tra danh sách checkpoint trước, không ghi file:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  REEVAL_DRY_RUN=true \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Chạy thật:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Tensor parallel cho từng evaluator vLLM:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  CUDA_VISIBLE_DEVICES=0,1 \
  REEVAL_VLLM_TENSOR_PARALLEL_SIZE=2 \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Script chạy tuần tự một subprocess vLLM cho mỗi checkpoint để trả VRAM sau từng lượt. Nó từ chối
ghi lịch sử mới nếu phát hiện thư mục eval cũ không có checkpoint tương ứng, tránh trộn kết quả
temperature 0 và 1. Sau khi hoàn tất, chạy lại lệnh plot periodic training evaluation bên dưới.

## Plot periodic training evaluation

So sánh cả ba phương pháp:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  PLOT_METHODS="opd ta rac" \
  bash scripts/plot_training_progress.sh
```

So sánh hai phương pháp bất kỳ (đổi danh sách theo nhu cầu):

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  PLOT_METHODS="opd rac" \
  bash scripts/plot_training_progress.sh --plot-name opd_vs_rac

OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" \
  PLOT_METHODS="opd ta" \
  bash scripts/plot_training_progress.sh --plot-name opd_vs_ta

TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  PLOT_METHODS="ta rac" \
  bash scripts/plot_training_progress.sh --plot-name ta_vs_rac
```

Vẽ riêng một phương pháp để báo cáo, với ba đường MATH-500/AIME24/AIME25 trong cùng một ảnh:

```bash
RUN_NAME="$OPD_RUN_NAME" PLOT_METHODS=opd \
  bash scripts/plot_training_progress.sh --plot-name opd_report

RUN_NAME="$TA_RUN_NAME" PLOT_METHOD=ta \
  bash scripts/plot_training_progress.sh --plot-name ta_report

RUN_NAME="$RAC_RUN_NAME" PLOT_METHOD=rac \
  bash scripts/plot_training_progress.sh --plot-name rac_report
```

`PLOT_METHODS` nhận danh sách cách nhau bởi dấu cách hoặc dấu phẩy. Chế độ riêng chỉ cần
`eval_history.jsonl`; chế độ so sánh còn đọc `metrics.jsonl` để vẽ loss. Mọi chế độ sinh cả
PNG/PDF cùng CSV/JSON số liệu. `PLOT_METHOD=both` cũ vẫn có nghĩa `ta rac`.

Sau final aggregate:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  bash scripts/plot_results.sh
```

Chế độ so sánh tạo `results/<run1>_vs_<run2>.../plots/plot_YYYYMMDD_HHMMSS/`; chế độ riêng tạo
`results/<run>/plots/plot_YYYYMMDD_HHMMSS/`. Override tên folder:

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

Không kết luận method nào tốt hơn cho tới khi cả ba controlled B200 run và full evaluation hoàn tất.
