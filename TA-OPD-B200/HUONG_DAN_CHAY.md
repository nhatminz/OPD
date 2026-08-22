# Hướng dẫn nhanh OPD, TA-OPD và Bellman-RAC trên B200

Hướng dẫn chi tiết hơn nằm trong [`RUN_B200.md`](RUN_B200.md).

## 1. Cài và kiểm tra

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python scripts/check_b200_env.py
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_test_b200.sh
```

## 2. Chạy full và công bằng

Đặt ba tên riêng nhưng dùng cùng suffix và cùng mọi shared hyperparameter:

```bash
PAIR=$(date +%Y%m%d_%H%M%S)
export OPD_RUN_NAME="opd_qwen3_4b_to_1p7b_${PAIR}"
export TA_RUN_NAME="ta_qwen3_4b_to_1p7b_${PAIR}"
export RAC_RUN_NAME="rac_bellman_qwen3_4b_to_1p7b_${PAIR}"

RUN_NAME="$OPD_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/train_opd_b200.sh

RUN_NAME="$TA_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/train_ta_b200.sh

RUN_NAME="$RAC_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/train_rac_b200.sh
```

Không đặt `MAX_STEPS` (hoặc để `-1`) để dùng toàn bộ epoch/full DAPO. OPD dùng uniform weight `1`
trên mọi valid response token. Cả ba method dùng cùng vLLM rollout và eval step 0 / mỗi 50 step /
final. Nếu cần hạ memory, đặt cùng giá trị cho cả ba lệnh.

```bash
GLOBAL_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_RESPONSE_LEN=4096 \
  RUN_NAME="$OPD_RUN_NAME" CUDA_VISIBLE_DEVICES=0 bash scripts/train_opd_b200.sh
GLOBAL_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_RESPONSE_LEN=4096 \
  RUN_NAME="$TA_RUN_NAME" CUDA_VISIBLE_DEVICES=0 bash scripts/train_ta_b200.sh
GLOBAL_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_RESPONSE_LEN=4096 \
  RUN_NAME="$RAC_RUN_NAME" CUDA_VISIBLE_DEVICES=0 bash scripts/train_rac_b200.sh
```

## 3. Resume

Tự tìm checkpoint hoàn chỉnh mới nhất trong run:

```bash
RUN_NAME="$OPD_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_opd_b200.sh
RUN_NAME="$TA_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_ta_b200.sh
RUN_NAME="$RAC_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_rac_b200.sh
```

Hoặc chỉ rõ checkpoint:

```bash
RESUME_FROM_CHECKPOINT="outputs/$RAC_RUN_NAME/rac_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_rac_b200.sh
```

Giữ nguyên model/data, batch, rollout length, seed, LR và hyperparameter khoa học khi resume.

## 4. Eval và plot

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval_all_b200.sh

OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  PLOT_METHODS="opd ta rac" \
  bash scripts/plot_training_progress.sh
```

So sánh hai method bất kỳ:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  PLOT_METHODS="opd rac" \
  bash scripts/plot_training_progress.sh --plot-name opd_vs_rac
```

Vẽ riêng accuracy của một method trên cả ba bộ eval (ba đường trong cùng một ảnh):

```bash
RUN_NAME="$OPD_RUN_NAME" PLOT_METHODS=opd \
  bash scripts/plot_training_progress.sh --plot-name opd_report

RUN_NAME="$TA_RUN_NAME" PLOT_METHOD=ta \
  bash scripts/plot_training_progress.sh --plot-name ta_report

RUN_NAME="$RAC_RUN_NAME" PLOT_METHOD=rac \
  bash scripts/plot_training_progress.sh --plot-name rac_report
```

`PLOT_METHODS` nhận một, hai hoặc ba giá trị trong `opd ta rac`. `PLOT_METHOD=both` vẫn giữ tương
thích với biểu đồ TA-OPD/RAC cũ.

Manual eval riêng checkpoint RAC:

```bash
RUN_NAME="$RAC_RUN_NAME" \
RAC_CHECKPOINT="outputs/$RAC_RUN_NAME/rac_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_rac_b200.sh
```

Periodic và final eval luôn chạy toàn bộ MATH-500, AIME24, AIME25; `max_num_seqs` là concurrency,
không phải subset size.

Eval lại toàn bộ checkpoint đã lưu với vLLM `temperature=1.0`, đồng thời ghi đè lịch sử/file eval
cũ của OPD, TA-OPD và RAC:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  REEVAL_DRY_RUN=true CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_all_checkpoints_b200.sh

OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Dòng đầu chỉ kiểm tra danh sách; dòng thứ hai mới thực sự ghi đè. Script mặc định eval lại cả base
step 0 để không trộn temperature giữa baseline và checkpoint.
