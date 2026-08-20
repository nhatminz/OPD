# Hướng dẫn nhanh TA-OPD và Bellman-RAC trên B200

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

Đặt hai tên riêng nhưng dùng cùng suffix và cùng mọi shared hyperparameter:

```bash
PAIR=$(date +%Y%m%d_%H%M%S)
export TA_RUN_NAME="ta_qwen3_4b_to_1p7b_${PAIR}"
export RAC_RUN_NAME="rac_bellman_qwen3_4b_to_1p7b_${PAIR}"

RUN_NAME="$TA_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/train_ta_b200.sh

RUN_NAME="$RAC_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/train_rac_b200.sh
```

Không đặt `MAX_STEPS` (hoặc để `-1`) để dùng toàn bộ epoch/full DAPO. Nếu cần hạ memory, đặt cùng
giá trị cho hai lệnh, ví dụ:

```bash
GLOBAL_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_RESPONSE_LEN=4096 \
  RUN_NAME="$TA_RUN_NAME" CUDA_VISIBLE_DEVICES=0 bash scripts/train_ta_b200.sh
GLOBAL_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_RESPONSE_LEN=4096 \
  RUN_NAME="$RAC_RUN_NAME" CUDA_VISIBLE_DEVICES=0 bash scripts/train_rac_b200.sh
```

## 3. Resume

Tự tìm checkpoint hoàn chỉnh mới nhất trong run:

```bash
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
TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval_all_b200.sh

TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  bash scripts/plot_training_progress.sh
```

Manual eval riêng checkpoint RAC:

```bash
RUN_NAME="$RAC_RUN_NAME" \
RAC_CHECKPOINT="outputs/$RAC_RUN_NAME/rac_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_rac_b200.sh
```

Periodic và final eval luôn chạy toàn bộ MATH-500, AIME24, AIME25; `max_num_seqs` là concurrency,
không phải subset size.
