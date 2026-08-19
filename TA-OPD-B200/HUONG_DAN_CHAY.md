# Hướng dẫn train, plot, eval và resume TA-OPD/RAC trên B200

Tất cả lệnh dưới đây chạy từ project root:

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
source .venv/bin/activate
```

## 1. Cấu hình mặc định hiện tại

Hai file `scripts/train_ta_b200.sh` và `scripts/train_rac_b200.sh` có block tham số dễ sửa ở đầu
file. Giá trị mặc định hiện tại là:

```text
global batch size              = 8
training rollout max_new_tokens = 2048
max prompt tokens               = 512
training rollout max_model_len  = 4096
RAC branch M                    = 4
rho                             = 0.10
top K                           = 16
save interval                   = 100 step
```

`max_model_len=4096` tính cả prompt và response, và đủ cho cấu hình hiện tại:

```text
512 prompt + 2048 generated = 2560 context tokens <= 4096
```

Batch 8 là **global batch**. Chuyển giữa một, hai hoặc ba GPU không nhân batch lên theo số GPU.
Với 17.398 mẫu, một epoch có `ceil(17398 / 8) = 2175` optimizer step.

Sau khi đổi batch/context như trên, nên chạy lại preflight trên B200 một lần:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_test_b200.sh
```

## 2. Train mới TA-OPD

Không cần tự đặt `RUN_NAME`. Script lấy thời gian bắt đầu lệnh theo định dạng
`YYYYMMDD_HHMMSS`, ví dụ `20260819_143012`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_ta_b200.sh
```

Terminal in tên cần giữ lại ở đầu/cuối run, ví dụ:

```text
TA-OPD run: 20260819_143012
TA-OPD output: .../outputs/20260819_143012/ta_opd
TA-OPD completed. Keep this identifier: TA_RUN_NAME=20260819_143012
```

## 3. Train mới RAC

RAC là một lệnh hoàn toàn độc lập:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_rac_b200.sh
```

Ví dụ RAC bắt đầu muộn hơn và được gán tên khác:

```text
RAC run: 20260820_090530
RAC output: .../outputs/20260820_090530/rac_opd
RAC completed. Keep this identifier: RAC_RUN_NAME=20260820_090530
```

Không truyền `RESUME_FROM_CHECKPOINT` khi muốn train từ đầu. Nếu biến này từng được export trong
shell hiện tại, xóa nó trước:

```bash
unset RESUME_FROM_CHECKPOINT
```

## 4. Vẽ hình so sánh log training

Sau khi cả TA và RAC hoàn tất, ghép đúng hai timestamp đã được terminal in ra:

```bash
TA_RUN_NAME=20260819_143012 \
RAC_RUN_NAME=20260820_090530 \
bash scripts/plot_training_progress.sh
```

`COMPARISON_NAME` là tùy chọn. Khi bỏ qua như trên, script tự đặt
`20260819_143012_vs_20260820_090530`. Chỉ truyền `COMPARISON_NAME=...` nếu muốn tên kết quả ngắn hoặc
dễ đọc hơn.

Kết quả:

```text
results/20260819_143012_vs_20260820_090530/training_eval_history.csv
results/20260819_143012_vs_20260820_090530/training_eval_history.json
results/20260819_143012_vs_20260820_090530/plots/accuracy_over_steps.png
results/20260819_143012_vs_20260820_090530/plots/loss_comparison.png
```

Giữ `TRAIN_EVAL_ENABLED=true` trong cả hai script để có `accuracy_over_steps.png`.

## 5. Final eval Base, TA-OPD và RAC

Có thể eval bằng một GPU, không phụ thuộc số GPU đã dùng để train:

```bash
TA_RUN_NAME=20260819_143012 \
RAC_RUN_NAME=20260820_090530 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval_all_b200.sh
```

Script tự đọc:

```text
outputs/20260819_143012/ta_opd/final
outputs/20260820_090530/rac_opd/final
```

Kết quả final eval và plots nằm trong:

```text
results/20260819_143012_vs_20260820_090530/
```

## 6. Resume một run dùng cấu hình mặc định mới

Ví dụ TA bị ngắt và checkpoint mới nhất là step 100. `MAX_STEPS` là tổng step mục tiêu, không phải
số step chạy thêm:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RESUME_FROM_CHECKPOINT=outputs/20260819_143012/ta_opd/checkpoint-000100 \
MAX_STEPS=2175 \
bash scripts/train_ta_b200.sh
```

RAC tương tự:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RESUME_FROM_CHECKPOINT=outputs/20260820_090530/rac_opd/checkpoint-000100 \
MAX_STEPS=2175 \
bash scripts/train_rac_b200.sh
```

Khi có `RESUME_FROM_CHECKPOINT`, script tự suy ra `OUTPUT_DIR` là parent của checkpoint và append
vào đúng run cũ. Checkpoint phải có cả `config.json` và `optimizer.pt`.

## 7. Resume từ ba GPU xuống hai GPU

Có thể đổi số GPU. Ví dụ hôm trước TA chạy trên `0,1,2`, hôm sau chỉ còn `0,1`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
RESUME_FROM_CHECKPOINT=outputs/20260819_143012/ta_opd/checkpoint-000100 \
MAX_STEPS=2175 \
bash scripts/train_ta_b200.sh
```

RAC:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
RESUME_FROM_CHECKPOINT=outputs/20260820_090530/rac_opd/checkpoint-000100 \
MAX_STEPS=2175 \
bash scripts/train_rac_b200.sh
```

Phải giữ nguyên global batch 8, rollout 2048 và các scientific hyperparameter của run. DDP chỉ đổi cách chia batch
cục bộ: ba GPU nhận khoảng `3/3/2`, còn hai GPU nhận `4/4`; global update vẫn có cùng định nghĩa.

## 8. Resume checkpoint cũ được train với batch 64

Checkpoint cũ phải dùng lại đúng hyperparameter cũ, không dùng các default mới 8/4096/M=4. Ví dụ
checkpoint TA cũ dùng batch 64, rollout 256 token và branch setting 2:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_BATCH_SIZE=64 \
MAX_NEW_TOKENS=256 \
ROLLOUT_VLLM_MAX_MODEL_LEN=1024 \
BRANCH_M=2 \
RESUME_FROM_CHECKPOINT=outputs/ta_opd/checkpoint-000100 \
MAX_STEPS=272 \
bash scripts/train_ta_b200.sh
```

RAC cũ:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_BATCH_SIZE=64 \
MAX_NEW_TOKENS=256 \
ROLLOUT_VLLM_MAX_MODEL_LEN=1024 \
BRANCH_M=2 \
RESUME_FROM_CHECKPOINT=outputs/rac_opd/checkpoint-000100 \
MAX_STEPS=272 \
bash scripts/train_rac_b200.sh
```

Resume validator sẽ dừng nếu batch, method, LR, rho, K/M hoặc generation length khác checkpoint.
Không dùng `RESUME_ALLOW_CONFIG_MISMATCH=true` chỉ để vượt qua lỗi này; hãy đặt lại đúng giá trị của
run cũ.

Nếu log hiện tại đã chứa step lớn hơn checkpoint được chọn, trainer cũng từ chối append để tránh ghi
trùng hoặc rewind lịch sử. Khi đó cần dùng checkpoint mới nhất hoặc một `OUTPUT_DIR` mới.
