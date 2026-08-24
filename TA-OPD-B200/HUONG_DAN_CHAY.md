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
CUDA_VISIBLE_DEVICES=0,1 METHOD=rac bash scripts/smoke_test_fsdp_multigpu.sh
```

Hãy đặt các biến dưới đây trước khi chạy các lệnh kiểm tra phía trên. Model và train dataset có thể
chọn ngay bằng biến môi trường; không cần sửa YAML. Default hiện tại là
Qwen3-8B → Qwen3-1.7B-Base trên Competition-MATH:

```bash
export TEACHER_MODEL_PATH=models/Qwen3-8B
export STUDENT_MODEL_PATH=nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base
export TRAIN_DATASET=competition_math
```

Chuyển sang full DAPO-Math-17k-Processed:

```bash
export TRAIN_DATASET=dapo_math
```

Preset DAPO tự đặt path `nlp/minhpn19/data/DAPO-Math-17k-Processed`, split `all`, prompt key
`prompt`; preset Competition-MATH tự đặt file train, split `null`, prompt key `problem`. Dataset
khác dùng `TRAIN_DATASET=custom TRAIN_DATA_PATH=... TRAIN_DATA_SPLIT=... TRAIN_PROMPT_KEY=...`.
Có thể sửa các default tập trung trong `scripts/common_b200.sh`. Mỗi model/data selection và số
GPU tự có `results/preflight/asset-<fingerprint>-<N>gpu.json`; chỉ cần chạy
`smoke_test_b200.sh` lần đầu cho selection/topology mới. Các workload 1/2/4 GPU không ghi đè report
của nhau; autotune cũng được lưu riêng theo cả fingerprint và số GPU. `smoke_test_b200.sh` ẩn GPU
trong lúc chạy unit tests để không tự spawn thêm rank; smoke FSDP thật chạy riêng bằng:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 METHOD=opd bash scripts/smoke_test_fsdp_multigpu.sh
```

Student tokenizer là tokenizer duy nhất. Teacher nhận trực tiếp cùng prompt/trajectory token IDs,
không decode rồi tokenize lại. `special_tokens_map` được phép khác, nhưng `get_vocab()`, added vocab,
model vocab size và tokenizer length vẫn bắt buộc khớp tuyệt đối. Prompt chung luôn dùng
`enable_thinking=false`, nên teacher luôn no-think.

## 2. Chạy full và công bằng

Đặt ba tên riêng nhưng dùng cùng suffix và cùng mọi shared hyperparameter:

```bash
PAIR=$(date +%Y%m%d_%H%M%S)
export OPD_RUN_NAME="opd_qwen3_8b_to_1p7b_base_${PAIR}"
export TA_RUN_NAME="ta_qwen3_8b_to_1p7b_base_${PAIR}"
export RAC_RUN_NAME="rac_bellman_qwen3_8b_to_1p7b_base_${PAIR}"

RUN_NAME="$OPD_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_opd_b200.sh

RUN_NAME="$TA_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_ta_b200.sh

RUN_NAME="$RAC_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_rac_b200.sh
```

Không đặt `MAX_STEPS` (hoặc để `-1`) để dùng toàn bộ epoch/full Competition-MATH train. OPD dùng uniform weight `1`
trên mọi valid response token. Cả ba method dùng cùng vLLM rollout và eval step 0 / mỗi 50 step /
final. Nếu cần hạ memory, đặt cùng giá trị cho cả ba lệnh.

Production defaults: FSDP `FULL_SHARD`, global batch 64, `n=1`, 32 trajectory/GPU,
`MICRO_BATCH_SIZE_PER_GPU=8`, LR `1e-6`. Fast defaults đã bật sẵn: crop sau EOS, length bucketing,
response-only Qwen3 logits, `SCORE_MICRO_BATCH_SIZE=8`, vLLM throughput/chunked-prefill/async và
joint scoring hai-forward cho TA/RAC. Tuning theo thứ tự `8 → 16 → 4`, luôn giữ global
`BATCH_SIZE=64`, `NUM_RESPONSES=1`. Sau khi chọn, phải dùng đúng cùng giá trị cho cả ba method.
Xem giải thích và
lệnh đầy đủ trong mục “Fast path không đổi protocol” của `RUN_B200.md`.

```bash
BATCH_SIZE=64 MICRO_BATCH_SIZE_PER_GPU=4 NUM_RESPONSES=1 \
  RUN_NAME="$OPD_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_opd_b200.sh
BATCH_SIZE=64 MICRO_BATCH_SIZE_PER_GPU=4 NUM_RESPONSES=1 \
  RUN_NAME="$TA_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_ta_b200.sh
BATCH_SIZE=64 MICRO_BATCH_SIZE_PER_GPU=4 NUM_RESPONSES=1 \
  RUN_NAME="$RAC_RUN_NAME" CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_rac_b200.sh
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

## 4. TensorBoard

```bash
tensorboard --logdir_spec \
  "OPD:outputs/$OPD_RUN_NAME/opd/tensorboard,TA:outputs/$TA_RUN_NAME/ta_opd/tensorboard,RAC:outputs/$RAC_RUN_NAME/rac_opd/tensorboard" \
  --bind_all --port 6006
```

## 5. Eval và plot

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

Vẽ riêng avg@16 của một method trên cả bốn bộ eval (bốn đường trong cùng một ảnh):

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

Hoặc dùng một launcher chung cho checkpoint bất kỳ của OPD, TA-OPD hoặc RAC:

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_TEMPERATURE=1 EVAL_NUM_RESPONSES=1 \
  bash scripts/eval_checkpoint_b200.sh opd \
  outputs/my_run/opd/checkpoint-000050 \
  results/checkpoint_eval/opd_step50

# Thay opd bằng ta-opd hoặc rac và thay đường dẫn checkpoint tương ứng.
```

Nếu bỏ đối số output cuối, script tự tạo thư mục timestamp. Kết quả có `summary.json`, các file
prediction theo dataset và `model_outputs_detailed.jsonl.gz` chứa prompt, đáp án chuẩn, toàn bộ
response cùng cờ đúng/sai của từng response.

Periodic và final eval luôn chạy toàn bộ Competition-MATH test, MATH-500, AIME24, AIME25; `max_num_seqs` là concurrency,
không phải subset size.
Bốn bộ có tổng cộng 1.060 problem; `n=16` nên vLLM báo 16.960 processed responses. Base step 0 chỉ
generate một lần rồi dùng cùng artifact đã kiểm tra fingerprint cho OPD/TA-OPD/RAC. Những step
sau vẫn phải generate riêng vì weights của ba method đã khác nhau.

Nếu chỉ cần một kết quả cho mỗi problem, thêm
`TRAIN_EVAL_NUM_RESPONSES=1 TRAIN_EVAL_TEMPERATURE=1` khi train. Khi đó mỗi checkpoint chỉ có 1.060
generation và metric/biểu đồ được ghi đúng là `accuracy`. Eval final dùng
`EVAL_NUM_RESPONSES=1`; re-eval checkpoint dùng `REEVAL_NUM_RESPONSES=1`.

Eval lại toàn bộ checkpoint đã lưu với vLLM `n=16`, `temperature=0.7`, `top_p=0.95`, metric
`avg@16`, đồng thời ghi đè lịch sử/file eval cũ của OPD, TA-OPD và RAC:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  REEVAL_DRY_RUN=true CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_all_checkpoints_b200.sh

OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Dòng đầu chỉ kiểm tra danh sách; dòng thứ hai mới thực sự ghi đè. Script mặc định eval lại cả base
step 0 để không trộn protocol eval giữa baseline và checkpoint, nhưng chỉ generate base một lần.
