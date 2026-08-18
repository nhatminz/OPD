# TA-OPD vs RAC — standalone NVIDIA B200 project

Đây là project độc lập để chạy thí nghiệm `Qwen3-4B -> Qwen3-1.7B` trên B200. Project không
import, symlink hay đọc bất kỳ file nào từ folder `TA-OPD` cũ. Logic TA được port theo mã nguồn công
khai [wyy-code/TA-OPD](https://github.com/wyy-code/TA-OPD); RAC chỉ thay token selector, không thay
rollout hay OPD loss.

## Vị trí và dữ liệu trên máy B200

Sau khi copy, project phải ở:

```text
/workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
```

Các path cố định trong `configs/qwen3_b200_base.yaml`:

| Asset | Path |
|---|---|
| Teacher | `/workspace/storage-shared/models/Qwen3-4B` |
| Student/base | `/workspace/storage-shared/models/Qwen3-1.7B` |
| Full DAPO | `/workspace/storage-shared/nlp/minhpn19/data/DAPO-Math-17k-Processed` |
| MATH-500 | `/workspace/storage-shared/nlp/minhpn19/data/eval/math500` |
| AIME24 | `/workspace/storage-shared/nlp/minhpn19/data/eval/aime24/test-00000-of-00001.parquet` |
| AIME25 | `/workspace/storage-shared/nlp/minhpn19/data/eval/aime25/test.jsonl` |

Loader đọc toàn bộ split `all` (17k); nó không cộng thêm các mirror `cn`/`en` gây trùng dữ liệu.
Các row được shuffle xác định theo từng epoch và không pad/lặp batch cuối. Khi
`training.max_steps: null`, một epoch đi qua toàn bộ dataset đúng một lần.

## Cài đặt và smoke/autotune bắt buộc

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
# Python 3.10+; Python 3.11 is recommended.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_test_b200.sh
```

`requirements.txt` đã chứa cả PyTorch/Transformers, thư viện đọc parquet, plotting, `math-verify`
và `vllm`; venv mới không cần cài thêm package thủ công.

Không có batch size “cuối cùng” được đoán trước trên máy phát triển. Smoke test thực hiện:

<!-- B200_AUTOTUNE_RESULT_START -->
**Measured target result:** chưa chạy trên máy B200; `scripts/smoke_test_b200.sh` sẽ cập nhật block này.
<!-- B200_AUTOTUNE_RESULT_END -->

1. unit tests độc lập;
2. xác minh đúng B200, tokenizer/vocabulary và các model path;
3. đọc toàn bộ DAPO và ghi số dòng/schema;
4. đọc đúng AIME24 parquet và AIME25 `test.jsonl`, tự suy ra schema rồi lưu ba ví dụ/ground truth;
5. binary-search danh sách batch `[16,32,64,96,128,160,192]` bằng một optimizer step TA và RAC cho mỗi
   candidate được thử;
6. chỉ nhận batch khi loss hữu hạn, rollout hash giống nhau, số token chọn bằng nhau và peak reserved
   memory không vượt 90% VRAM.

Mặc định có tối đa ba candidate, tức tối đa ba bước TA và ba bước RAC. Có thể đổi phạm vi hợp lý:

```bash
BATCH_CANDIDATES="16 32 64 96 128 160 192" bash scripts/smoke_test_b200.sh
```

Kết quả được lưu vào:

```text
results/preflight.json
outputs/autotune/batch_autotune.json
configs/qwen3_b200_autotuned.yaml
B200_VALIDATION.md
```

Hai script full training từ chối chạy nếu preflight/config autotune chưa tồn tại hoặc không hợp lệ.
Smoke test không tự khởi chạy full training và luôn tắt periodic evaluation trong các bước dò batch.

## Training

Config mặc định train đúng **1 epoch**. Vì `training.max_steps: null`, số optimizer step thực tế là
`ceil(17.398 / batch_size_autotuned)`. Ví dụ batch 64 tạo 272 step.

Chạy tuần tự cả TA và RAC rồi tự động vẽ các đường accuracy/loss:

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
CUDA_VISIBLE_DEVICES=0 bash scripts/train_all_b200.sh
```

Các tham số thường đổi (`EPOCHS`, LR, `rho`, rollout length, K/M, checkpoint interval, số lần eval và
các giới hạn memory/concurrency của vLLM) nằm trong block `BASIC PARAMETERS: EDIT HERE` ngay đầu
`scripts/train_all_b200.sh`. Có thể sửa trực tiếp block đó hoặc override bằng biến môi trường.

Hoặc chạy TA-OPD riêng:

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
CUDA_VISIBLE_DEVICES=0 bash scripts/train_ta_b200.sh
```

RAC:

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rac_b200.sh
```

Hai wrapper dùng cùng generated batch/micro-batch, seed, full DAPO order, epoch, LR, optimizer,
rollout, K, rho và OPD objective. Những override chung nên truyền giống nhau cho cả hai, ví dụ:

```bash
EPOCHS=2 LEARNING_RATE=1e-5 RHO=0.10 MAX_NEW_TOKENS=256 \
  bash scripts/train_ta_b200.sh
EPOCHS=2 LEARNING_RATE=1e-5 RHO=0.10 MAX_NEW_TOKENS=256 \
  bash scripts/train_rac_b200.sh
```

### Evaluation trong lúc train

Mặc định mỗi run có **16 lần eval**, được trải đều sau khi code biết `max_steps` thực tế:

```text
step 0 (Base Qwen3-1.7B chưa update)
14 mốc gần cách đều trong quá trình train
step cuối của epoch
```

Với ví dụ batch 64/272 step, lịch gồm đúng 16 mốc từ `0` đến `272`, gap 18–19 step. Mỗi mốc chạy đủ
MATH-500, AIME24 và AIME25 bằng greedy decoding (`temperature=0`, `max_new_tokens=2048`). Backend
mặc định là vLLM: ba benchmark được gom vào **một** lệnh generate/một thanh tiến trình. Step 0 đọc
base model; step có checkpoint tái sử dụng checkpoint; mốc còn lại lưu một snapshot tạm của student,
chạy evaluator trong subprocess rồi xóa snapshot. Cách này trả toàn bộ engine/KV-cache cho GPU trước
khi train tiếp và không làm thay đổi model/optimizer/RNG của process training. Thời gian eval được ghi
riêng, không cộng vào training step time.

Periodic eval luôn có `limit=null`, tức chạy **toàn bộ sample của cả ba dataset**, không dùng subset.
`max_num_seqs=256` chỉ là mức concurrency của scheduler, không phải số sample được eval. vLLM được
đặt `gpu_memory_utilization=auto`: evaluator đo VRAM còn trống tại mỗi mốc rồi dùng gần như toàn bộ,
chỉ chừa mặc định 4 GiB cho CUDA workspace. Phần VRAM student, teacher và optimizer đang giữ được
tính trực tiếp vào lượng đã dùng, thay vì áp một giới hạn phần trăm cố định.

Có thể đổi số lần mà vẫn giữ giống nhau cho TA/RAC:

```bash
TRAIN_EVAL_TARGET=16 TRAIN_EVAL_BACKEND=vllm \
  bash scripts/train_all_b200.sh
```

Nếu cần lịch cố định, đặt `TRAIN_EVAL_INTERVAL=100`; biến này ưu tiên hơn `TRAIN_EVAL_TARGET`. Có thể
fallback về evaluator Transformers bằng `TRAIN_EVAL_BACKEND=hf` (`TRAIN_EVAL_BATCH_SIZE` chỉ dùng
cho backend này). Tắt bằng `TRAIN_EVAL_ENABLED=false`. Không nên dùng giá trị khác nhau giữa TA/RAC.

Config dùng bf16, FlashAttention 2 khi environment hỗ trợ và tự fallback sang SDPA, fused AdamW,
full-parameter Qwen3-1.7B training, không gradient accumulation sau autotune (`micro_batch=batch`).
Teacher luôn frozen. RAC dùng exact full-vocabulary `Delta`, batched top-M branches, KV-cache reuse và
mọi counterfactual probe nằm trong `torch.inference_mode()`.

Mọi lệnh train TA/RAC đều có tqdm cho setup và optimizer step, kèm stage hiện tại, loss, số token
được chọn, thời gian selector và VRAM. Mọi lệnh eval đều có tqdm theo sample trong lúc generate và
chấm đáp án; vLLM vẫn gom cả ba benchmark trong một lần generate để terminal không bị tràn log.

## Logs

Mỗi run tạo:

```text
outputs/ta_opd/metrics.jsonl
outputs/ta_opd/selector_scores/selected_steps_*.jsonl.gz
outputs/ta_opd/eval_history.jsonl
outputs/ta_opd/training_eval/step-*/summary.json
outputs/rac_opd/metrics.jsonl
outputs/rac_opd/selector_scores/selected_steps_*.jsonl.gz
outputs/rac_opd/eval_history.jsonl
outputs/rac_opd/training_eval/step-*/summary.json
```

`metrics.jsonl` chứa loss, selector/RAC counterfactual/step time, tokens/s, peak allocated/reserved
GPU memory, selected fraction và rollout hash theo step. File gzip chỉ chứa token đã chọn và được ghi
tăng dần theo chunk 50 step. TA lưu `D,C,D_norm,C_norm,s_TA`; RAC lưu `Delta,A,F,B,s_RAC`, kèm
sample ID, dataset index, response position, token ID và token text.

Khi run thứ hai hoàn tất, training wrapper tự đọc hai `eval_history.jsonl` và sinh:

```text
results/training_eval_history.csv
results/training_eval_history.json
results/plots/accuracy_over_steps.png
results/plots/loss_comparison.png
```

`accuracy_over_steps.png` có ba subplot MATH-500/AIME24/AIME25, đường TA và RAC theo optimizer step,
và đường ngang Base Qwen3-1.7B lấy từ step 0. Có thể vẽ lại thủ công bằng
`bash scripts/plot_training_progress.sh`.

## Evaluation và plots

Chạy đúng ba model Base/TA/RAC trên MATH-500, AIME24 và AIME25:

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_all_b200.sh
```

Các script eval cuối cũng dùng vLLM mặc định và một thanh tiến trình cho cả ba benchmark; dùng
`EVAL_BACKEND=hf` nếu cần fallback.

Hoặc chạy riêng:

```bash
bash scripts/eval_base_b200.sh
TA_CHECKPOINT=/path/to/ta/checkpoint bash scripts/eval_ta_b200.sh
RAC_CHECKPOINT=/path/to/rac/checkpoint bash scripts/eval_rac_b200.sh
bash scripts/plot_results.sh
```

Output cuối:

```text
results/comparison.csv
results/comparison.json
results/eval/{base,ta_opd,rac}/*_predictions.jsonl.gz
results/plots/accuracy_comparison.png
results/plots/accuracy_over_steps.png
results/plots/loss_comparison.png
```

Loader không giả định schema cũ. Nó ưu tiên exact B200 filename, kiểm tra columns và hỗ trợ cấu hình
`question_key`/`answer_key` nếu schema máy đích dùng tên hoàn toàn khác các alias thông dụng.

## Fairness invariant

```text
TA:  s_TA  = D_norm * C_norm
RAC: s_RAC = Delta * A * F
```

Sau khi selector tạo shared global mask `ceil(rho*N_valid)`, cả hai đi qua cùng sampled reverse-KL
advantage và PPO-clipped OPD policy loss. RAC không thay token rollout và counterfactual token không
đi vào loss.

Project hiện tối ưu cho một process trên B200 đầu tiên trong `CUDA_VISIBLE_DEVICES`; không hard-code
physical GPU ID. Mỗi process chứa cả student và frozen teacher để giữ exact selector distributions.
