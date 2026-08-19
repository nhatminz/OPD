# TA-OPD vs RAC — standalone NVIDIA B200 project

Hướng dẫn lệnh chạy mới nhất, bao gồm timestamp tự động và resume, nằm trong
[`HUONG_DAN_CHAY.md`](HUONG_DAN_CHAY.md).

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

## Cài đặt và smoke/preflight bắt buộc

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
# Python 3.10+; Python 3.11 is recommended.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_test_b200.sh
```

`requirements.txt` đã chứa cả PyTorch/Transformers, thư viện đọc parquet, plotting, `math-verify`,
`requests` và vLLM `0.17.x`; venv mới không cần cài thêm package thủ công. Version vLLM được giữ ở
`>=0.17.1,<0.18` vì training dùng trực tiếp API CUDA-IPC weight transfer và sleep mode của dòng này.

Batch train là hyperparameter **global** cố định, mặc định **8**, được khai báo bằng
`TRAIN_BATCH_SIZE` ngay đầu `scripts/train_ta_b200.sh` và `scripts/train_rac_b200.sh`. Cùng một giá
trị được áp vào rollout batch và micro-batch của TA/RAC, nên gradient accumulation luôn bằng 1. Khi dùng nhiều GPU, code tự chia
batch global thành các shard liên tiếp cân bằng (ví dụ 8 trên 3 GPU thành 3/3/2); không nhân batch
với số GPU và không đổi số optimizer step. Smoke test mặc định thực hiện:

<!-- B200_AUTOTUNE_RESULT_START -->
**Measured target result:** chưa chạy trên máy B200; `scripts/smoke_test_b200.sh` sẽ cập nhật block này.
<!-- B200_AUTOTUNE_RESULT_END -->

1. unit tests độc lập;
2. xác minh đúng B200, tokenizer/vocabulary và các model path;
3. đọc toàn bộ DAPO và ghi số dòng/schema;
4. đọc đúng AIME24 parquet và AIME25 `test.jsonl`, tự suy ra schema rồi lưu ba ví dụ/ground truth;
5. xác minh package vLLM đã được cài trong đúng venv;
6. ghi báo cáo preflight nhưng không khởi chạy full training.

Kết quả bắt buộc được lưu ở `results/preflight.json`. Full training chỉ cần preflight này, không yêu
cầu generated autotune config. Đổi batch trực tiếp trong bash hoặc bằng biến môi trường:

```bash
RUN_NAME=batch32_test TRAIN_BATCH_SIZE=32 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_ta_b200.sh
```

Autotune cũ vẫn có thể bật chủ động nếu muốn tìm batch lớn nhất cùng chạy được cho TA và RAC:

```bash
USE_BATCH_AUTOTUNE=true \
  BATCH_CANDIDATES="16 32 64 96 128 160 192" \
  CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_test_b200.sh

RUN_NAME=autotuned_run USE_BATCH_AUTOTUNE=true CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_ta_b200.sh
```

Chế độ tùy chọn này mới tạo `outputs/autotune/batch_autotune.json`,
`configs/qwen3_b200_autotuned.yaml` và `B200_VALIDATION.md`. Smoke test không tự khởi chạy full train.

## Training

Config mặc định train đúng **1 epoch**. Vì `training.max_steps: null`, số optimizer step thực tế là
`ceil(17.398 / TRAIN_BATCH_SIZE)`. Batch mặc định 8 tạo 2175 step.

Mỗi method có một launcher độc lập. Dùng cùng `RUN_NAME` để ghép thành một cặp so sánh:

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
RUN_NAME=run01 CUDA_VISIBLE_DEVICES=0 bash scripts/train_ta_b200.sh
RUN_NAME=run01 CUDA_VISIBLE_DEVICES=0 bash scripts/train_rac_b200.sh
```

Launcher tự đếm số phần tử trong `CUDA_VISIBLE_DEVICES`: một GPU chạy Python bình thường, từ hai GPU
trở lên tự chạy một DDP worker trên mỗi GPU. Không cần sửa config hay hard-code world size:

```bash
# TA trên 2 B200
RUN_NAME=run01 CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_ta_b200.sh

# RAC trên 3 B200; vẫn global batch 8, cùng LR/rho/số step
RUN_NAME=run01 CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_rac_b200.sh
```

Tên biến môi trường phân biệt hoa/thường và phải viết đúng `CUDA_VISIBLE_DEVICES`. Mọi GPU visible đều
được preflight xác nhận là B200. Multi-GPU giữ nguyên cùng global DAPO batch/order, seed theo global
sample, TA quantile normalization trên toàn global rollout, một global top-budget
`ceil(rho*N_valid)`, OPD loss trung bình theo global batch và DDP all-reduce gradient. Batch cuối không
bị pad/lặp; code scale gradient đúng cả khi shard cuối không đều. Sai số floating-point rất nhỏ do thứ
tự all-reduce có thể khác single-GPU là thuộc tính chung của DDP; vì vậy không thể cam kết checkpoint
bitwise-identical hoặc accuracy hữu hạn tuyệt đối giống từng chữ số nếu chưa chạy thực nghiệm.

Các tham số thường đổi (`TRAIN_BATCH_SIZE`, `EPOCHS`, LR, `rho`, rollout length, rollout backend,
K/M, checkpoint interval, số lần eval và các giới hạn memory/concurrency của vLLM) nằm trong block
`PARAMETERS: EDIT HERE` ngay đầu hai script train. Có thể sửa trực tiếp block đó hoặc override bằng
biến môi trường.

### vLLM rollout trong lúc train

`ROLLOUT_BACKEND=vllm` là mặc định. Mỗi run giữ một vLLM server local duy nhất thay vì khởi tạo lại
engine ở mỗi step. Server được tạo bằng dummy load trước khi nạp student/teacher; trước **từng**
rollout, trainer thực hiện theo thứ tự:

```text
wake weights -> CUDA-IPC sync student hiện tại -> wake KV cache -> vLLM generate
-> sleep level 2 -> HF student/teacher scoring -> TA/RAC selector -> backward/update
```

Như vậy rollout luôn dùng đúng weights student ở đầu optimizer step, không phải checkpoint cũ.
Level-2 sleep loại bỏ cả weights lẫn KV cache phía vLLM trong lúc scoring/backward để nhường VRAM;
step sau mới cấp phát lại. Các request trong một batch được gửi đồng thời để scheduler vLLM dynamic
batching, và tqdm hiển thị một thanh `vLLM rollout` theo số sample. Student log-prob dùng trong
PPO-clipped OPD loss vẫn được HF chấm lại trên chính rollout đó, nên objective TA/RAC không đổi.

Các knob rollout nằm ngay đầu bash:

```bash
ROLLOUT_BACKEND=vllm
ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION=0.25
ROLLOUT_VLLM_MAX_NUM_SEQS=8
ROLLOUT_VLLM_MAX_MODEL_LEN=4096
ROLLOUT_VLLM_MAX_CONCURRENT_REQUESTS=8
ROLLOUT_VLLM_WAKE_HEADROOM_GIB=2
```

`gpu_memory_utilization=0.25` là phần VRAM dành cho engine đồng vị trí khi nó đang generate, không
phải giới hạn số sample. Trên B200 180 GB giá trị này dành khoảng 45 GB cho weights/KV scheduler;
student, teacher và optimizer vẫn cùng resident. Nếu cần debug hoặc environment chưa hỗ trợ IPC,
fallback không đổi objective bằng `ROLLOUT_BACKEND=hf bash scripts/train_ta_b200.sh` (hoặc RAC).
Trainer chỉ gọi `torch.cuda.empty_cache()` khi VRAM thật sự còn trống không đủ để wake engine cộng
thêm headroom nói trên; bình thường nó giữ CUDA cache để tránh mất tốc độ ở mọi step.

Workflow khuyến nghị là chạy từng method độc lập rồi chủ động plot/eval. Hai file
`train_ta_b200.sh` và `train_rac_b200.sh` đều có block `PARAMETERS: EDIT HERE` ở đầu file cho các
tham số cơ bản. Fresh run tự tạo `RUN_NAME=YYYYMMDD_HHMMSS`; vì hai method chạy độc lập, dùng
`TA_RUN_NAME` và `RAC_RUN_NAME` để chọn cặp khi plot/eval. Có thể truyền cùng một `RUN_NAME` thủ công
nếu muốn hai method nằm chung một namespace. Layout mặc định:

```text
outputs/<RUN_NAME>/ta_opd
outputs/<RUN_NAME>/rac_opd
results/<RUN_NAME>
```

Ví dụ chạy mới TA-OPD riêng trên ba GPU:

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
RUN_NAME=exp01_b64_lr1e-5_rho010 CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/train_ta_b200.sh
```

Sau khi TA kết thúc, chạy RAC độc lập với cùng tên run và cùng các shared hyperparameter:

```bash
RUN_NAME=exp01_b64_lr1e-5_rho010 CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/train_rac_b200.sh
```

Hai script chỉ train và log, không tự chạy method còn lại hoặc tự vẽ hình. Khi cả hai đã xong, vẽ
loss và accuracy trong lúc train bằng:

```bash
RUN_NAME=exp01_b64_lr1e-5_rho010 bash scripts/plot_training_progress.sh
```

Sau đó chạy final eval Base/TA/RAC và vẽ comparison cuối:

```bash
RUN_NAME=exp01_b64_lr1e-5_rho010 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval_all_b200.sh
```

Mỗi wrapper vẫn ghi đầy đủ metrics, selector-token logs, checkpoints và periodic eval. Biến đặt trên
lệnh chạy được ưu tiên hơn giá trị mặc định trong block đầu file. Chẳng hạn có thể thử một cấu hình
khác mà không sửa file:

```bash
EPOCHS=2 LEARNING_RATE=1e-5 RHO=0.10 MAX_NEW_TOKENS=256 \
  bash scripts/train_ta_b200.sh
EPOCHS=2 LEARNING_RATE=1e-5 RHO=0.10 MAX_NEW_TOKENS=256 \
  bash scripts/train_rac_b200.sh
```

### Resume sau khi train bị ngắt

Checkpoint có `optimizer.pt` hỗ trợ true resume cho cả TA và RAC. Trainer khôi phục full student,
AdamW state và global optimizer step; data order/epoch và rollout seed tiếp tục theo step đó. Có thể
đổi số GPU khi resume vì checkpoint là model/optimizer replicated bình thường, trong khi configured
batch vẫn là global batch.

Ví dụ run mới `exp01_b8_lr1e-5_rho010` bị dừng ở checkpoint 100 khi đang dùng ba GPU. Hôm sau chỉ có
hai GPU, TA tiếp tục từ step 101 như sau:

```bash
RUN_NAME=exp01_b8_lr1e-5_rho010 \
CUDA_VISIBLE_DEVICES=0,1 \
RESUME_FROM_CHECKPOINT=outputs/exp01_b8_lr1e-5_rho010/ta_opd/checkpoint-000100 \
MAX_STEPS=2175 \
  bash scripts/train_ta_b200.sh

# RAC tương tự, dùng đúng checkpoint RAC
RUN_NAME=exp01_b8_lr1e-5_rho010 \
CUDA_VISIBLE_DEVICES=0,1 \
RESUME_FROM_CHECKPOINT=outputs/exp01_b8_lr1e-5_rho010/rac_opd/checkpoint-000100 \
MAX_STEPS=2175 \
  bash scripts/train_rac_b200.sh
```

Checkpoint theo layout cũ vẫn resume được; phải chỉ rõ output cũ để append đúng log:

```bash
CUDA_VISIBLE_DEVICES=0,1 TRAIN_BATCH_SIZE=64 MAX_NEW_TOKENS=256 BRANCH_M=2 \
OUTPUT_DIR=outputs/ta_opd \
RESUME_FROM_CHECKPOINT=outputs/ta_opd/checkpoint-000100 \
MAX_STEPS=272 \
  bash scripts/train_ta_b200.sh
```

`MAX_STEPS` là **total target**, không phải số step chạy thêm. Nếu vẫn dùng config một epoch/batch 64
trên đúng 17.398 mẫu thì có thể bỏ `MAX_STEPS=272`, trainer tự suy ra 272 và chạy 101..272.

Resume mặc định kiểm tra method và các hyperparameter khoa học trong `resolved_config.yaml`; TA
checkpoint không được dùng để tiếp tục RAC. Nó cũng từ chối append nếu `metrics.jsonl`,
`eval_history.jsonl` hoặc selector logs đã có step lớn hơn checkpoint, tránh âm thầm duplicate/rewind
log. Trong trường hợp đó hãy dùng checkpoint mới nhất, hoặc đặt một `OUTPUT_DIR` mới. Chỉ khi chủ động
muốn đổi thí nghiệm mới dùng `RESUME_ALLOW_CONFIG_MISMATCH=true`.

### Evaluation trong lúc train

Mặc định mỗi run có **16 lần eval**, được trải đều sau khi code biết `max_steps` thực tế:

```text
step 0 (Base Qwen3-1.7B chưa update)
14 mốc gần cách đều trong quá trình train
step cuối của epoch
```

Với batch mặc định 8/2175 step, lịch gồm đúng 16 mốc từ `0` đến `2175`, gap khoảng 145 step. Mỗi mốc chạy đủ
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
  bash scripts/train_ta_b200.sh
TRAIN_EVAL_TARGET=16 TRAIN_EVAL_BACKEND=vllm \
  bash scripts/train_rac_b200.sh
```

Nếu cần lịch cố định, đặt `TRAIN_EVAL_INTERVAL=100`; biến này ưu tiên hơn `TRAIN_EVAL_TARGET`. Có thể
fallback về evaluator Transformers bằng `TRAIN_EVAL_BACKEND=hf` (`TRAIN_EVAL_BATCH_SIZE` chỉ dùng
cho backend này). Tắt bằng `TRAIN_EVAL_ENABLED=false`. Không nên dùng giá trị khác nhau giữa TA/RAC.

Config dùng bf16, FlashAttention 2 khi environment hỗ trợ và tự fallback sang SDPA, fused AdamW,
full-parameter Qwen3-1.7B training, không gradient accumulation (`micro_batch=batch`).
Teacher luôn frozen. RAC dùng exact full-vocabulary `Delta`, batched top-M branches, KV-cache reuse và
mọi counterfactual probe nằm trong `torch.inference_mode()`. RAC tái sử dụng chính student top-K đã
tính cho TA cross-diagnostics, nên bỏ được một full-vocabulary top-K trùng lặp mà score/mask không đổi.
Với multi-GPU, exact RAC branches được chia theo global batch và chạy song song trên từng B200.

Mọi lệnh train TA/RAC đều có tqdm cho setup và optimizer step, kèm stage hiện tại, loss, số token
được chọn, thời gian selector và VRAM; rollout vLLM có thêm một thanh sample lồng bên trong. Mọi lệnh
eval đều có tqdm theo sample trong lúc generate và chấm đáp án; vLLM vẫn gom cả ba benchmark trong
một lần generate để terminal không bị tràn log.

## Logs

Mỗi run tạo:

```text
outputs/<RUN_NAME>/ta_opd/metrics.jsonl
outputs/<RUN_NAME>/ta_opd/vllm_rollout_server.rank-*.log
outputs/<RUN_NAME>/ta_opd/selector_scores/selected_steps_*.jsonl.gz
outputs/<RUN_NAME>/ta_opd/selector_scores/selected_steps_*_rank-*.jsonl.gz
outputs/<RUN_NAME>/ta_opd/eval_history.jsonl
outputs/<RUN_NAME>/ta_opd/training_eval/step-*/summary.json
outputs/<RUN_NAME>/rac_opd/metrics.jsonl
outputs/<RUN_NAME>/rac_opd/vllm_rollout_server.rank-*.log
outputs/<RUN_NAME>/rac_opd/selector_scores/selected_steps_*.jsonl.gz
outputs/<RUN_NAME>/rac_opd/selector_scores/selected_steps_*_rank-*.jsonl.gz
outputs/<RUN_NAME>/rac_opd/eval_history.jsonl
outputs/<RUN_NAME>/rac_opd/training_eval/step-*/summary.json
```

`metrics.jsonl` chứa loss, rollout backend, thời gian CUDA-IPC sync/vLLM generation/sleep,
selector/RAC counterfactual/step time, tokens/s, peak allocated/reserved GPU memory, selected fraction
và rollout hash theo step. File gzip chỉ chứa token đã chọn và được ghi
tăng dần theo chunk 50 step. TA lưu `D,C,D_norm,C_norm,s_TA`; RAC lưu `Delta,A,F,B,s_RAC`, kèm
sample ID, dataset index, response position, token ID và token text.

Khi cả hai run hoàn tất, lệnh plot độc lập đọc hai `eval_history.jsonl` và sinh:

```text
results/<RUN_NAME>/training_eval_history.csv
results/<RUN_NAME>/training_eval_history.json
results/<RUN_NAME>/plots/accuracy_over_steps.png
results/<RUN_NAME>/plots/loss_comparison.png
```

`accuracy_over_steps.png` có ba subplot MATH-500/AIME24/AIME25, đường TA và RAC theo optimizer step,
và đường ngang Base Qwen3-1.7B lấy từ step 0.

## Evaluation và plots

Chạy đúng ba model Base/TA/RAC trên MATH-500, AIME24 và AIME25:

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
RUN_NAME=exp01_b64_lr1e-5_rho010 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval_all_b200.sh
```

Checkpoint tạo bởi DDP là checkpoint Hugging Face bình thường, nên `eval_all_b200.sh` không cần biết
train đã dùng bao nhiêu GPU. Eval greedy trên một B200 như lệnh trên là lựa chọn luôn tương thích. Nếu
muốn tensor-parallel eval, expose các GPU và đặt rõ kích thước (chỉ dùng giá trị tương thích số
attention heads của model), ví dụ:

```bash
RUN_NAME=exp01_b64_lr1e-5_rho010 \
CUDA_VISIBLE_DEVICES=0,1 EVAL_VLLM_TENSOR_PARALLEL_SIZE=2 \
  bash scripts/eval_all_b200.sh
```

Các script eval cuối cũng dùng vLLM mặc định và một thanh tiến trình cho cả ba benchmark; dùng
`EVAL_BACKEND=hf` nếu cần fallback.

Hoặc chạy riêng:

```bash
RUN_NAME=exp01_b64_lr1e-5_rho010 bash scripts/eval_base_b200.sh
RUN_NAME=exp01_b64_lr1e-5_rho010 TA_CHECKPOINT=/path/to/ta/checkpoint \
  bash scripts/eval_ta_b200.sh
RUN_NAME=exp01_b64_lr1e-5_rho010 RAC_CHECKPOINT=/path/to/rac/checkpoint \
  bash scripts/eval_rac_b200.sh
RUN_NAME=exp01_b64_lr1e-5_rho010 bash scripts/plot_results.sh
```

Output cuối:

```text
results/<RUN_NAME>/comparison.csv
results/<RUN_NAME>/comparison.json
results/<RUN_NAME>/eval/{base,ta_opd,rac}/*_predictions.jsonl.gz
results/<RUN_NAME>/plots/accuracy_comparison.png
results/<RUN_NAME>/plots/accuracy_over_steps.png
results/<RUN_NAME>/plots/loss_comparison.png
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

Project dùng một DDP process và một vLLM rollout server đồng vị trí trên mỗi B200 trong
`CUDA_VISIBLE_DEVICES`; không hard-code physical GPU ID. Mỗi process chứa cả student và frozen teacher
để giữ exact selector distributions. Rank 0 là process duy nhất ghi `metrics.jsonl`, checkpoint,
summary và chạy periodic eval; selector-token gzip được tách theo rank để tránh concurrent corruption.
