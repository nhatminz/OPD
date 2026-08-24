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

Chạy smoke FSDP thật hai step trên hai GPU trước full run (đổi `METHOD` để kiểm tra từng method):

```bash
CUDA_VISIBLE_DEVICES=0,1 METHOD=opd bash scripts/smoke_test_fsdp_2gpu.sh
CUDA_VISIBLE_DEVICES=0,1 METHOD=ta  bash scripts/smoke_test_fsdp_2gpu.sh
CUDA_VISIBLE_DEVICES=0,1 METHOD=rac bash scripts/smoke_test_fsdp_2gpu.sh
```

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

export CUDA_VISIBLE_DEVICES=0,1
export DISTRIBUTED_STRATEGY=fsdp
export BATCH_SIZE=64
export MICRO_BATCH_SIZE_PER_GPU=8
export NUM_RESPONSES=1
export LR=1e-6
export NUM_EPOCHS=1
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=7168
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
cùng Top-K OPD core, vLLM rollout, data order, model, seed, batch/micro-batch, LR, optimizer,
checkpoint và lịch
eval mặc định step 0 / mỗi 50 step / final. Nên chạy tuần tự trên cùng GPU layout để tránh nhiễu
tài nguyên giữa các run.

Full eval có 560 problem và `n=16`, vì vậy progress của vLLM hiển thị 8.960 generated responses
(`560 * 16`); đây không phải rollout train `n=1`. Để không lặp lại lượt base tốn thời gian, step 0
được cache theo fingerprint của model, dataset, evaluator và toàn bộ protocol. OPD sinh lần đầu;
TA-OPD/RAC copy đúng cùng prediction. Một run bị lỗi trước step train đầu tiên cũng tái sử dụng
`training_eval/step-000000` hợp lệ khi chạy lại. Có thể tắt bằng
`TRAIN_EVAL_REUSE_BASE=false`, hoặc đổi chỗ lưu qua `TRAIN_EVAL_BASE_CACHE_DIR`.

Không export `MAX_STEPS`, hoặc đặt `MAX_STEPS=-1`, để consume full configured epoch. `MAX_STEPS=N`
chỉ dành cho debug/explicit total target; nó không phải số step chạy thêm sau resume.

Một override hợp lệ phải áp giống nhau cho cả ba method.

```bash
LR=5e-7 BATCH_SIZE=4 EVAL_INTERVAL=40 \
RUN_NAME="$OPD_RUN_NAME" bash scripts/train_opd_b200.sh
LR=5e-7 BATCH_SIZE=4 EVAL_INTERVAL=40 \
RUN_NAME="$TA_RUN_NAME" bash scripts/train_ta_b200.sh
LR=5e-7 BATCH_SIZE=4 EVAL_INTERVAL=40 \
RUN_NAME="$RAC_RUN_NAME" bash scripts/train_rac_b200.sh
```

Các OOM knob chính là `MICRO_BATCH_SIZE_PER_GPU`, `SCORE_MICRO_BATCH_SIZE`, `MAX_RESPONSE_LENGTH`,
`ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION`, `ROLLOUT_VLLM_MAX_NUM_SEQS` và
`ROLLOUT_VLLM_MAX_MODEL_LEN`. Chưa có peak-memory measurement trên B200 local, nên không coi default
là fit guarantee.

## Fast path không đổi protocol

Các launcher hiện mặc định bật cùng một nhóm tối ưu chính xác cho cả ba method:

- vLLM rollout persistent, CUDA-IPC weight sync, prefix cache, chunked prefill, async scheduler và
  throughput scheduling;
- bỏ LM-head projection ở prompt positions của Qwen3 khi chỉ cần response logits;
- crop mọi suffix sau EOS và bucket trajectory theo response length ở scoring lẫn backward;
- scoring inference micro-batch 8 thay vì 1, đồng thời tái sử dụng `logsumexp` khi temperature bằng 1;
- TA/RAC lấy hai chiều student/teacher Top-K trong hai forward thay vì forward student lần thứ ba;
- không yêu cầu/serialize vLLM rollout token log-probs khi sync sanity check đang tắt.

Các mục này không đổi prompt, response, seed, sampling, Top-K OPD loss, selector hay optimizer step.
Có thể kiểm tra resolved config của một run tại `outputs/<run>/<method>/resolved_config.yaml`.

Trên đúng máy B200, speed knob có tác động lớn nhất tiếp theo là tăng trajectory micro-batch. Hãy
probe RAC (đường có peak memory lớn nhất) một step với cùng full length trước; mỗi probe phải dùng
tên output mới:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_NAME=probe_rac_mb8 MAX_STEPS=1 \
  TRAIN_EVAL_ENABLED=false BATCH_SIZE=64 NUM_RESPONSES=1 \
  MICRO_BATCH_SIZE_PER_GPU=8 SCORE_MICRO_BATCH_SIZE=8 \
  bash scripts/train_rac_b200.sh
```

Nếu peak trong `outputs/probe_rac_mb8/rac_opd/metrics.jsonl` còn xa giới hạn, thử
`MICRO_BATCH_SIZE_PER_GPU=16`; nếu OOM thì lùi về `4`. Không đổi global `BATCH_SIZE=64` trong quá
trình tuning này. Có thể tune `SCORE_MICRO_BATCH_SIZE` riêng sau khi chọn training micro-batch.
Micro-batch chỉ chia cùng global weighted objective thành các lượt accumulate; hãy khóa đúng cùng
hai giá trị đã chọn cho OPD, TA và RAC. Sai khác floating-point ở mức thứ tự cộng gradient vẫn có
thể xảy ra khi đổi micro-batch, vì vậy không trộn các giá trị giữa ba run trong một comparison.

DDP chỉ còn là regression/debug option rõ ràng:

```bash
DISTRIBUTED_STRATEGY=ddp CUDA_VISIBLE_DEVICES=0,1 \
  BATCH_SIZE=64 NUM_RESPONSES=1 MICRO_BATCH_SIZE_PER_GPU=8 \
  bash scripts/train_opd_b200.sh
```

## TensorBoard

Rank 0 ghi event vào `outputs/<run>/<method>/tensorboard`. Theo dõi riêng một run:

```bash
tensorboard --logdir "outputs/$RAC_RUN_NAME/rac_opd/tensorboard" \
  --bind_all --port 6006
```

So sánh ba run trong cùng giao diện:

```bash
tensorboard --logdir_spec \
  "OPD:outputs/$OPD_RUN_NAME/opd/tensorboard,TA:outputs/$TA_RUN_NAME/ta_opd/tensorboard,RAC:outputs/$RAC_RUN_NAME/rac_opd/tensorboard" \
  --bind_all --port 6006
```

Production event chỉ có loss/grad norm/LR, ba OPD diagnostic, response length/EOS, bốn system
metric; TA thêm selected-token fraction, RAC thêm normalized effective-token fraction. Debug
vLLM/HF log-prob MAE chỉ xuất hiện khi chủ động bật sanity validation. Mọi giá trị đã được reduce
toàn cục trước khi rank 0 ghi.

Full avg@16 trên 560 problem bắt buộc sinh 8.960 response cho mỗi checkpoint khác nhau; không thể
giảm con số này mà vẫn giữ nguyên metric. Nếu ưu tiên thời gian train không bị chặn bởi eval, có thể
lưu đúng mỗi 50 step, tắt eval inline, rồi chạy evaluator trên các checkpoint sau train (kết quả
checkpoint không đổi, nhưng về mặt vận hành đây không còn là eval đồng bộ bên trong train):

```bash
PAIR=$(date +%Y%m%d_%H%M%S)
export OPD_RUN_NAME="opd_qwen3_4b_to_1p7b_${PAIR}"
export TA_RUN_NAME="ta_qwen3_4b_to_1p7b_${PAIR}"
export RAC_RUN_NAME="rac_bellman_qwen3_4b_to_1p7b_${PAIR}"
export TRAIN_EVAL_TEMPERATURE=1
TRAIN_EVAL_ENABLED=false SAVE_INTERVAL=50 bash scripts/train_all_b200.sh

REEVAL_TEMPERATURE="$TRAIN_EVAL_TEMPERATURE" \
  OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  CUDA_VISIBLE_DEVICES=0 bash scripts/reeval_all_checkpoints_b200.sh
```

Step 0 vẫn chỉ generate một lần nhờ shared fingerprint cache. Không đổi training `NUM_RESPONSES=1`,
`TRAIN_EVAL_NUM_RESPONSES=16` hoặc `MAX_RESPONSE_LENGTH` nếu mục tiêu là so sánh đúng protocol hiện
tại; các thay đổi đó nhanh hơn nhưng là thí nghiệm khác.

### Accuracy một response thay cho avg@16

Nếu không cần avg@16, đặt `TRAIN_EVAL_NUM_RESPONSES=1`. Mỗi checkpoint chỉ sinh một response cho
mỗi problem, tức 560 generation trên ba bộ thay vì 8.960. Output, CSV và biểu đồ tự ghi nhãn
`accuracy`, không còn ghi nhầm `avg@16`:

```bash
TRAIN_EVAL_NUM_RESPONSES=1 TRAIN_EVAL_TEMPERATURE=1 \
  bash scripts/train_all_b200.sh
```

Eval final thủ công dùng `EVAL_NUM_RESPONSES=1`; eval lại và ghi đè mọi checkpoint dùng
`REEVAL_NUM_RESPONSES=1`:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval_all_b200.sh

OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  REEVAL_NUM_RESPONSES=1 REEVAL_TEMPERATURE=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Với temperature 1 đây là sampled accuracy@1 theo seed cố định, không phải greedy decoding. Chế độ
n=1 nhanh hơn nhiều nhưng có variance lớn hơn n=16; phải dùng cùng n/temperature/top-p/seed cho cả
ba method.

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

Một launcher chung có thể eval checkpoint bất kỳ của cả ba method. Đối số thứ ba là thư mục
output và có thể bỏ qua (script sẽ tự tạo thư mục có timestamp):

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_TEMPERATURE=1 EVAL_NUM_RESPONSES=1 \
  bash scripts/eval_checkpoint_b200.sh opd \
  outputs/my_opd_run/opd/checkpoint-000050 \
  results/checkpoint_eval/opd_step50

CUDA_VISIBLE_DEVICES=1 EVAL_TEMPERATURE=1 EVAL_NUM_RESPONSES=16 \
  bash scripts/eval_checkpoint_b200.sh ta-opd \
  outputs/my_ta_run/ta_opd/checkpoint-000100

CUDA_VISIBLE_DEVICES=2 EVAL_TEMPERATURE=1 EVAL_NUM_RESPONSES=1 \
  bash scripts/eval_checkpoint_b200.sh rac \
  outputs/my_rac_run/rac_opd/checkpoint-000150
```

`METHOD` nhận `opd`, `ta-opd` (hoặc `ta`) và `rac`. Script dùng vLLM mặc định; có thể đặt
`EVAL_BACKEND=hf`. Mặc định riêng của launcher chung là `temperature=1`, `n=16`; đặt
`EVAL_NUM_RESPONSES=1` để lấy accuracy một response thay vì avg@n.

Mỗi thư mục eval chứa `summary.json`, ba file `*_predictions.jsonl.gz`, và file gộp
`model_outputs_detailed.jsonl.gz`. File gộp lưu dataset, ID, đề bài, đáp án chuẩn, prompt thực tế,
toàn bộ response, đúng/sai từng response và generation parameters. Đọc nhanh bằng:

```bash
gzip -cd results/checkpoint_eval/opd_step50/model_outputs_detailed.jsonl.gz | less
```

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

Evaluation mặc định dùng vLLM sampling (`n=16`, `temperature=0.7`, `top_p=0.95`,
`max_new_tokens=7168`) và báo cáo `avg@16`; mỗi problem lưu đủ 16 generation/correctness.
Fallback: `EVAL_BACKEND=hf`. Tensor parallel example:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
CUDA_VISIBLE_DEVICES=0,1 EVAL_VLLM_TENSOR_PARALLEL_SIZE=2 \
bash scripts/eval_all_b200.sh
```

## Re-eval mọi checkpoint đã lưu theo protocol avg@16

Script dưới đây tìm `checkpoint-<step>` và `final/` trong cả ba output. Nó cũng eval lại base ở
step 0 để toàn bộ đường avg@16 dùng cùng protocol. Mỗi `training_eval/step-*` tương ứng,
`eval_history.jsonl` và `eval_metrics.csv` cũ sẽ bị thay thế; raw prediction và `summary.json`
trong từng step cũng bị thay thế. Base được generate một lần rồi tái sử dụng cho hai method còn
lại; checkpoint sau train vẫn được eval độc lập. Đặt `REEVAL_REUSE_BASE=false` nếu cần cố ý sinh
lại base ba lần.

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
protocol cũ và mới. Sau khi hoàn tất, chạy lại lệnh plot periodic training evaluation bên dưới.

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
