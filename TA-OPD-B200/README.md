# TA-OPD vs Bellman-RAC trên NVIDIA B200

Project độc lập này huấn luyện cùng student Qwen3-1.7B từ teacher Qwen3-4B bằng hai phương pháp:

- **TA-OPD gốc**: local teachability và hard top-`rho` token budget.
- **Bellman-RAC**: cùng local teachability, truyền thông tin tương lai theo transition thực tế mà
  teacher ủng hộ, rồi weight mềm mọi response token.

Hai phương pháp dùng chung data order, rollout, teacher/student scoring, sampled-token PPO-clipped
OPD loss, optimizer, checkpoint và evaluation. Chỉ cơ chế phân bổ supervision token khác nhau.
Xem lệnh đầy đủ từ shell mới trong [`RUN_B200.md`](RUN_B200.md), hoặc bản tiếng Việt ngắn trong
[`HUONG_DAN_CHAY.md`](HUONG_DAN_CHAY.md).

## Runtime paths

Mọi asset được resolve từ đúng một giá trị:

```yaml
paths:
  storage_root: /workspace/storage-shared
```

Có thể đổi ở launch time bằng `STORAGE_ROOT=/mount/khac`. Các path tương đối còn lại là:

| Asset | Relative path dưới `STORAGE_ROOT` |
|---|---|
| Teacher | `models/Qwen3-4B` |
| Student | `models/Qwen3-1.7B` |
| Full DAPO | `nlp/minhpn19/data/DAPO-Math-17k-Processed` |
| MATH-500 | `nlp/minhpn19/data/eval/math500` |
| AIME 2024 | `nlp/minhpn19/data/eval/aime24` |
| AIME 2025 | `nlp/minhpn19/data/eval/aime25` |

Training luôn đọc toàn bộ split `all`; batch cuối không được pad hay lặp. Loader hỗ trợ parquet,
JSON và JSONL. Evaluation chọn file hỗ trợ theo thứ tự xác định, xác minh schema question/answer và
lưu schema đã dùng trong output.

## Định nghĩa TA-OPD

Tại mọi response position hợp lệ `t`, code lấy top-`K` của student và teacher, tạo literal union
`U_t`, rồi renormalize cả hai phân phối trên mọi ID trong union:

```text
D_t = KL(qbar_t^U || pbar_t^U)
C_t = sum_{v in TopK(student)} q_t(v)
```

`D` và `C` được robust-normalize trên **toàn global rollout batch**, kể cả khi dùng DDP:

```text
Norm_B(z) = clip((z - Q05(z)) / (Q95(z) - Q05(z) + eps), 0, 1)
s_teach   = Norm_B(D) * Norm_B(C)
```

TA chọn chính xác `ceil(rho * N_valid)` vị trí có `s_teach` lớn nhất. Default `K=16`,
`rho=0.10`. Tie được giải ổn định theo flat global token index.

## Định nghĩa Bellman-RAC

Bellman-RAC không sinh token counterfactual và không tạo branch rollout. Nó tái sử dụng original
student trajectory và các score đã cần cho TA:

```text
g_t = s_teach_t
a_t = exp(clamp(log q_t(y_t) - log p_t(y_t), -20, 0))

R_t = g_t + gamma * a_t * R_{t+1}
M_t = 1   + gamma * a_t * M_{t+1}
V_t = R_t / (M_t + eps)

z_t = Norm_B(V_t)
w_t = w_min + (1 - w_min) * z_t^beta
```

Recurrence reset độc lập tại terminal/padding của từng response. Default `gamma=0.995`,
`w_min=0.10`, `beta=2.0`. Mọi statistic selector đều detached; gradient chỉ đi qua OPD token loss
trên original rollout:

```text
L_RAC = sum_t w_t * ell_t_OPD / (sum_t w_t + eps)
```

Vì `w_t >= w_min`, mọi valid response token đều nhận gradient. `rho` chỉ được TA dùng; nó vẫn nằm
trong shared config để audit fairness.

## Hot path

- Student rollout được tạo đúng một lần mỗi optimizer step bằng persistent co-located vLLM server;
  HF fallback có thể bật bằng `ROLLOUT_BACKEND=hf`.
- Student và frozen teacher mỗi model score original trajectory đúng một lần; RAC không yêu cầu
  KV cache hay second-generation pass.
- TA top-K/KL, actual-token log-prob ratio, global quantiles, RAC weights và loss weighting đều là
  tensor operations trên GPU, với FP32 cho score/reduction nhạy số.
- Bellman mặc định dùng associative affine suffix scan O(log T), vector hóa theo batch. Backend
  `reference` chứa recurrence rõ ràng để debug/cross-check.
- BF16, FlashAttention 2 nếu có (SDPA fallback), fused AdamW nếu runtime hỗ trợ, DDP world size tự
  phát hiện và vLLM CUDA-IPC live-weight sync được giữ nguyên.

Không có speedup nào được khẳng định trước khi đo trên B200. Metrics tách riêng rollout, teacher
score, TA local score, Bellman scan, forward/backward, optimizer, evaluation, throughput và peak
allocated/reserved VRAM.

## Config công bằng

Ba config hiện có:

```text
configs/qwen3_b200_base.yaml
configs/qwen3_b200_ta.yaml
configs/qwen3_b200_rac.yaml
```

Hai method config chỉ override `experiment.method` và `experiment.output_dir`; toàn bộ model, data,
rollout, seed, batch, optimizer, schedule và evaluation settings được kế thừa từ cùng base config.
Các launcher đều expose:

```text
LR NUM_EPOCHS MAX_STEPS GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE GRAD_ACCUM_STEPS
MAX_PROMPT_LEN MAX_RESPONSE_LEN TOP_K TA_RHO
RAC_GAMMA RAC_W_MIN RAC_BETA RAC_SCAN_BACKEND
EVAL_INTERVAL SAVE_INTERVAL LOG_INTERVAL SEED
```

Defaults là BF16, full-parameter student, `LR=1e-6`, một epoch, global batch 8, micro-batch 1,
prompt/response `2048/8192`, eval/save mỗi 50 step. Đây là điểm bắt đầu thận trọng nhưng **không phải
cam kết fit** cho mọi driver/package/GPU layout; chạy preflight và điều chỉnh batch/length/memory
fraction đối xứng cho TA và RAC.

<!-- B200_AUTOTUNE_RESULT_START -->
**Measured target result:** chưa chạy trên máy B200; `scripts/smoke_test_b200.sh` sẽ cập nhật block này.
<!-- B200_AUTOTUNE_RESULT_END -->

## Output và resume

Fresh launch tự tạo tên:

```text
ta_qwen3_4b_to_1p7b_YYYYMMDD_HHMMSS
rac_bellman_qwen3_4b_to_1p7b_YYYYMMDD_HHMMSS
```

Mỗi output chứa `resolved_config.yaml`, metadata, `metrics.jsonl`, `train_metrics.csv`,
`eval_metrics.csv`, periodic full eval, compact `token_score_stats`, checkpoints và `latest.json`.
Checkpoint được ghi qua temporary directory rồi atomic rename; model, optimizer step/state và
Torch/CUDA RNG state được khôi phục. Dùng checkpoint cụ thể hoặc `RESUME=auto` cùng tên run cũ.

Detailed TA selected-token JSONL được tắt mặc định để tránh tăng disk không giới hạn; compact global
histogram/quantile và bounded scalar sample vẫn luôn đủ cho plots. Có thể chủ động bật bằng
`--set logging.selected_tokens_enabled=true` cho một run audit ngắn.

Plot launch tạo một folder timestamp mới `results/.../plots/plot_YYYYMMDD_HHMMSS/`, sinh PNG và PDF
cho accuracy, loss, TA score distribution, Bellman-RAC `g/V/w`, và mean alignment/V/weight.
`plot_training_progress.sh` hỗ trợ `PLOT_METHOD=both` để so sánh hoặc `PLOT_METHOD=ta|rac` để vẽ
riêng ba đường accuracy MATH-500/AIME24/AIME25 của một phương pháp.

## Validation

Các test nhẹ bao phủ literal TA formula, robust normalization, Bellman recurrence tính tay,
padding/trajectory reset, bounds, detachment, gradient path, optimized/reference equivalence,
global DDP normalization/budget, resume, eval schedule, loaders, vLLM wrappers và plotting.

```bash
python -m unittest discover -s tests -v
ruff check b200_experiment tests
bash -n scripts/*.sh
```

`scripts/smoke_test_b200.sh` chạy unit tests rồi preflight model/data/GPU; nó không tự bắt đầu full
training trừ khi chủ động bật batch autotune.
