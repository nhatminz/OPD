# Pure OPD vs TA-OPD vs Bellman-RAC trên NVIDIA B200

Project độc lập này huấn luyện cùng student Qwen3-1.7B từ teacher Qwen3-4B bằng ba phương pháp:

- **OPD thuần**: mọi valid response token có uniform weight `1`.
- **TA-OPD gốc**: local teachability và hard top-`rho` token budget.
- **Bellman-RAC**: cùng local teachability, truyền thông tin tương lai theo transition thực tế mà
  teacher ủng hộ, rồi weight mềm mọi response token.

Ba phương pháp dùng chung data order, vLLM rollout, teacher scoring, **Top-K OPD core**, optimizer,
checkpoint và evaluation. Chỉ cơ chế phân bổ loss qua response position khác nhau. Core được port
từ [`thunlp/OPD`](https://github.com/thunlp/OPD) tại commit
`ac26e38d6f1572eb027597b48a9f4e01f6915ef8`.
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

## Common Top-K OPD core

Tại mỗi prefix, student chọn `S_t = StudentTopK(p_t, K=16)`. Teacher được evaluate trên chính các
ID trong `S_t`. Theo recipe upstream `only_stu + student_p + token_reward_direct`, core tính:

```text
alpha_t,k = softmax_k(log p_t(k))
A_t,k     = (log q_t(k) - log p_old,t(k)) * alpha_t,k
ell_t     = sum_k PPOClippedLoss(log p_t(k), log p_old,t(k), A_t,k)
```

`K=16` là optimization support thật, không phải diagnostic và không còn objective sampled-token-only.
Sau đó cả ba method dùng cùng normalization trên **global rollout batch**, bao gồm mọi DDP rank:

```text
L = sum_t w_t * ell_t / sum_t w_t
```

OPD đặt `w_t=1`; TA đặt hard mask top-`rho`; RAC đặt continuous Bellman weight. Vì vậy per-position
signal và denominator convention hoàn toàn chung.

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
- Common core score student Top-K và teacher-on-student IDs một lần. Với TA/RAC, student và teacher
  được forward chung theo từng bounded micro-batch: code lấy luôn student-on-teacher Top-K từ logit
  view đang có, nên literal union vẫn chính xác nhưng bỏ được forward student thứ ba. OPD không cần
  thống kê chéo này. Mọi scoring chỉ giữ tensor `[B,T,K]` qua toàn rollout; hai full-vocabulary logit
  view BF16 chỉ cùng tồn tại bên trong một scoring micro-batch rồi được giải phóng.
  Mỗi prompt được interleave `n=4` lần và gửi thành bốn request có seed riêng; RAC không yêu cầu
  counterfactual generation.
- TA top-K/KL, actual-token log-prob ratio, global quantiles, RAC weights và loss weighting đều là
  tensor operations trên GPU, với BF16 model forward và FP32 cho logsumexp/score/reduction nhạy số.
- Bellman mặc định dùng associative affine suffix scan O(log T), vector hóa theo batch. Backend
  `reference` chứa recurrence rõ ràng để debug/cross-check.
- BF16, FlashAttention 2 nếu có (SDPA fallback), fused AdamW nếu runtime hỗ trợ, DDP world size tự
  phát hiện và vLLM CUDA-IPC live-weight sync được giữ nguyên.

Không có speedup nào được khẳng định trước khi đo trên B200. Metrics tách riêng rollout, teacher
score, TA local score, Bellman scan, forward/backward, optimizer, evaluation, throughput và peak
allocated/reserved VRAM.

## Config công bằng

Bốn config hiện có:

```text
configs/qwen3_b200_base.yaml
configs/qwen3_b200_opd.yaml
configs/qwen3_b200_ta.yaml
configs/qwen3_b200_rac.yaml
```

Ba method config chỉ override `experiment.method` và `experiment.output_dir`; toàn bộ model, data,
rollout, seed, batch, optimizer, schedule và evaluation settings được kế thừa từ cùng base config.
Các launcher đều expose:

```text
LR EPOCHS MAX_STEPS BATCH_SIZE MICRO_BATCH_SIZE NUM_RESPONSES
MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH TOP_K TA_RHO
RAC_GAMMA RAC_W_MIN RAC_BETA RAC_SCAN_BACKEND
EVAL_INTERVAL SAVE_INTERVAL LOG_INTERVAL SEED
```

Defaults là BF16, full-parameter student, `LR=1e-6`, một epoch, prompt batch 16, `n=4`
(tối đa 64 trajectories), micro-batch 1, prompt/response `1024/7168`, eval/save mỗi 50 step. Đây là
điểm bắt đầu thận trọng nhưng **không phải
cam kết fit** cho mọi driver/package/GPU layout; chạy preflight và điều chỉnh batch/length/memory
fraction đối xứng cho OPD, TA và RAC.

Evaluation mặc định dùng vLLM `n=16`, `temperature=0.7`, `top_p=0.95`, `max_new_tokens=7168`; metric
chính là `avg@16`, tức mean của `number_correct/16` theo problem. Để eval lại toàn bộ checkpoint đã
lưu và thay thế lịch sử/file eval cũ, dùng
`scripts/reeval_all_checkpoints_b200.sh`; lệnh dry-run/chạy thật nằm trong `RUN_B200.md`.
560 problem tạo đúng 8.960 responses. Step-0 base giống hệt giữa ba method nên được generate một
lần và cache có fingerprint; mọi checkpoint đã train vẫn eval riêng. Evaluator bật vLLM
`performance_mode=throughput`, chunked prefill và async scheduling mặc định.

<!-- B200_AUTOTUNE_RESULT_START -->
**Measured target result:** chưa chạy trên máy B200; `scripts/smoke_test_b200.sh` sẽ cập nhật block này.
<!-- B200_AUTOTUNE_RESULT_END -->

## Output và resume

Fresh launch tự tạo tên:

```text
opd_qwen3_4b_to_1p7b_YYYYMMDD_HHMMSS
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
cho avg@16, loss, TA score distribution, Bellman-RAC `g/V/w`, và mean alignment/V/weight.
`plot_training_progress.sh` hỗ trợ `PLOT_METHODS='opd ta rac'` với một, hai hoặc cả ba method.
Một method tạo ba đường MATH-500/AIME24/AIME25; từ hai method trở lên tạo ba subplot benchmark,
mỗi subplot có một đường cho từng method. `PLOT_METHOD=both` vẫn tương thích và có nghĩa TA+RAC.

## Validation

Các test nhẹ bao phủ Top-K OPD khớp upstream trên synthetic logits, n=4 expansion, global
weighted-token mean/DDP, uniform pure-OPD allocation, literal TA formula, robust normalization,
Bellman recurrence tính tay,
padding/trajectory reset, bounds, detachment, gradient path, optimized/reference equivalence,
global DDP normalization/budget, resume, eval schedule, loaders, vLLM wrappers và plotting.

```bash
python -m unittest discover -s tests -v
ruff check b200_experiment tests
bash -n scripts/*.sh
```

`scripts/smoke_test_b200.sh` chạy unit tests rồi preflight model/data/GPU; nó không tự bắt đầu full
training trừ khi chủ động bật batch autotune. Khi autotune được bật, step đầu của OPD/TA/RAC còn
phải có cùng rollout hash và đúng allocation policy trước khi batch candidate được chấp nhận.
