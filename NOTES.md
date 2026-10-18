# NOTES.md: Project Log, Decisions, and Assumptions

## 1. Project Overview & Scope
- Scope: Phase 1 Oracle-Gap Gate on frozen dataset v1.0.
- Dataset location: `data/data_v1` (as instructed by user, winning over default `data_v1/`).
- Python environment: Python 3.13.11 (`D:\Miniconda\miniconda3\python.exe` on Windows).
- Pinned packages in `requirements.txt`:
  - `numpy==2.5.3`
  - `scipy==1.18.1`
  - `numba==0.68.0`
  - `networkx==3.7`
  - `pandas==3.0.6`
  - `cma==4.5.0`
  - `pytest==9.1.1`
  - `tqdm==4.67.1`
  - `matplotlib==3.11.2`

## 2. Gate Design: RLF in Priority API & Reference R(F)
- **RLF Representation in Priority Function API**:
  - `rlf_priority(adj, colors, sat, deg, udeg)` strictly implemented:
    - Active color class: $k = \max(colors)$.
    - Candidate vertices: uncolored and non-adjacent to any vertex colored $k$.
    - Excluded vertices: uncolored and adjacent to at least one vertex colored $k$.
    - Candidate score: $n_e \cdot (n + 1) - n_c$.
    - When candidates are empty, open new class $k+1$ using maximum uncolored degree.
  - Verified 100% exact equality against standalone `rlf(adj)` across random and dataset graphs (`test_rlf_priority_matches_standalone_rlf` passing).
- **Reference $R(F)$**:
  - $R(F) = \min(\text{DSATUR}, \text{RLF}, \text{universal})$ chosen per family on `val` graphs.
- **DSATUR Fallback**:
  - For each family, if CMA-ES tuned heuristic does not strictly improve over DSATUR on `val`, it falls back to DSATUR weights ($w = [1, 1, 0, 0, 0]$).
- **Control Group**:
  - The 7 `interval` families serve as negative control (`is_control = True`, DSATUR is optimal = clique number). Separated in all aggregate statistics.

## 3. Phase 1 Gate Gap A Results (on TEST graphs, N=50/family)
- **Non-Control Families (N=99)**:
  - **Gap A vs $R(F)$ (colors)**: **-0.698 colors** [95% CI: -0.990, -0.481]
  - **Gap A vs $R(F)$ (%)**: **-5.87%** [95% CI: -8.05%, -4.29%]
  - **Gain vs DSATUR (colors)**: **+0.021 colors** [95% CI: -0.011, +0.077]
  - **Gain vs DSATUR (%)**: **+0.17%** [95% CI: -0.09%, +0.62%]

## 4. Diagnostics Suite (D1', D4, D5, D6, D7)

### Nature of D1' and D5: Instance-Level Search (Not Fixed Heuristic Ceilings)
- **D1' (Iterated Greedy + TabuCol)** and **D5 (Min over 200 random rollouts)** are **tìm kiếm theo từng đồ thị riêng lẻ (instance-level search)**, KHÔNG PHẢI là trần của một heuristic cố định, đơn nhất.
- Một hàm priority heuristic cố định (deterministic single-pass heuristic) do con người thiết kế hay do LLM sinh ra rất khó có thể đạt tới mức của các bộ tìm kiếm tối ưu hóa theo từng đồ thị này.
- Mục tiêu thực tế của heuristic sinh bởi LLM trong bài toán này không phải là bắt kịp trần tìm kiếm D1'/D5, mà là **vượt qua tham chiếu cố định $R(F)$ theo ngưỡng 3-5%** (Gap B).

### D1'. Near-Optimal Graph Ceiling on VAL (Instance-Level Search)
- TabuCol hạ dần $K$ lặp lại trong vòng lặp `while cur_target >= 1:` cho đến khi không thể giải được.
- Non-Control Families (N=99):
  - Mean Headroom vs $R(F)$: **0.951 colors** [95% CI: 0.662, 1.324], **8.01%** [95% CI: 5.86%, 10.48%]
  - Median Headroom: **0.600 colors**, **5.88%**
  - Families with Headroom $\ge 3.0\%$: **66 / 99 (66.7%)**
- Interval Control (N=7): Headroom = **0.000 colors (0.00%)**

### D4. Parameterized RLF Evaluation on TEST
- Đính chính mã family đạt $\ge 3\%$ trên VAL là `kpart_k10_p0.3_n320-500` (heldout size, $n \in [320, 500]$), không phải `kpart_k10_p0.4_n200`.
- Trên TEST (khóa toàn bộ quyết định và fallback từ VAL):
  - Non-Control Families (N=99):
    - **Mean Gap vs $R(F)$ trên TEST**: **+0.038 colors** [95% CI: -0.015, +0.136], **+0.32%** [95% CI: -0.13%, +1.16%]
    - **Median Gap**: **+0.000 colors**, **+0.00%**
    - **Families với Gap $\ge 3.0\%$ trên TEST**: **1 / 99 (1.0%)** (`kpart_k10_p0.3_n320-500` đạt +2.10 màu, +7.47%)

### D5. Constructive Search Ceiling (Instance-Level Search)
- $\min$ qua 200 rollouts ngẫu nhiên mỗi đồ thị: Mean Headroom vs $R(F)$ = **0.500 colors (4.21%)**, Median = **3.75%**, **60.6% families $\ge 3.0\%$**.

### D6. Deterministic Fixed Heuristics vs Rollout Expectations (on TEST)
- **D6(a) Expected Rollout Mean**: Mean colors = 12.246 vs $R(F)$ = 11.883 (-3.05%).
- **D6(b) 20 Named Deterministic Rules (Selected on VAL, Evaluated on TEST)**:
  - Gap vs $R(F)$ on TEST = **+0.002 colors (+0.02%)** [95% CI: -0.28%, +0.51%], Median = **0.00%**, Families $\ge 3.0\% = 1/99$.

### D7. Card-based Algorithm Selector {DSATUR, RLF} (No LLM)
- **Báo cáo chính: Cross-Validation theo Loại Cấu trúc (Leave-One-Group-Out CV across 11 types)**:
  - Cây quyết định (Decision Tree): **Độ chính xác = 82.1%** | **Regret trung bình = 0.034 colors (0.45%)**
  - Hồi quy Logistic: **Độ chính xác = 81.1%** | **Regret trung bình = 0.025 colors (0.32%)**
  - Chính sách "Luôn dùng RLF" (Always RLF): **Độ chính xác = 75.5%** | **Regret trung bình = 0.028 colors (0.42%)** (trên non-control: 0.030 màu, 0.45%)
  - Chính sách "Luôn dùng DSATUR" (Always DSATUR): **Độ chính xác = 24.5%** | **Regret trung bình = 0.674 colors (4.49%)**
- *Số liệu bổ sung (CV theo Family - LOOCV across 106 families)*:
  - Cây quyết định: Độ chính xác = 86.8% | Regret trung bình = 0.010 colors (0.13%)
  - Hồi quy Logistic: Độ chính xác = 87.7% | Regret trung bình = 0.014 colors (0.16%)
- *Ý nghĩa then chốt*: Bài toán lựa chọn qua lại giữa hai thuật toán kinh điển {DSATUR, RLF} đã được giải quyết bằng mô hình học máy thẻ đơn giản với regret chỉ ~0.3-0.4%. Giá trị của một code model LLM (EoH) là phải sinh ra được logic priority mới vượt lên trên cả hai thuật toán này.

## 5. Giao thức Pilot Tiến hóa Đã Khóa (Locked Pilot Evolution Protocol)

### 1. Bảng 6 Họ Đồ thị Pilot & Giá trị Cơ sở R(F) Khóa trên teacher_val (Seeds 3000-3019):
| Họ đồ thị (amily_id) | Loại cấu trúc | Thuộc Pool | DSATUR (	eacher_val) | RLF (	eacher_val) | (F)_{\\text{teacher\\_val}}$ | Thuật toán tốt nhất |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| kpart_k10_p0.3_n100-200 | equipartite | 	rain_pool | 13.950 | 12.850 | **12.850** | RLF |
| gnp_d0.30_n100-200 | gnp | 	rain_pool | 15.400 | 14.500 | **14.500** | RLF |
| sbm_k10_in0.5_out0.05_n200-300 | planted_partition | 	rain_pool | 10.300 | 9.550 | **9.550** | RLF |
| ws_k10_p0.1_n100-200 | small_world | 	rain_pool | 6.900 | 7.000 | **6.900** | DSATUR |
| interval_L0.10_n100-200 *(Control 1)* | interval | 	rain_pool | 24.300 | 24.300 | **24.300** | DSATUR |
| a_m12_n200-300 *(Control 2)* | scale_free | 	rain_pool (v1.1) | 9.150 | 9.800 | **9.150** | DSATUR |

*Ghi chú*: Tập al (seeds 1000-1019) được giữ sạch cho early stopping trong giai đoạn RL/GRPO về sau; các số liệu al chỉ được ghi nhận như chẩn đoán phụ.

### 2. Tham số Tiến hóa Đã Khóa:
- **Ngân sách mỗi họ**: Tối đa **150 lệnh gọi LLM** (tổng 6 họ x 150 = 900 calls).
- **Trần bảo vệ**: 450,000 tokens / họ, .50 / họ (tính với hệ số an toàn x2).
- **Nhịp độ gọi (Pacing)**: Tối thiểu 5.0 giây giữa hai lệnh gọi liên tiếp; tự động exponential backoff khi nhận mã HTTP 429.
- **Tỷ lệ và Nhiệt độ Toán tử**:
  - explore: **40%** lệnh gọi, nhiệt độ  = 0.9$ (kèm danh sách 5 hành vi đã thử gần nhất để ép mô hình tìm giả thuyết cấu trúc mới: 2-hop, common neighbors dj @ adj, tỷ số bậc, v.v.).
  - crossover: **30%** lệnh gọi, nhiệt độ  = 0.7$.
  - 
efine: **30%** lệnh gọi, nhiệt độ  = 0.5$.
- **Khử trùng lặp hành vi (Behavioral Deduplication)**: Quần thể cha mẹ và danh sách Top-10 chỉ lưu **1 đại diện duy nhất** cho mỗi vector 20 chiều số màu train (ưu tiên candidate có fitness cao nhất, hòa thì lấy mã ngắn nhất). Bản sao tuyệt đối (trùng hash 3 đồ thị đầu) bị cấm vào tập cha mẹ.

### 3. Phân hoạch Seed & Giao thức Chọn lọc:
- **Tập Train**: Seeds 0-19 (20 đồ thị aseline_train).
- **Tập Chọn lọc (Selection)**: 	eacher_val seeds 3000-3019 (20 đồ thị). Đánh giá Top-10 hành vi phân biệt tốt nhất trên 	eacher_val và chọn candidate ^*$ có số màu trung bình nhỏ nhất.
- **Tập Test Pilot**: Seeds 4000-4199 (200 đồ thị). Được đo duy nhất một lần bằng scripts/pilot_test.py sau khi hoàn thành chọn lọc (có khóa .done chống rò rỉ dữ liệu).

### 4. Ngưỡng Thành công Pilot (Go/No-Go Decision Gate):
- **Pilot PASS / Thành công**:
  - Ít nhất **>= 2 / 4 họ mục tiêu** đạt Gap B ghép cặp >= 1.5% trên 	est_pilot (200 đồ thị) với khoảng tin cậy 95% bootstrap loại trừ 0 (CI_lo > 0).
  - Họ đối chứng interval: 95% CI chứa 0 (không xuất hiện cải thiện ảo trên họ đã tối ưu).
  - Họ đối chứng scale_free: không kém R(F) có ý nghĩa thống kê (không suy thoái so với DSATUR).
- **Trường hợp chưa đạt ngưỡng**: Mở rộng thêm 300-400 lệnh gọi tập trung vào 2 họ có headroom/tín hiệu tốt nhất, không dừng dự án.

### Bảng Giá Ước tính Hiện hành (Full Evolutionary Prompt - 900 Lệnh gọi):
- Cấu hình: 6 family $\times$ 150 calls = **900 calls**, prompt đầy đủ (mô tả bài toán + 3 mã cha + fitness + reflection) $\approx$ 2,800 tokens input, 600 tokens output ($\approx 2.52\text{M}$ input tokens, $0.54\text{M}$ output tokens).

| Mô hình | Phân loại | Giá Input / Output (trên 1M tokens) | Chi phí ước tính (900 calls) | Tương đương VNĐ |
| :--- | :--- | :--- | :---: | :---: |
| **DeepSeek V3** | **Rẻ / Tiết kiệm** | \$0.14 / \$0.28 *(cache: \$0.014)* | **~$0.45 - $0.50** | **~12.000 đ** |
| **GPT-4o-mini** | Rẻ (OpenAI) | \$0.15 / \$0.60 | **~$0.70** | **~17.500 đ** |
| **Claude 3.5 Sonnet** | **Mạnh nhất (Anthropic)** | \$3.00 / \$15.00 *(cache: \$0.30)* | **~$9.50 - $15.50** | **~240.000 - 390.000 đ** |
| **GPT-4o** | Mạnh (OpenAI) | \$2.50 / \$10.00 | **~$11.70** | **~290.000 đ** |

### Dataset v1.1 & Cập nhật Metadata (ĐÃ ĐÓNG BĂNG):
- **Nguyên tắc bất biến**: Thư mục `data/data_v1/` được bảo tồn nguyên vẹn (đã đóng băng). Toàn bộ metadata mới được lưu độc lập tại `data/data_v1_1/` (dùng chung các file đồ thị npy có sẵn).
- **Chuyển `scale_free` sang `train_pool`**:
  - Toàn bộ 7 họ `scale_free` được chuyển từ `heldout_type` sang `train_pool`.
  - *Ghi chú phương pháp luận quan trọng*: Quyết định này được đưa ra **sau khi quan sát kết quả gate và TUYỆT ĐỐI TRƯỚC KHI có bất kỳ kết quả huấn luyện nào (trước mọi thử nghiệm RL / GRPO / fine-tuning)**.
  - *Lý do*: Đồ thị scale-free là cấu trúc then chốt duy nhất mà DSATUR thắng áp đảo RLF (RLF kém hơn DSATUR -6.02%). Việc đưa scale-free vào tập train là bắt buộc để mô hình học được khi nào cần ưu tiên bậc/hub thay vì chỉ thiên về RLF.
- **Đóng băng `heldout_type` mới = `small_world`**:
  - Chuyển 7 họ `small_world` tiêu chuẩn ($N \le 300$) từ `train_pool` sang `heldout_type`. Giữ nguyên 2 họ `heldout_size`.
  - **KHÔNG chuyển `mycielski`** (giữ nguyên ở `train_pool` 8 họ và `heldout_size` 2 họ).
  - **Tiêu chí lựa chọn `small_world`**:
    1. Số lượng họ đủ lớn: $\ge 5$ họ (thực tế có 7 họ $N \le 300$).
    2. Khác biệt cấu trúc rõ rệt: mạng small-world (Watts-Strogatz) với hệ số clustering cao và khoảng cách trung bình ngắn, mô hình tái lập trình phân cụm.
    3. RLF hơn DSATUR $\ge 3\%$: RLF tốt hơn DSATUR **$+4.34\%$** trên `small_world`, bảo đảm hai thuật toán mặc định DSATUR và RLF cho kết quả khác biệt rõ ràng, thích hợp để đánh giá khả năng tổng quát hóa OOD.
- **Đính chính quan trọng về mặt lý thuyết**:
  - **Mycielski của ta KHÔNG triangle-free**: Đồ thị cơ sở xuất phát là $G(n_0, p)$ với $p \in [0.05, 0.30]$ nên chứa rất nhiều tam giác từ đầu. Phép biến đổi Mycielskian chỉ bảo toàn tính không tam giác nếu đồ thị cơ sở là triangle-free.
  - **D1' và D5 là trần tìm kiếm theo từng graph (per-graph search ceiling)**, KHÔNG phải trần của một heuristic cố định duy nhất, nên **không dùng làm tiêu chí chọn loại heldout**.
- **Quy ước thẻ mô tả (Cards)**:
  - **Thẻ chuẩn `card` là điều kiện đầu vào chính** trong toàn bộ pipeline LLM / RL.
  - Thẻ chi tiết `card_detailed` (tích hợp tham số generator: $k, p_{in}, p_{out}, m, r, q, L, keep, t$) **chỉ dùng cho ablation study** để đo lường mức độ phụ thuộc vào tham số tường minh.
- **Phân hoạch seed `teacher_val` [3000, 20]**:
  - Dải seed 3000-3019 hoàn toàn tách biệt với `train` (0-99), `val` (1000-1019), `test` (2000-2049), `probe` (9000-9009).
  - Phân bố pool v1.1: 55 `train_pool`, 26 `heldout_type`, 15 `heldout_size`, 10 `heldout_combo` (tổng 106 họ).
  - Test suite `test_dataset.py` khẳng định 100% không trùng seed và đúng phân bổ pool.

### Chẩn đoán D8: Benchmark Chuẩn Vectorized THẬT vs Vertex-level API:
- **Chuẩn hóa API đúng theo tài liệu**:
  - Khung chạy: `candidates = toàn bộ đỉnh chưa tô` (không dựng sẵn tập ứng viên cho lớp màu hiện tại ở khung).
  - Hàm `priority_func(v, graph_dict, color_map, candidates)` tự suy ra lớp màu hiện tại từ `color_map` và tự xác định tính hợp lệ của $v$.
  - Bản đối chứng: **Vectorized THẬT** (dùng tích ma trận NumPy `adj.dot(excl)` và `adj.dot(cand)`, triệt tiêu hoàn toàn vòng lặp Python qua tập ứng viên).
- **Thử nghiệm trên 40 đồ thị** (10 đồ thị val $\times$ 4 họ đại diện: `kpart_k10_p0.3_n100-200`, `gnp_d0.30_n100-200`, `sbm_k10_in0.5_out0.05_n200-300`, `ws_k10_p0.1_n100-200`).
- **Độ chính xác**: 40/40 đồ thị (100%) cho kết quả tô màu **giống hệt từng đỉnh (bitwise identical `np.array_equal`)**.
- **Kết quả đo thời gian thực tế**:
  - `kpart_k10_p0.3_n100-200`: Vectorized **35.24 ms/graph** vs Vertex-level **7,664.67 ms/graph** (Chậm hơn **217.5x**).
  - `gnp_d0.30_n100-200`: Vectorized **34.91 ms/graph** vs Vertex-level **8,257.00 ms/graph** (Chậm hơn **236.5x**).
  - `sbm_k10_in0.5_out0.05_n200-300`: Vectorized **94.34 ms/graph** vs Vertex-level **18,174.80 ms/graph** (Chậm hơn **192.7x**).
  - `ws_k10_p0.1_n100-200`: Vectorized **23.96 ms/graph** vs Vertex-level **2,170.91 ms/graph** (Chậm hơn **90.6x**).
  - **Toàn bộ 40 đồ thị**: Vectorized **47.11 ms/graph** (tổng 1.88s) vs Vertex-level **9,066.85 ms/graph** (tổng 362.67s).
  - **Tỷ lệ chậm hơn trung bình: 192.5x**.
- **Kết luận chọn API**:
  - Vertex API khi không có lọc trước từ khung buộc hàm phải duyệt $O(N)$ ứng viên mỗi lần gọi, dẫn tới độ phức tạp $O(N^3)$ thuần Python ($\approx 18.2\text{s}$ cho 1 đồ thị SBM 250 đỉnh). Đánh giá 20 đồ thị val sẽ tốn tới ~6 phút/candidate, vượt quá trần sandbox.
  - Do đó, **bản Vectorized THẬT bắt buộc phải là API cốt lõi** cho hệ thống tiến hóa và đánh giá!
- **Đã cập nhật**: Thay thế `rlf_priority` trong `evaluator/coloring.py` và `SEED_CODE_RLF` trong prompt (`gate/evolved/task.py`) bằng bản Vectorized THẬT (cách tô giống hệt `rlf()` độc lập). Giữ nguyên hạn mức thời gian sandbox hiện tại (không siết).

## 6. Pre-Pilot Sandbox, Diagnostics & Whitelist Hardening

### 1. Security & Key Hygiene
- API keys are strictly read from active session `$env:GEMINI_API_KEY`. No persistent user/system environment variables, registry entries, or files store any key.
- All log files and repo scanned and scrubbed. Zero keys printed to stdout/stderr.
- Worker processes spawned with thoroughly scrubbed environment (`*_KEY*`, `*API*` deleted from child env).

### 2. Detailed Rejection Logging
- Sandbox logs precise construct/attribute violation names (e.g. `np.diag`, `np.fill_diagonal`, `IndexError`, `Timeout`).
- `smoke_report.py` aggregates and prints a dedicated summary table of rejection reasons.

### 3. Diagnosis of the 2 Purity / Statelessness Failures
- **Candidate #1 (`IndexError: 113 vs 154`)**:
  - Investigation confirmed **zero graph mixing**. Both 113 and 154 occurred on Graph 0 ($n = 154$ nodes).
  - Cause: Candidate wrote `rlf_score = (ne[cand] * (n + 1) - nc[cand])` (length 113, corresponding to the 113 uncolored candidates), and subsequently executed `combined[cand] += rlf_score[cand]`. Applying a length-154 boolean mask `cand` to a length-113 array caused NumPy to throw `IndexError: boolean index did not match indexed array along axis 0; size of axis is 113 but size of corresponding boolean axis is 154`. Rightfully rejected as buggy model-generated code.
- **Candidate #6 (`KeyError: '__import__'`)**:
  - In Python 3.13, NumPy C-extension reductions (`ndarray.any()` / `ndarray.all()`) invoke `PyImport_ImportModule('numpy._core._methods')` when executed in restricted environments where `__builtins__` is a dict lacking `__import__`.
  - Resolution: Implemented `_safe_import` in `gate/evolved/sandbox.py` allowing only modules starting with `numpy.`, while strictly blocking `os`, `sys`, `subprocess`, etc. AST validator continues to ban all user `import` statements and calls to `__import__`.
  - Re-evaluating Candidate #6 across all 20 train graphs achieved **13.500 mean colors** (outperforming RLF baseline of 13.700!).

### 4. Whitelist Expansion for Harmless NumPy Math
- Expanded `ALLOWED_NP_ATTRS` with safe array operations: `diag`, `diagonal`, `matmul`, `triu`, `tril`, `outer`, `bincount`, `unique`, `sign`, `minimum`, `maximum`, `count_nonzero`.
- Expanded `ALLOWED_OBJECT_ATTRS` with `.diagonal`, `.clip`, `.round`.
- Added explicit unit tests (`test_whitelisted_numpy_math_functions` and `test_safe_import_blocks_external_modules`) in `tests/test_sandbox.py`.

### 5. Pacing, 429 Backoff & Wording
- Added minimum 5-second pacing delay between LLM calls and exponential backoff on HTTP 429.
- Updated wording across diagnostic reports and scripts to `"Excess colors vs R(F) (dương = tệ hơn)"`.
- Full test suite: **49 / 49 tests passing** (`pytest`).

**Trạng thái hiện tại: TẤT CẢ VÁ LỖI VÀ CHẨN ĐOÁN ĐÃ XONG. DỪNG CHỜ DUYỆT TRƯỚC KHI CHẠY PILOT.**




## Phase 6 / Batch A Implementation Summary (2026-10-16)

### Step 0 Architectural Summary
1. API: `priority(adj, colors, sat, deg, uncolored_deg) -> scores` returning `(n,)` float ndarray scores (pure vectorized function).
2. Greedy framework masks colored vertices, selects max score (lowest index tie-breaking), assigns min feasible color >= 0, and incrementally updates `sat` and `uncolored_deg`.
3. Sandbox: 4 defense layers in `gate/evolved/sandbox.py` (AST whitelist, isolated restricted builtins with `_safe_import`, purity/statelessness check, and isolated subprocess with 1.5GB memory limit and timeout).
4. Sandbox invocation: `evaluate_candidate_safe(code_str, graphs, per_graph_timeout, overall_timeout)` returning `dict` with `success`, `color_counts`, `mean_colors`, `behavior_hash`, and `error`.
5. DSATUR: standalone Numba fast path `dsatur(adj)` == `greedy_color(adj, dsatur_priority)` with formula `sat * (n + 1) + uncolored_deg`.
6. RLF: standalone Numba `rlf(adj)` == `greedy_color(adj, rlf_priority)` with formula `ne * (n + 1) - nc` (vectorized matrix-product implementation).
7. Verifier: independent `evaluator/verify.py` checks dimensions, symmetry, no self-loops, all colors >= 0, and proper edge coloring (no adjacent vertices share colors).
8. R1-R2 Plan: `SEEDS.md` (13 pairwise disjoint closed ranges), `configs/teacher_llm.yaml`, `gate/evolved/loop.py` latency/status/token cost logging; deterministic generation of `data/data_v1_2/splits.json` on all 106 existing families.
9. R3-R4 Plan: `eval/coverage.py` (pure metrics: coverage@k, best-of-k, pass@k, beats R(F)), `eval/eval_model.py` (cached and test-locked offline evaluation CLI with lazy backends), `train/reward.py` (pure GRPO reward dataclass, clipping, Kendall-tau diversity, advantages).
10. Integrity: `tests/test_frozen_files.py` guarding frozen assets (`data/data_v1`, `data/data_v1_1`, `evaluator/verify.py`, sandbox whitelist), lazy imports for heavy optional deps, all 52 existing pytest green + new test suites.

### Modules Added and Modified
- **Created**:
  * `SEEDS.md` (Global closed-range seed registry)
  * `configs/teacher_llm.yaml` (Teacher model config with env-var-only API key reference)
  * `scripts/build_splits.py` (Deterministic splits builder for Dataset v1.2)
  * `data/data_v1_2/splits.json` (106 family partition into 6 groups with attribute binning)
  * `eval/coverage.py` (Pure metrics for "beats R(F)", pass@k, coverage@k, best-of-k)
  * `eval/eval_model.py` (Offline evaluation harness CLI with caching and test-split locking)
  * `train/reward.py` (Pure GRPO reward engine and advantage computation)
- **New Tests**:
  * `tests/test_seeds.py` (SEEDS.md schema, closure, and pairwise disjointness)
  * `tests/test_frozen_files.py` (Cryptographic manifest check of all frozen files and sandbox whitelist)
  * `tests/test_splits.py` (Validation of dataset v1.2 partitions, disjointness, extrapolation, heldout rules)
  * `tests/test_eval_harness.py` (Unit tests for coverage metrics, mock backend, caching, test-locks)
  * `tests/test_reward.py` (GRPO reward calculation, failure handling, clipping, Kendall-tau diversity, bounds)
  * `tests/test_loop_telemetry.py` (Verification of additive latency/token/status/cost logging in loop.py)
  * `tests/test_adversarial_m1.py` (Adversarial integrity tests for sandbox and configuration safety)
- **Modified**:
  * `gate/evolved/loop.py` (Strictly additive telemetry logging: `prompt_tokens`, `completion_tokens`, `latency_s`, `operator`, `temperature`, `status`, `cost_dollars`)

### Number of Families per Split (All 106 Families)
Partition of the 106 existing dataset families in `data/data_v1_2/splits.json`:
- `train_pool`: 41 families (includes `scale_free`, excludes `small_world` and held-out combinations)
- `val`: 7 families
- `test_interp`: 7 families
- `test_extrap`: 15 families (exceeds train range in nodes or density)
- `test_comp`: 10 families (held-out attribute combinations absent from `train_pool`)
- `heldout_type`: 26 families (`small_world` held out type, plus 2-community planted partition, interval control, etc.)
- **Flag status**: Zero splits have fewer than 3 families; all splits contain at least 7 families.

### Canonical Definition of "beats R(F)" and eps
- A candidate heuristic beats $R(F)$ on a card iff its mean colors across the split graphs satisfies:
  $$\text{mean\_colors} < R(F) - \varepsilon$$
- Default threshold: $\varepsilon = 0.05$ colors (configurable).
- Ties do **NOT** count as wins.
- $R(F) = \min(\text{mean DSATUR}, \text{mean RLF})$ evaluated on the exact same graphs and seeds.

### Reward Default Coefficients and Mathematical Bounds
- **Default Hyperparameters**:
  * $a = 0.0$ (tie-breaking weight)
  * $b = 0.1$ (runtime penalty weight)
  * $d = 0.0$ (diversity reward weight)
  * $\text{time\_factor} = 3.0$ ($t_{\text{budget}} = 3.0 \times t_{\text{DSATUR}}$)
  * $\varepsilon_{\text{std}} = 10^{-8}$ (numerical stability epsilon)
- **Mathematical Bounds**:
  * Savings component: $r_{\text{sav}} \in [-1.0, 1.0]$ due to explicit clipping.
  * Tie-breaking component: $r_{\text{tb}} \in [0.0, 1.0]$.
  * Runtime penalty: $r_{\text{time}} \in [0.0, 1.0]$, entering as $-b \cdot r_{\text{time}} \in [-b, 0.0]$.
  * Pre-reward: $r_{\text{pre}} = r_{\text{sav}} + a \cdot r_{\text{tb}} - b \cdot r_{\text{time}} \in [-1.0 - b, 1.0 + a]$.
  * Diversity component: $r_{\text{div}} \in [0.0, 1.0]$, entering as $+d \cdot r_{\text{div}} \in [0.0, d]$.
  * Total reward bounds for valid candidates:
    $$r \in [-1.0 - b, 1.0 + a + d]$$
  * With defaults ($a=0.0, b=0.1, d=0.0$), the range is exactly $[-1.1, 1.0]$.
  * Failure status (AST violation, purity violation, timeout, memory, invalid coloring, runtime exception) yields $r = -1.0$ exactly, and failed candidates are excluded from diversity pair computations.

### Exact Commands to Reproduce
1. Run complete unit and regression test suite:
   ```bash
   pytest
   ```
2. Run frozen files cryptographic integrity check:
   ```bash
   pytest tests/test_frozen_files.py
   ```
3. Run smoke demo evaluation CLI on validation split with mock backend:
   ```bash
   python eval/eval_model.py --backend mock --split val --k 4 --n-samples 4 --temperature 0.7 --n-graphs 2 --run-id smoke_demo --seed 42
   ```

### Additional Findings
1. **Module path for parametric heuristic**: `AGENTS.md` originally specified `evaluator/parametric.py`, while the Phase 1 implementation placed it at `gate/parametric.py`. All imports and tests correctly resolve via `gate/parametric.py` (or wrapper interfaces) without modifying frozen files.
2. **Windows multiprocessing & subprocess isolation**: In `eval/eval_model.py`, candidate evaluation across cards uses `concurrent.futures.ThreadPoolExecutor` at the harness layer on Windows, while each candidate's safety is enforced through isolated worker subprocesses spawned via `gate/evolved/sandbox.py`. This avoids Windows `multiprocessing.spawn` pickling limitations with local closures while guaranteeing strict 1.5GB memory limit and per-graph timeout isolation.

## Phase 6 / Batch B: D7 Classifier, Cards Catalog, LLM4AD Adapter & Gate 2 Report

### Completed Deliverables
1. **Task 9.1: D7 Classifier Baseline (`baselines/d7_classifier.py`)**:
   - Built a card-attribute-based classifier mapping instance family features (`density`, `clustering`, `degree_cv`, `n_max`) to algorithm choice $\{DSATUR, RLF\}$.
   - Trained on `train_pool` (41 families) and evaluated on all 6 splits (`train_pool`, `val`, `test_interp`, `test_extrap`, `test_comp`, `heldout_type`).
   - Results: Decision Tree achieved 90.2% on train, 100% on extrap/comp, and 88.5% on heldout_type. Logistic Regression achieved 85.4% on train and 96.2% on heldout_type.
   - Across all splits, D7 selector regret vs true $R(F)$ is $\le 0.086$ colors ($< 1\%$), setting a competitive benchmark for LLMs.
   - Unit tests in `tests/test_d7_classifier.py` passed (6/6).

2. **Task 8.4: Cards & Paraphrase Generation (`scripts/make_cards.py`)**:
   - Generated full catalog of 106 families with `card`, `card_detailed`, and 3 paraphrases per card.
   - Enforced automated verification: all numbers from original cards are strictly preserved in paraphrases, and no forbidden label-leak keywords (e.g., `dsatur`, `rlf`, `winner`, `optimal`) appear.
   - Exported `data/data_v1_2/cards.json` and a 50-sample randomized spot-check CSV `data/data_v1_2/cards_spotcheck_50.csv`.
   - Unit tests in `tests/test_cards.py` passed (3/3).

3. **Task 7.1: LLM4AD Task Adapter (`baselines/llm4ad_task.py`)**:
   - Created `GraphColoringEvaluation` class inheriting from LLM4AD `Evaluation` base class (with standalone fallback if `llm4ad` is not installed).
   - Wrapped project's strict 4-layer sandbox (`evaluate_candidate_safe`) and verification pipeline (`evaluator/verify.py`).
   - Evaluates programs using exact vectorized API contract and returns `-mean_colors` (as LLM4AD/EoH/ReEvo maximize fitness).
   - Unit tests in `tests/test_llm4ad_adapter.py` passed (5/5), confirming that valid code matches direct sandbox scores with zero error, and AST/timeout/stateful violations are rejected.

4. **Task 6.2: Gate 2 Decision Report Script (`scripts/gate2_report.py`)**:
   - Selects top-1 candidate per family across runs based strictly on `teacher_val` (seeds 3000-3019).
   - Evaluates once on `test_pilot` (seeds 4000-4199) with `.done` lock mechanism.
   - Computes paired 10,000-resample bootstrap 95% CIs and paired Wilcoxon signed-rank test.
   - Evaluates predefined Gate 2 criteria: PASS (>= 2/4 target pass with improvement >= 1.0% and valid_rate 100%), MARGINAL (1/4 target pass or 0.5-1.0% improvement), FAIL otherwise.
   - Produces markdown report (`runs/gate2/report.md`) and JSON summary (`runs/gate2/report.json`).
   - Unit tests in `tests/test_gate2_report.py` passed (3/3).

5. **Comprehensive Test Suite**:
   - Full test suite: **186/186 tests passing** in 101.61s.
   - Zero modifications to frozen data/manifests (`data/data_v1/`, `data/data_v1_1/`, `evaluator/verify.py`).
   - Zero API keys written, printed, or exposed.

## Full Implementation of Roadmap: Phase 7 through Phase 13

### 1. New Modules and Pipelines Implemented
1. **Task 7.2: EoH & ReEvo Baseline Runners (`baselines/run_eoh.py`, `baselines/run_reevo.py`)**:
   - Implemented matched-budget evolution runners utilizing the unified 4-layer sandbox adapter (`GraphColoringEvaluation`).
   - Tracked full cost metrics (`prompt_tokens`, `completion_tokens`, `latency_s`, `temperature`, `operator`, `status`).
   - ReEvo incorporates explicit reflection steps (`reevo_reflection`) counted toward the total call budget.
   - Verified by unit tests in `tests/test_run_eoh_reevo.py` (2/2 green).

2. **Task 8.2 & 8.3: Large-Scale Teacher Dataset Pipeline & SFT Filter (`scripts/build_teacher_data.py`, `scripts/filter_sft_data.py`)**:
   - `build_teacher_data.py`: Gathers surviving candidate heuristics across evolution engines, evaluates `base_alg(card)` on `teacher_val` (3000-3019), dedups by behavioral fingerprint on `probe` (9000-9009), rescores on clean `teacher_rescore` seeds (6000-6049), and outputs `records.jsonl` matching Section 14.1 schema.
   - `filter_sft_data.py`: Strictly partitions records into `win` (savings vs R(F) > 0.05%), `tie`, and `contrast` per decision rule D-9. Partitions train and val datasets strictly by CARD (disjoint sets) to eliminate prompt data leakage. Computes sample weights inversely proportional to rank.
   - Verified by unit tests in `tests/test_teacher_sft_pipeline.py` (green).

3. **Phase 10: SFT Training Pipeline (`train/train_sft.py`)**:
   - HuggingFace + LoRA training pipeline for Qwen2.5-Coder-7B-Instruct with prompt format conforming to Section 14.2.
   - Features custom evaluation callback tracking real graph coloring savings on `sft_val_probe` (5000-5019) via sandbox.
   - Uses lazy imports of heavy dependencies (torch, transformers, peft, trl) to ensure portability on local CPU environments.

4. **Phase 11: GRPO Rollout & RL Training Engine (`train/rollout.py`, `train/train_grpo.py`)**:
   - `rollout.py`: Dynamically samples fresh graphs from `rl_train_pool` (seeds 100000-199999), evaluates candidate groups of size $G=8$ on the identical graph set in parallel, and produces `CandidateResult` objects for `train/reward.py`.
   - `train_grpo.py`: Implements GRPO training across Arms A2 (GRPO-only), A3 (SFT -> GRPO), and A4 (A3 + diversity + KL).
   - Telemetry tracks advantage normalization, verified reward components, failure rates, and baseline similarity.
   - Verified by unit tests in `tests/test_sft_grpo_training.py` (2/2 green).

5. **Task 12.4: Analytical Cost & Break-Even Model (`analysis/cost_model.py`)**:
   - Evaluates per-family from-scratch evolution cost vs one-off amortized training investment (teacher collection + SFT + GRPO) vs per-card inference deployment.
   - Computes break-even family threshold and sensitivity grid across API price multipliers and GPU hourly rates.
   - Output generated at `runs/analysis/cost_model/cost_report.md` and `cost_report.json`.

6. **Task 13.1 & 13.3: Master Paper Tables Generator (`scripts/make_all_figures.py`)**:
   - Reads strictly from saved `runs/` outputs without recomputation or LLM calls.
   - Formats complete master table covering all arms (A0, A1, A2, A3, A4, A5, A6, D7-classifier, DSATUR, RLF, R(F), EoH, ReEvo).
   - Outputs Markdown (`runs/paper/table_main.md`) and publication-ready LaTeX (`runs/paper/table_main.tex`).
   - Verified by unit tests in `tests/test_cost_and_paper_tables.py` (2/2 green).

### 2. Test Suite & Integrity Status
- Total unit and integration tests: **193 passed** (100% green in 2m38s).
- Cryptographic SHA256 integrity: `data/data_v1/`, `data/data_v1_1/`, and `evaluator/verify.py` remain strictly unchanged.
- Zero secrets or API keys exposed anywhere in code, configuration, or logs.

## Live API Key & Pilot Evolution Execution Findings (October 2026)

### 1. API Quota and Model Availability Diagnostics
- **Gemini Free Tier Quota**: The provided Google Gemini API key is on the free tier (`generate_content_free_tier_requests`).
  - `gemini-3.8-flash`: Exceeded daily free-tier project quota (`RESOURCE_EXHAUSTED`, 20 requests/day per project limit).
  - `gemini-3.5-flash`: Experiences temporary high-demand spikes (HTTP 503).
  - `gemini-3.5-flash-lite`: 100% available with active quota, returns HTTP 200, fast response latency (< 3s), very cost-effective ($0.075/1M input, $0.30/1M output).
  - `gemini-flash-lite-latest`: Available with HTTP 200.
- **Model Configuration**: Updated active model in `configs/pricing.json` to `gemini-3.5-flash-lite`.

### 2. Code Extraction & Sandbox Robustness
- **AST Import Filtering**: LLMs frequently emit `import numpy as np` at the top of code blocks as a standard Python programming reflex. `extract_code_block` in `gate/evolved/loop.py` was updated to automatically strip redundant `import numpy` lines since `np` is already provided in the sandbox execution scope.
- **Resume Filter Generalization**: Updated `loop.py` so that mock tests (`mock=True`) accept test fixture records regardless of model string tag, while maintaining strict model isolation for real API runs.

### 3. Live Pilot Evolution Execution
- Successfully ran pilot evolution with live Gemini API on `kpart_k10_p0.3_n100-200`:
  - 5 calls completed with real API telemetry (`prompt_tokens`, `completion_tokens`, `latency_s`, `cost_dollars`, `status`).
  - Exponential backoff retry successfully recovered from temporary HTTP 503 spikes without crash.
  - Candidate heuristics were evaluated on 20 baseline train graphs and `teacher_val` graphs.
  - Frozen selection saved to `results/selections/kpart_k10_p0.3_n100-200.json`.
- Test suite integrity: **193/193 unit and integration tests passing (100% green in 2m33s)**.
- Cryptographic file integrity: `data/data_v1/`, `data/data_v1_1/`, and `evaluator/verify.py` remain untouched.

## Repository Cleanup & Refactoring (October 2026)
- **Stray & Scratch Scripts Removed**: Cleaned up 14 obsolete debug, exploratory, and duplicate scripts from `scripts/` (including duplicate `scripts/test_adversarial_m1.py`, `pilot_test.py`, `pilot_final.py`, `smoke_report.py`, `run_dryrun.py`, etc.).
- **Internal Artifacts Removed**: Removed temporary `.agents/` teamwork preview logs and `ORIGINAL_REQUEST.md`.
- **Clean Scripts Directory (14 files remaining)**: `build_splits.py`, `build_teacher_data.py`, `build_v1_1_metadata.py`, `filter_sft_data.py`, `gate2_report.py`, `make_all_figures.py`, `make_cards.py`, `pilot_evolve.py`, `smoke.py`, `run_gate.py`, and the baseline experiment runners (`run_d1_prime.py`, `run_d6_and_eval.py`, `run_diagnostics.py`, `run_val_diagnostics.py`).
- **Integrity & Test Suite**: All 193 unit & integration tests remain 100% GREEN. Frozen files (`data/data_v1/`, `data/data_v1_1/`, `evaluator/verify.py`) are strictly verified via SHA256.
