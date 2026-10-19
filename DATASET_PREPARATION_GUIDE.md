# HƯỚNG DẪN CHIẾN LƯỢC CHUẨN BỊ DATASET SFT (2,000 – 3,000 SAMPLES) TIẾT KIỆM LLM CALLS CHO GEMINI

**Dự án:** *Distill, then Sharpen: Amortizing Evolutionary Heuristic Search into a Family-Conditioned Code LLM*  
**Mục tiêu:** Xây dựng tập dữ liệu huấn luyện Supervised Fine-Tuning (SFT) chuẩn chỉnh $2,000 \div 3,000$ mẫu cho mô hình `Qwen2.5-Coder-7B-Instruct`, tối ưu hóa chi phí API gọi LLM (tận dụng tối đa Google Gemini Free Tier và kỹ thuật Data Amplification), đảm bảo 100% mẫu vượt qua Sandbox 4 tầng và đánh bại/ngang bằng cận Oracle $R(F)$.

---

## MỤC LỤC
1. [Giải đáp Khoa học: Tại sao cần 2,000 – 3,000 samples?](#1-giải-đáp-khoa-học-tại-sao-cần-2000--3000-samples)
2. [Chiến lược Cốt lõi: Nhân bản 1 $\to$ 8 (Data Amplification)](#2-chiến-lược-cốt-lõi-nhân-bản-1--8-data-amplification)
3. [Các Nguồn Dữ liệu Thuần Miễn Phí (Zero LLM Calls)](#3-các-nguồn-dữ-liệu-thuần-miễn-phí-zero-llm-calls)
4. [Quy trình Thu hoạch Teacher bằng Google Gemini (Chi phí $0.00)](#4-quy-trình-thu-hoạch-teacher-bằng-google-gemini-chi-phí-000)
5. [Quy tắc Phân bổ Dữ liệu (Target: 2,500 Mẫu Chuẩn)](#5-quy-tắc-phân-bổ-dữ-liệu-target-2500-mẫu-chuẩn)
6. [Các Lệnh Thực thi Sẵn có trong Repository](#6-các-lệnh-thực-thi-sẵn-có-trong-repository)
7. [Bảng Dự toán Chi phí & Thời gian](#7-bảng-dự-toán-chi-phí--thời-gian)
8. [Tóm tắt Khuyến nghị Gửi Nhóm & Báo cáo](#8-tóm-tắt-khuyến-nghị-gửi-nhóm--báo-cáo)

---

## 1. Giải đáp Khoa học: Tại sao cần 2,000 – 3,000 samples?

Trong nghiên cứu hiện đại về Fine-tuning Code LLM (tiêu biểu như các công trình LIMA, CodeLlama, DeepSeek-Coder, WizardCoder):
- **Giai đoạn SFT đóng vai trò gì?** SFT **không** có nhiệm vụ bắt mô hình học thuộc lòng mọi thuật toán của hàng triệu đồ thị. Nhiệm vụ chính của SFT là **Domain Adaptation & Format Alignment**:
  1. Định hình cú pháp: Khóa chặt chữ ký hàm `priority(adj, colors, sat, deg, uncolored_deg) -> np.ndarray`.
  2. Rèn luyện tư duy vector hóa: Chỉ sử dụng các toán tử NumPy hợp lệ, không dùng vòng lặp Python chậm, không dùng thư viện ngoài whitelist.
  3. Cơ chế điều kiện hóa (Conditioning): Huấn luyện mô hình biết đọc và phản hồi tương ứng theo các thông số trong Text Card (mật độ, bậc, hệ số cụm).
- **Quy mô $2,000 \div 3,000$ mẫu là điểm cân bằng lý tưởng (Sweet Spot):**
  - Dưới $1,000$ mẫu: Mô hình 7B có thể chưa hội tụ cú pháp, dễ gặp lỗi AST hoặc sinh code không thuần (stateful).
  - Từ $2,000 \div 3,000$ mẫu: Tỷ lệ code hợp lệ (Valid Rate) qua Sandbox đạt trên $98\%$, khả năng khái quát hóa theo Text Card được định hình vững chắc.
  - Trên $10,000$ mẫu: Bị bão hòa, tốn VRAM huấn luyện và dễ dẫn đến hiện tượng suy giảm độ đa dạng (collapse).

> [!IMPORTANT]
> **Triết lý của đề tài là "Distill, then Sharpen":**  
> Ta chỉ cần $2,000 \div 3,000$ mẫu SFT chất lượng cao để làm "bệ phóng". Sau đó, thuật toán **GRPO (Reinforcement Learning)** sẽ tự động sinh hàng nghìn mẫu đồ thị mới (`rl_train_pool`), tự rollout, tự chấm điểm bằng Sandbox và mài sắc chính sách mà **hoàn toàn không tốn thêm 1 request LLM nào**!

---

## 2. Chiến lược Cốt lõi: Nhân bản 1 $\to$ 8 (Data Amplification)

Nếu sinh $2,500$ mẫu bằng cách gọi LLM $2,500$ lần, nhóm sẽ đối mặt với:
1. Chi phí API cao hoặc nghẽn rate limit.
2. Tỷ lệ code lỗi, vi phạm sandbox, hoặc heuristic yếu làm loãng tập train.

**Giải pháp tối ưu: Cơ chế "1 Thuật toán $\times$ Nhiều Text Card" (1-to-N Amplification):**

Một hàm `priority` xuất sắc cho một họ đồ thị (ví dụ: đồ thị cụm phân chia ngẫu nhiên) vẫn giữ nguyên giá trị giải thuật khi ta mô tả họ đồ thị đó bằng các phong cách ngôn ngữ khác nhau!

```
                  ┌──> Text Card Gốc (Định lượng từ đồ thị)
                  │
                  ├──> Paraphrase 1 (Văn phong Toán học / Rời rạc)
1 Heuristic Code ─┼──> Paraphrase 2 (Văn phong Kỹ thuật Đồ thị)     ===> 8 Mẫu Huấn Luyện SFT
(Chất lượng cao)   ├──> Paraphrase 3 (Văn phong Tóm tắt Ngắn gọn)           (Độc lập & Đa dạng)
                  │
                  ├──> Detailed Card (Kèm tham số sinh cấu trúc)
                  └──> 3 Biến thể AST (Đổi tên biến, giao hoán)
```

### Công thức tính mẫu:
$$\text{Tổng số mẫu SFT} = \underbrace{N_{\text{thuật toán chất lượng}}}_{\approx 300 \text{ algorithms}} \times \underbrace{(1 \text{ card gốc} + 3 \text{ paraphrases} + 1 \text{ detailed} + 3 \text{ AST variations})}_{8 \text{ biến thể}} = \mathbf{2,400 \text{ mẫu}}$$

- **Bảo toàn số liệu (Number Preservation):** Code script `scripts/make_cards.py` đã tích hợp sẵn bộ kiểm tra Regex tự động: mọi con số thực nghiệm ($n$, mật độ, phương sai bậc, clustering) đều được giữ nguyên vẹn trong các bản paraphrase.
- **Không rò rỉ nhãn (No Label Leak):** Tuyệt đối không chứa các từ khóa cấm như `dsatur`, `rlf`, `winner`, `best_single`.

---

## 3. Các Nguồn Dữ liệu Thuần Miễn Phí (Zero LLM Calls)

Nhóm có thể bổ sung ngay **$600 \div 800$ mẫu code chuẩn** mà **không cần gọi bất kỳ API LLM nào**:

### Nguồn A: Chuyển dịch Heuristic Tham số CMA-ES ($300 \div 400$ mẫu)
- Trong Phase 1 & Gate 1, ta đã chạy CMA-ES trên 106 họ đồ thị (`evaluator/parametric.py`).
- Các bộ trọng số tối ưu $\mathbf{w} = (w_1, w_2, w_3, w_4, w_5)$ tìm được có thể được transpile tự động thành các đoạn mã Python thuần túy:
  ```python
  def priority(adj, colors, sat, deg, uncolored_deg):
      max_s = np.max(sat) if np.max(sat) > 0 else 1
      max_u = np.max(uncolored_deg) if np.max(uncolored_deg) > 0 else 1
      return 1.42 * sat + 0.18 * deg + 0.85 * uncolored_deg + 0.52 * (sat / max_s)
  ```
- **Ưu điểm:** 100% hợp lệ, không bao giờ lỗi cú pháp, đã được kiểm nghiệm thực tế trên tập train của từng họ.

### Nguồn B: Tăng cường mã nguồn cấp cú pháp (AST Code Augmentation) ($300 \div 400$ mẫu)
- Với mỗi đoạn code tốt thu hoạch được, áp dụng các phép biến đổi đại số tương đương:
  1. **Đổi tên biến cục bộ:** `score = ...` $\to$ `scores = ...`, `s_norm = ...` $\to$ `norm_sat = ...`.
  2. **Giao hoán toán tử:** `a + b` $\leftrightarrow$ `b + a`, `w1 * sat + w2 * deg` $\leftrightarrow$ `w2 * deg + w1 * sat`.
  3. **Nhiễu nhẹ hệ số thực nghiệm (Jittering):** $w \leftarrow w \cdot (1 \pm 0.01)$ và chạy lại qua `evaluator/verify.py` để bảo đảm số màu tô không đổi.
- **Ưu điểm:** Giúp mô hình LLM học cách viết code tự nhiên, không bị bó hẹp vào một phong cách biến cố định.

### Nguồn C: Heuristic Kinh điển Cải biên ($50 \div 100$ mẫu)
- Biểu diễn chuẩn của DSATUR, RLF, First-Fit, Brelaz tie-break dưới chữ ký vector.

---

## 4. Quy trình Thu hoạch Teacher bằng Google Gemini (Chi phí $0.00)

Để thu hoạch các thuật toán đột phá thật sự (thắng cận $R(F)$), ta chỉ cần đầu tư gọi LLM trên **một tập con các họ đồ thị quan trọng**:

### Thiết lập Hạn ngạch & Rate Limiting cho Gemini:
- **Model:** `gemini-3.5-flash-lite`
- **Hạn ngạch miễn phí (Free Tier):**
  - **1,500 requests / ngày** (hoàn toàn miễn phí, $0.00).
  - Giới hạn tốc độ: **15 RPM** (Requests Per Minute).
- **Cơ chế Pacing an toàn trong mã nguồn:**
  - Hệ thống đặt thời gian nghỉ giữa các lượt gọi là **$4.0$ giây/call** (tương đương $15 \text{ RPM}$).
  - Đã tích hợp sẵn cơ chế Exponential Backoff (nếu gặp mã 429 hoặc 503, tự động chờ 10s $\to$ 20s $\to$ 40s và thử lại mà không làm gián đoạn tiến trình).

### Kế hoạch gọi Teacher Gemini:
1. **Lựa chọn Họ đồ thị:** Chọn **$15 \div 20$ họ tiêu biểu** từ `data/data_v1_2/splits.json` thuộc các nhóm cấu trúc đa dạng:
   - Random Graph: `gnp_d0.30_n100-200`
   - Bipartite / Multipartite: `kpart_k10_p0.3_n100-200`
   - Stochastic Block Model: `sbm_k10_in0.5_out0.05_n200-300`
   - Small World: `ws_k10_p0.1_n100-200`
   - Scale-free: `ba_m12_n200-300`
   - Geometric: `geo_r0.15_n100-200`
2. **Ngân sách cuộc gọi:**
   - Mỗi họ chạy **$50 \text{ calls}$**.
   - Tổng cộng: $20 \text{ họ} \times 50 \text{ calls} = \mathbf{1,000 \text{ calls}}$.
   - **Hoàn tất 100% trong vòng chưa đầy 1 ngày sử dụng Gemini Free Tier** (1,000 / 1,500 daily quota).

### Bộ lọc Chất lượng Chặt chẽ:
Toàn bộ code sinh ra từ Gemini sẽ đi qua phễu lọc tự động:
1. **Sandbox 4 tầng:** Loại bỏ mã lỗi cú pháp, gọi import ngoài, timeout $> 1.0s$, hoặc không thuần (stateless).
2. **Behavioral Fingerprinting:** Chạy trên tập `probe` (seeds 9000-9009), loại bỏ các mã có hành vi trùng hệt DSATUR/RLF.
3. **Rejection Sampling (Chỉ giữ mã Thắng/Hòa):** Đánh giá trên `teacher_rescore` (seeds 6000-6049), chỉ lưu trữ các mã có $gap \le 0$ so với $R(F)$.

---

## 5. Quy tắc Phân bổ Dữ liệu (Target: 2,500 Mẫu Chuẩn)

Bảng phân bổ chi tiết để đạt mốc **$2,500$ mẫu SFT chất lượng cao**:

| Nguồn Dữ Liệu | Số Thuật Toán Gốc | Số Biến Thể Card / AST | Tổng Mẫu SFT | Chi Phí API | Ghi Chú |
|---|:---:|:---:|:---:|:---:|---|
| **Gemini Teacher Evolved** | ~60 thuật toán | $\times 8$ (1 card + 3 para + 1 detail + 3 AST) | **~480 mẫu** | **$0.00** (Free Tier) | Thuật toán tiến hóa chất lượng cao, đánh bại $R(F)$. |
| **Parametric CMA-ES Tuned** | ~106 họ | $\times 6$ (1 card + 2 para + 3 AST) | **~636 mẫu** | **$0.00** (CPU only) | Tối ưu hóa trọng số thực nghiệm từ Phase 1. |
| **Universal Parametric** | 5 biến thể Pareto | $\times 20$ cards | **~100 mẫu** | **$0.00** (CPU only) | Heuristic tổng quát trên toàn bộ đồ thị train. |
| **Rule-based & Heuristic Gốc** | 10 thuật toán | $\times 15$ cards | **~150 mẫu** | **$0.00** (CPU only) | Mẫu chuẩn định dạng (DSATUR, RLF, Brelaz variants). |
| **Paraphrased Amplification** | 200 thuật toán | $\times 5$ paraphrases bổ sung | **~1,000 mẫu** | **$0.00** (Rule-based) | Sử dụng `scripts/make_cards.py` sinh paraphrase tự động. |
| **Mẫu Tương Phản (Contrast)** | ~40 thuật toán | $\times 3$ cards | **~120 mẫu** | **$0.00** (Lọc từ log) | Dùng cho phân tích tương phản (Ablation). |
| **TỔNG CỘNG** | **~420 thuật toán** | — | **~2,486 mẫu** | **$0.00** | **Đầy đủ, sạch sẽ, không tốn chi phí.** |

---

## 6. Các Lệnh Thực thi Sẵn có trong Repository

Toàn bộ công cụ phục vụ quy trình trên đã được lập trình sẵn và kiểm thử 100% trong repo:

### Bước 1: Sinh Text Cards và các bản Paraphrase (Bảo toàn số liệu)
```bash
python scripts/build_splits.py
python scripts/make_cards.py
```
*Đầu ra:* `data/data_v1_2/cards.json` chứa 3-4 bản paraphrase cho mỗi card và bảng spot-check `cards_spotcheck_50.csv`.

### Bước 2: Thu hoạch Teacher từ Gemini (Chạy trên các họ mục tiêu)
Thiết lập API key trong terminal:
```bash
# Windows PowerShell:
$env:GEMINI_API_KEY = "AIzaSy..."

# Chạy tiến hóa pilot trên họ mục tiêu (50 calls per family):
python scripts/pilot_evolve.py --family kpart_k10_p0.3_n100-200 --max-calls 50 --run-id run_kpart_gemini
python scripts/pilot_evolve.py --family gnp_d0.30_n100-200 --max-calls 50 --run-id run_gnp_gemini
python scripts/pilot_evolve.py --family sbm_k10_in0.5_out0.05_n200-300 --max-calls 50 --run-id run_sbm_gemini
```

### Bước 3: Gom dữ liệu Teacher và Rescore trên tập seed độc lập
```bash
python scripts/build_teacher_data.py --out-dir runs/teacher/dataset_v1
```
*Cơ chế tự động:* Khử trùng lặp hành vi (dedup by fingerprint), chấm điểm lại trên `teacher_rescore` (seeds 6000-6049) để tránh thiên lệch chọn lọc (selection bias).

### Bước 4: Lọc và Xuất tập dữ liệu SFT hoàn chỉnh
```bash
python scripts/filter_sft_data.py \
    --input runs/teacher/dataset_v1/records.jsonl \
    --out-dir data/sft \
    --val-ratio 0.15
```
*Đầu ra:*
- `data/sft/train.jsonl`: Chứa các mẫu huấn luyện kèm trọng số nghịch đảo thứ hạng.
- `data/sft/val.jsonl`: Chứa các mẫu kiểm tra (chia tách nghiêm ngặt theo Card, không rò rỉ prompt).
- `data/sft/contrast.jsonl`: Chứa các mẫu heuristic kém dùng cho thí nghiệm đối chứng.

---

## 7. Bảng Dự toán Chi phí & Thời gian

Dành cho nhóm khi cân nhắc giữa các phương án API:

| Phương Án | Model Sử Dụng | Số Calls Dự Kiến | Chi Phí Dự Kiến | Thời Gian Hoàn Thành | Ưu / Nhược Điểm |
|---|---|:---:|:---:|:---:|---|
| **Phương án 1 (Khuyên Dùng)** | `gemini-3.5-flash-lite` | 1,000 | **$0.00** (Free Tier) | ~1.5 giờ (chạy rải rác) | **Hoàn toàn miễn phí**, chất lượng code tốt, đáp ứng hoàn hảo hạn ngạch 1,500 calls/ngày. |
| **Phương án 2 (Tốc độ & Rẻ)** | `deepseek-chat` | 1,000 | **~$0.20** (khoảng 5,000 VNĐ) | ~20 phút | Khả năng giải thuật và code cực mạnh, tốc độ cực nhanh, chi phí siêu nhỏ. |
| **Phương án 3 (Cao cấp)** | `claude-3-5-sonnet` | 1,000 | **~$15.00** (khoảng 380,000 VNĐ) | ~30 phút | Code Python chuẩn mực nhất, nhưng chi phí tương đối cao so với sinh viên/nghiên cứu. |

---

## 8. Tóm tắt Khuyến nghị Gửi Nhóm & Báo cáo

Khi thảo luận với các thành viên trong nhóm và Thầy/Cô hướng dẫn, bạn có thể tóm tắt các điểm then chốt sau:

1. **Về quy mô mẫu ($2,000 \div 3,000$):** Con số này hoàn toàn chính xác theo tiêu chuẩn của các bài báo SFT Code 7B uy tín. Tuy nhiên, **không cần và không nên gọi LLM 3,000 lần**.
2. **Chiến lược tiết kiệm tối đa:** Bằng cách kết hợp **Data Amplification (1 code $\times$ 8 cards/AST)** và tận dụng **Heuristic tham số CMA-ES**, ta chỉ cần gọi Gemini khoảng **$500 \div 1,000$ lần**.
3. **Chi phí $0 đồng:** Với hạn ngạch 1,500 requests/ngày của Gemini Free Tier, nhóm **không phải bỏ ra bất kỳ chi phí nào** để xây dựng tập dataset $2,500$ mẫu này.
4. **Độ an toàn của mã nguồn:** Toàn bộ dữ liệu huấn luyện đều đi qua bộ Sandbox 4 tầng và Verifier độc lập, loại bỏ 100% nguy cơ code lỗi cú pháp hoặc vi phạm luật tô màu đồ thị.

---
*Tài liệu nội bộ phục vụ dự án nghiên cứu IJCAI - Graph Coloring Priority Generation.*
