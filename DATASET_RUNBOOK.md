# SỔ TAY HƯỚNG DẪN SINH DATASET SFT (DATASET GENERATION RUNBOOK)

**Dành cho:** Thành viên nhóm phụ trách sinh dữ liệu huấn luyện SFT  
**Dự án:** *Distill, then Sharpen: Amortizing Evolutionary Heuristic Search into a Family-Conditioned Code LLM*  
**Mục tiêu đầu ra:** Bộ file huấn luyện đặt tại thư mục `data/sft/` bao gồm:
- `data/sft/train.jsonl`: Tập mẫu huấn luyện (Target: $2,000 \div 2,500$ mẫu).
- `data/sft/val.jsonl`: Tập mẫu kiểm định độc lập (không trùng card với train).
- `data/sft/contrast.jsonl`: Tập mẫu đối chứng chất lượng thấp.

---

## 1. Yêu cầu Hệ thống & Cài đặt Môi trường (2 phút)

Không cần máy có GPU. Quá trình sinh dataset chạy hoàn toàn trên CPU thông thường (Windows, Linux, hoặc macOS).

### Bước 1: Mở Terminal tại thư mục gốc của project
```bash
# Di chuyển vào thư mục code dự án
cd "d:/Graph Coloring"  # (hoặc đường dẫn thư mục code trên máy bạn)
```

### Bước 2: Tạo và kích hoạt môi trường ảo Python (Python 3.11+)
```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt trên Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Kích hoạt trên Linux / macOS:
source .venv/bin/activate

# Cài đặt các thư viện cần thiết:
pip install -r requirements.txt
```

### Bước 3: Kiểm tra nhanh môi trường
Chạy lệnh kiểm thử sau (chỉ mất ~5 giây):
```bash
pytest tests/test_card_generation.py tests/test_seeds.py -q
```
*Kỳ vọng:* Hiển thị `... passed` (màu xanh lá cây) là môi trường đã chuẩn 100%.

---

## 2. Thiết lập API Key Google Gemini (Miễn Phí)

Dự án sử dụng Google Gemini (`gemini-3.5-flash-lite`), hoàn toàn miễn phí (**1,500 requests/ngày**).

1. Truy cập [Google AI Studio](https://aistudio.google.com) và tạo một API Key miễn phí (mất 30 giây).
2. Nạp API Key vào biến môi trường trong Terminal của bạn:

**Trên Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "AIzaSy..."
```

**Trên Linux / macOS / Git Bash:**
```bash
export GEMINI_API_KEY="AIzaSy..."
```

> [!CAUTION]
> **Quy tắc bảo mật:** Tuyệt đối không paste API Key trực tiếp vào code hay commit lên Git. Hệ thống sẽ tự động đọc biến môi trường `$env:GEMINI_API_KEY` một cách bảo mật.

---

## 3. CÁCH 1: Chạy Tự Động Trọn Gói (Khuyên Dùng - 1 Dòng Lệnh)

Nhóm đã đóng gói toàn bộ quy trình 5 bước vào script `scripts/run_dataset_pipeline.py`.

### A. Chạy thử nghiệm kiểm tra pipeline (Mock Mode - Mất 15 giây, $0 API)
Trước khi gọi API thật, hãy chạy thử cờ `--mock` để chắc chắn toàn bộ mắt xích hoạt động trơn tru:
```bash
python scripts/run_dataset_pipeline.py --mock --max-calls-per-family 3
```
*Kết quả:* Script sẽ tự động chia split, sinh cards, chạy tiến hóa giả lập, gom và lọc ra các file mẫu trong `data/sft/`.

---

### B. Chạy sinh dataset thật với Gemini (Production Mode)
Sau khi test thử nghiệm OK, bạn chỉ cần gõ đúng 1 lệnh để chạy thật:
```bash
python scripts/run_dataset_pipeline.py --max-calls-per-family 50
```

- **Tiến trình tự động thực hiện:**
  1. `build_splits.py`: Phân chia 106 họ đồ thị thành 6 tập split chuẩn khoa học.
  2. `make_cards.py`: Sinh thẻ mô tả đồ thị (Text Cards) kèm các bản diễn giải tương đương (Paraphrases).
  3. `pilot_evolve.py`: Gọi Gemini tiến hóa giải thuật trên các họ đồ thị mục tiêu (có cơ chế nghỉ an toàn 4 giây/call để không bao giờ bị khóa API).
  4. `build_teacher_data.py`: Tự động lọc qua Sandbox 4 tầng, khử trùng lặp hành vi trên tập `probe` (seeds 9000-9009) và chấm điểm lại trên tập `teacher_rescore` (seeds 6000-6049).
  5. `filter_sft_data.py`: Gán trọng số mẫu theo thứ hạng và xuất ra định dạng huấn luyện SFT.

- **Thời gian hoàn thành:** Khoảng $30 \div 45$ phút (chạy ngầm, bạn có thể để máy tự chạy).

---

## 4. CÁCH 2: Chạy Từng Bước Thủ Công (Khi muốn kiểm soát chi tiết)

Nếu bạn muốn chạy rải rác từng bước hoặc sinh thêm dữ liệu cho từng họ đồ thị riêng biệt:

### Bước 4.1: Sinh Splits và Text Cards
```bash
python scripts/build_splits.py
python scripts/make_cards.py
```
*File sinh ra:* `data/data_v1_2/splits.json` và `data/data_v1_2/cards.json`.

### Bước 4.2: Chạy tiến hóa Teacher cho từng họ đồ thị
Bạn có thể chạy lần lượt các họ đồ thị tiêu biểu:
```bash
# 1. Họ đồ thị Multipartite (đặc tính cấu trúc k-phân):
python scripts/pilot_evolve.py --family kpart_k10_p0.3_n100-200 --max-calls 50 --run-id gen_kpart

# 2. Họ đồ thị ngẫu nhiên Erdos-Renyi:
python scripts/pilot_evolve.py --family gnp_d0.30_n100-200 --max-calls 50 --run-id gen_gnp

# 3. Họ đồ thị phân cụm cộng đồng (Stochastic Block Model):
python scripts/pilot_evolve.py --family sbm_k10_in0.5_out0.05_n200-300 --max-calls 50 --run-id gen_sbm

# 4. Họ đồ thị thế giới nhỏ (Watts-Strogatz):
python scripts/pilot_evolve.py --family ws_k10_p0.1_n100-200 --max-calls 50 --run-id gen_ws

# 5. Họ đồ thị hình học ngẫu nhiên (Geometric):
python scripts/pilot_evolve.py --family geo_r0.15_n100-200 --max-calls 50 --run-id gen_geo

# 6. Họ đồ thị mạng không quy mô (Barabasi-Albert):
python scripts/pilot_evolve.py --family ba_m12_n200-300 --max-calls 50 --run-id gen_ba
```

### Bước 4.3: Gom dữ liệu Teacher và Rescore
```bash
python scripts/build_teacher_data.py --out-dir runs/teacher/dataset_v1
```
*File sinh ra:* `runs/teacher/dataset_v1/records.jsonl` và `summary.json`.

### Bước 4.4: Lọc và xuất tập SFT chuẩn (~2,500 mẫu)
```bash
python scripts/build_full_sft_dataset.py --target-samples 2500 --out-dir data/sft
```
*Cơ chế tự động:* Script sẽ tự động lấy các thuật toán đã tiến hóa thành công của Gemini, kết hợp với bộ Paraphrases đa dạng và các biến thể cú pháp AST của 106 họ để xuất ra đủ **~2,500 mẫu SFT** chuẩn chỉnh.

---

## 5. Nghiệm Thu & Kiểm Tra Kết Quả

Sau khi tiến trình chạy xong, bạn kiểm tra xem các file kết quả đã xuất hiện đầy đủ trong thư mục `data/sft/`:

```bash
# Xem số dòng của các file đã sinh:
python -c "import pathlib; [print(f.name, sum(1 for _ in open(f, encoding='utf-8')), 'samples') for f in pathlib.Path('data/sft').glob('*.jsonl')]"
```

*Kết quả chuẩn cần đạt:*
- `train.jsonl`: $\ge 1,500 \div 2,500$ dòng.
- `val.jsonl`: $\ge 200 \div 400$ dòng.
- `contrast.jsonl`: $\ge 50 \div 100$ dòng.

Mỗi dòng trong file JSONL là một cặp dữ liệu chuẩn mực gồm:
- `input_text`: Prompt hoàn chỉnh chứa chỉ dẫn hệ thống, ràng buộc NumPy và Text Card của họ đồ thị.
- `target_code`: Mã Python chuẩn mực của hàm `priority(adj, colors, sat, deg, uncolored_deg)`.
- `weight`: Trọng số nghịch đảo theo chất lượng thứ hạng ($1.0, 0.5, 0.33...$).

---

## 6. Xử lý Sự cố & Tình huống Thường gặp (Troubleshooting)

### Q1: Bị ngắt điện, tắt máy hoặc mất mạng giữa chừng thì sao?
**Đáp:** Hệ thống có tính năng **Resume tự động**. Bạn chỉ cần gõ lại đúng dòng lệnh ban đầu:
`python scripts/run_dataset_pipeline.py --max-calls-per-family 50`  
Hệ thống sẽ tự nhận diện các cuộc gọi LLM và thuật toán đã lưu từ trước, bỏ qua các bước đã hoàn thành và tiếp tục chạy phần còn lại mà không làm mất dữ liệu cũ.

### Q2: Gặp thông báo `HTTP Error 429: Resource Exhausted`?
**Đáp:** Nghĩa là bạn đã gửi request quá nhanh hoặc hết hạn ngạch ngày.
- Code dự án đã tích hợp sẵn **Exponential Backoff**: Nó sẽ tự động dừng chờ 10s $\to$ 20s $\to$ 40s rồi tự thử lại. Bạn không cần làm gì cả.
- Nếu tài khoản Google AI Studio bị chạm trần 1,500 calls/ngày, bạn chỉ cần tạo 1 API key mới ở 1 tài khoản Google khác và export lại key.

### Q3: Muốn chạy sang model DeepSeek (khi nhóm có tài khoản nạp tiền)?
**Đáp:** 
1. Mở file `configs/pricing.json`, sửa `"model": "deepseek-chat"`.
2. Trong terminal: `$env:DEEPSEEK_API_KEY = "sk-..."`.
3. Chạy lại script bình thường. Tốc độ sẽ nhanh gấp 3 lần và chi phí chỉ khoảng $0.20 (5,000 VNĐ).

---

## 7. Đóng Gói Gửi Lại Cho Nhóm

Khi hoàn tất, bạn chỉ cần nén thư mục `data/sft/` (kích thước file nén thường chỉ khoảng $2 \div 5$ MB) và gửi lại cho thành viên phụ trách phần GPU Fine-tuning SFT / GRPO:

```bash
# Nén thư mục data/sft thành file zip (Windows PowerShell):
Compress-Archive -Path data/sft -DestinationPath sft_dataset_final.zip
```

Gửi file `sft_dataset_final.zip` kèm file tóm tắt `runs/teacher/dataset_v1/summary.json` lên Google Drive hoặc nhóm chat là bạn đã hoàn thành xuất sắc nhiệm vụ!
