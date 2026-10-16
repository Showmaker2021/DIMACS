# PLAN_PHASE6_ONWARD.md
## Distill, then Sharpen cho Graph Coloring: kế hoạch triển khai từ Phase 6 đến hết bài báo

> **Đối tượng đọc:** các coding agent (và người điều phối là chủ dự án).
> **Cách dùng:** giao cho agent **từng task một** (hoặc một nhóm task cùng phase). Luôn dán kèm mục **[3. Quy tắc chung]** và mục **[2. Quyết định đã chốt]** vào đầu prompt. Mỗi task có sẵn: Mục tiêu, Đầu vào, Việc cần làm, Đầu ra, Nghiệm thu, Test.
> **Trạng thái repo khi bắt đầu:** Phase 1-5 đã xong, 52/52 unit test xanh, `NOTES.md` là nhật ký chính, thư mục `llm4ad/` (có EoH, ReEvo) đã có sẵn trong repo.

---

## 0. Mục lục

1. Bối cảnh và trạng thái hiện tại
2. Quyết định đã chốt (agent KHÔNG được tự đổi)
3. Quy tắc chung (dán vào đầu mọi prompt)
4. Cấu trúc thư mục mục tiêu
5. Sổ đăng ký seed (SEEDS.md)
6. Phase 6: Pilot evolution thật và Gate 2
7. Phase 7: Baseline EoH/ReEvo (llm4ad) trên cùng sandbox
8. Phase 8: Dữ liệu teacher quy mô lớn
9. Phase 9: Harness đánh giá, baseline D7, arm A0
10. Phase 10: SFT (arm A1)
11. Phase 11: GRPO (arm A2, A3, A4)
12. Phase 12: Ablation và phân tích
13. Phase 13: Tái lập và viết bài
14. Đặc tả dùng chung: schema record, prompt SFT, công thức reward
15. Đồ thị phụ thuộc và thứ tự chạy song song
16. Cổng quyết định (Decision Gates) và tiêu chí đặt trước
17. Rủi ro và cách phát hiện sớm
18. Câu hỏi mở cần chủ dự án trả lời
19. Mẫu prompt giao việc cho agent

---

## 1. Bối cảnh và trạng thái hiện tại

### 1.1 Ý tưởng bài báo
Fine-tune một code LLM 7B (Qwen2.5-Coder-7B-Instruct + LoRA) để, **chỉ đọc một "thẻ" (card) mô tả họ đồ thị**, viết ra ngay một hàm priority cho thuật toán tô màu tham lam, **không cần tiến hóa lúc triển khai**. Công thức huấn luyện hai bước:

1. **Distill:** SFT trên heuristic do tiến hóa offline (teacher) tìm được.
2. **Sharpen:** GRPO với reward kiểm chứng được (số màu thật, có phạt thời gian, thưởng đa dạng, KL neo về policy SFT).

Claim: bộ sinh có điều kiện theo họ đồ thị dịch chuyển đường Pareto "chất lượng vs số lượt gọi LLM" sang trái so với tiến hóa theo từng họ, và ta mô tả chính xác khi nào nó thất bại. Có **pivot rule**: nếu H1 sai, bài chuyển thành negative-result study bằng cùng bộ thí nghiệm.

### 1.2 Những gì đã có (Phase 1-5)
- Framework tô màu tham lam theo API vectorized: `priority(adj, colors, sat, deg, uncolored_deg) -> scores`.
- DSATUR, RLF (cả bản vertex-level lẫn **vectorized thật** nhanh hơn ~192x, cho kết quả tô màu giống hệt).
- `evaluator/verify.py`: kiểm tra độc lập tính hợp lệ coloring.
- Dataset v1.0 đóng băng: 106 họ, 11 loại cấu trúc; seed tách: train 0-99 (baseline dùng 0-19), val 1000-1019, test 2000-2049, probe 9000-9009.
- Metadata v1.1 tại `data/data_v1_1/`: `scale_free` chuyển sang `train_pool`; `small_world` là `heldout_type`; có `card_detailed`; thêm `teacher_val` 3000-3019 và `test_pilot` 4000-4199.
- **Gate A:** CMA-ES 5 trọng số kém R(F) = min(DSATUR, RLF) tới -5.87% (trên 99 họ) → tinh chỉnh tham số không đủ, cần LLM sinh logic mới.
- Chẩn đoán D1' (TabuCol), D4 (RLF tham số), D5 (rollouts), D7 (bộ phân loại thẻ → {DSATUR, RLF}, accuracy ~82%, regret ~0.3-0.4%), D8 (benchmark vectorized).
- Sandbox 4 lớp (AST whitelist, namespace cô lập, kiểm tra purity/statelessness, worker process với hard-timeout, lọc `*_API_KEY`, `OMP/MKL=1`, giới hạn RSS 1.5 GB).
- `gate/evolved/loop.py`: engine tiến hóa hợp nhất, dedup hành vi theo vector 20 chiều số màu train, toán tử 40% explore (T=0.9) / 30% crossover (T=0.7) / 30% refine (T=0.5), pacing 5s, backoff HTTP 429.
- Giao thức chọn lọc khóa trên `teacher_val`, R(F)_teacher_val đã khóa cho 6 họ pilot:

| Họ pilot | R(F) teacher_val | Thuật toán thắng | Vai trò |
|---|---|---|---|
| `kpart_k10_p0.3_n100-200` | 12.850 | RLF | mục tiêu |
| `gnp_d0.30_n100-200` | 14.500 | RLF | mục tiêu |
| `sbm_k10_in0.5_out0.05_n200-300` | 9.550 | RLF | mục tiêu |
| `ws_k10_p0.1_n100-200` | 6.900 | DSATUR | mục tiêu |
| `interval_L0.10_n100-200` | 24.300 | DSATUR | **Control 1** |
| `ba_m12_n200-300` | 9.150 | DSATUR | **Control 2** |

- `scripts/pilot_test.py` đo `test_pilot` đúng một lần (khóa `.done`).

### 1.3 Phần còn thiếu (nội dung của file này)
Chạy pilot thật → Gate 2 → baseline llm4ad → dữ liệu teacher lớn → harness đánh giá → SFT → GRPO → ablation → cost model → viết bài.

---

## 2. Quyết định đã chốt (agent KHÔNG được tự đổi)

**D-1. API là bản vectorized.**
Hàm do LLM sinh luôn có dạng `priority(adj, colors, sat, deg, uncolored_deg) -> scores` (không dùng dạng dict `priority_func(v, graph, color_map, candidates)` trong proposal gốc). Prompt SFT, dữ liệu teacher, reward GRPO, harness đều dùng API này. Chữ ký, kiểu dữ liệu (mảng NumPy) và ngữ nghĩa từng tham số phải được **trích nguyên văn từ code hiện có** của framework, không tự suy diễn.

**D-2. Mốc so sánh chính là R(F) = min(DSATUR, RLF), không chỉ DSATUR.**
Lý do: Gate A và D7 cho thấy chọn DSATUR/RLF theo thẻ đã gần tối ưu. Model chỉ học "chọn DSATUR hay RLF" thì không có đóng góp khoa học. Mọi bảng kết quả báo cáo **cả** gap vs R(F) và gap vs DSATUR, nhưng claim chính dựa trên R(F).

**D-3. Baseline cho reward cấp đồ thị.**
Với mỗi card, định nghĩa `base_alg(card) = argmin(mean colors DSATUR, mean colors RLF)` tính một lần trên `teacher_val`. Baseline cấp đồ thị là số màu của `base_alg` trên chính đồ thị đó (**không** dùng oracle min cấp từng đồ thị để tránh baseline quá khắc nghiệt và khó học). Khi báo cáo, có thêm cột "vs oracle per-graph min" để minh bạch.

**D-4. Classifier D7 là baseline bắt buộc** trong mọi bảng kết quả, trên mọi split (kể cả extrapolation, compositional, held-out type).

**D-5. Gate 2 phải qua trước khi tốn tiền SFT/GRPO** (xem mục 16). Teacher không thắng R(F) thì student không thể thắng.

**D-6. CALM không có trong llm4ad.** Giai đoạn đầu **bỏ CALM** khỏi baseline và ghi rõ lý do trong paper. Chỉ làm "CALM-lite" (Task 7.3) nếu chủ dự án yêu cầu sau.

**D-7. Tên gọi trung thực.** Engine `gate/evolved/loop.py` là "our evolution engine", **không** gọi là EoH hay ReEvo trong bất kỳ bảng/hình/văn bản nào. EoH/ReEvo chỉ là kết quả chạy qua llm4ad (Phase 7).

**D-8. Sandbox của dự án là nguồn chân lý duy nhất để chấm điểm.** Khi chạy baseline llm4ad, luôn chấm bằng sandbox 4 lớp và `evaluator/verify.py` của dự án, không dùng evaluator/sandbox của llm4ad.

**D-9. Chỉ giữ record "thắng hoặc hòa R(F)" làm dữ liệu SFT chính** (Task 8.3). Record thua R(F) chỉ dùng làm contrast và gắn nhãn riêng, vì SFT trên heuristic thua R(F) sẽ dạy model chép lại baseline.

**D-10. Phân tách môi trường chạy.** Code chạy trên máy local (Windows) và code chạy trên GPU box phải tách rõ (`train/` cần GPU; `scripts/`, `eval/` mặc định chạy được CPU). Mọi đường dẫn dùng `pathlib`, không hard-code `d:\...`. (Xem câu hỏi mở mục 18.)

---

## 3. Quy tắc chung (dán vào đầu MỌI prompt giao agent)

1. **Không sửa** (trừ khi task nói rõ): `data/data_v1/`, `data/data_v1_1/`, `evaluator/verify.py`, các file khóa `.done`, whitelist của sandbox, `NOTES.md` các mục cũ (chỉ được **thêm** mục mới ở cuối).
2. **Mỗi task phải kèm unit test**, và **52/52 test hiện có phải vẫn xanh**. Chạy `pytest` toàn bộ trước khi báo hoàn thành.
3. **Seed:** chỉ dùng các khoảng đã định trong mục 5. Mọi khoảng seed mới phải được ghi vào `SEEDS.md` **trước khi dùng**, kèm test tự động xác nhận không chồng lấn với khoảng khác.
4. **Tập test chỉ đo một lần.** Mọi tập test/test_pilot/DIMACS phải có cơ chế khóa (`.done` + hash cấu hình). Chạy lần hai phải từ chối trừ khi có cờ `--force-unlock` được ghi log.
5. **Bí mật:** không in, không ghi file, không commit khóa API. Log chỉ chứa tên biến môi trường, không chứa giá trị.
6. **Truy vết:** mọi kết quả ghi kèm `run_id`, config đầy đủ (JSON), git commit hash, thời điểm, phiên bản thư viện chính. Không ghi đè run cũ; mỗi run một thư mục mới.
7. **Tái lập:** mọi script nhận `--seed`; mọi nguồn ngẫu nhiên (random, numpy, torch) được seed và ghi lại.
8. **Không rò rỉ dữ liệu:** không dùng đồ thị/seed dùng để chọn lọc teacher để chấm điểm cuối; không dùng seed của test để huấn luyện/early stopping. Có test tự động kiểm tra.
9. **Chi phí API:** trước khi chạy bất kỳ job nào gọi LLM với >100 lượt, in ước tính số lượt/token/thời gian và **dừng chờ xác nhận** (hoặc yêu cầu cờ `--yes`). Hỗ trợ `--dry-run` dùng LLM giả (mock).
10. **Khả năng chạy lại từng phần (resumable):** job dài phải checkpoint định kỳ và tiếp tục được sau khi bị ngắt (đặc biệt vòng tiến hóa và vòng GRPO).
11. **Mỗi task kết thúc** bằng: (a) mục mới trong `NOTES.md` (đã làm gì, số liệu, quyết định, vấn đề tồn đọng), (b) liệt kê file đã tạo/sửa, (c) lệnh để chạy lại.
12. **Báo cáo trung thực:** nếu kết quả xấu, ghi nguyên văn, không chỉnh tiêu chí sau khi đã thấy số.
13. **Phạm vi:** chỉ làm đúng task được giao. Phát hiện vấn đề ngoài phạm vi thì ghi vào `NOTES.md` mục "Việc phát hiện thêm", không tự sửa.
14. **Ngôn ngữ:** code, tên biến, docstring bằng tiếng Anh; `NOTES.md` và báo cáo cho chủ dự án bằng tiếng Việt.

---

## 4. Cấu trúc thư mục mục tiêu

Chỉ **thêm** các thư mục/file mới; giữ nguyên cấu trúc cũ.

```
Graph Coloring/
├── PLAN_PHASE6_ONWARD.md          # file này
├── NOTES.md                       # nhật ký (chỉ append)
├── SEEDS.md                       # sổ đăng ký seed (mới, Task 0)
├── data/
│   ├── data_v1/                   # ĐÓNG BĂNG
│   ├── data_v1_1/                 # ĐÓNG BĂNG
│   └── data_v1_2/                 # MỚI: họ biến thể, splits.json (Task 8.1)
├── evaluator/verify.py            # ĐÓNG BĂNG
├── gate/evolved/loop.py           # engine tiến hóa của ta
├── baselines/
│   ├── llm4ad_task.py             # adapter (Task 7.1)
│   ├── run_eoh.py, run_reevo.py   # (Task 7.2)
│   └── d7_classifier.py           # (Task 9.1)
├── scripts/
│   ├── pilot_evolve.py, pilot_test.py (đã có)
│   ├── gate2_report.py            # (Task 6.2)
│   ├── make_cards.py              # (Task 8.4)
│   ├── build_teacher_data.py      # (Task 8.2)
│   └── make_all_figures.py        # (Task 13.1)
├── eval/
│   ├── eval_model.py              # harness (Task 9.2)
│   ├── coverage.py                # coverage@k, best-of-k
│   └── generators/                # backend sinh mẫu: HF local, vLLM, API
├── train/
│   ├── build_sft.py               # (Task 10.1)
│   ├── sft.py                     # (Task 10.2)
│   ├── reward.py                  # (Task 11.1)
│   ├── rollout.py                 # (Task 11.2)
│   └── grpo.py                    # (Task 11.3)
├── analysis/
│   ├── cost_model.py              # (Task 12.4)
│   ├── stats.py                   # bootstrap, paired tests
│   └── failure_analysis.py
├── runs/                          # mọi kết quả chạy; KHÔNG ghi đè
│   ├── pilot/ , baselines/ , teacher/ , sft/ , grpo/ , eval/
└── tests/                         # pytest, thêm test cho từng task
```

---

## 5. Sổ đăng ký seed (SEEDS.md)

### 5.1 Khoảng đã có (đóng băng)

| Tên | Khoảng | Mục đích |
|---|---|---|
| `train` | 0-99 (baseline dùng 0-19) | huấn luyện/đánh giá train của evolution |
| `val` | 1000-1019 | early stopping RL/SFT, chọn model |
| `test` | 2000-2049 | test chính (đo một lần) |
| `teacher_val` | 3000-3019 | chọn lọc teacher |
| `test_pilot` | 4000-4199 | test pilot (khóa `.done`) |
| `probe` | 9000-9009 | fingerprint hành vi, khoảng cách thứ tự cho `r_div` |

### 5.2 Khoảng **đề xuất** cho Phase mới (agent phải ghi vào SEEDS.md và có test không chồng lấn)

| Tên | Khoảng đề xuất | Mục đích |
|---|---|---|
| `teacher_rescore` | 6000-6049 | chấm lại heuristic teacher trên đồ thị mới (tránh selection bias) |
| `rl_train_pool` | 100000-199999 | đồ thị tươi cho GRPO (lấy mẫu ngẫu nhiên theo step, ghi seed đã dùng) |
| `sft_val_probe` | 5000-5019 | val cho callback early-stop SFT theo điểm heuristic |
| `test_interp` | 7000-7049 | test nội suy |
| `test_extrap` | 7100-7149 | test ngoại suy |
| `test_comp` | 7200-7249 | test tổ hợp thuộc tính |
| `test_heldout_type` | 7300-7349 | test loại cấu trúc chưa thấy (small_world) |
| `dimacs` | không dùng seed | đồ thị DIMACS thật, đo một lần cuối |

Nếu chủ dự án đã dùng khoảng nào trong số này cho việc khác, agent phải chọn khoảng khác và ghi lý do.

---

## 6. PHASE 6: Pilot evolution thật và Gate 2

Mục tiêu phase: biết **teacher có thật sự thắng R(F) hay không** trước khi đầu tư SFT/GRPO.

### Task 6.0: Chuẩn bị chung (làm đầu tiên, nhỏ)
- Tạo `SEEDS.md` theo mục 5, kèm `tests/test_seeds.py` kiểm tra không chồng lấn giữa mọi khoảng.
- Tạo `configs/teacher_llm.yaml` (tên model, nhiệt độ cơ bản, max_tokens, rate limit, nguồn khóa API là **tên biến môi trường**).
- Đảm bảo `loop.py` ghi đủ thông tin cho cost model: mỗi lượt gọi lưu `prompt_tokens`, `completion_tokens`, `latency_s`, `operator`, `temperature`, `status`.
- **Nghiệm thu:** test seeds xanh; một lượt gọi LLM giả (mock) cho thấy log có đủ trường trên.

### Task 6.1: Chạy pilot evolution thật (6 họ × ≥3 lần chạy)
- **Mục tiêu:** có 18+ lượt chạy tiến hóa (6 họ × ≥3 run seed khác nhau) với cùng teacher LLM và cùng budget.
- **Đầu vào:** 6 họ pilot ở mục 1.2; `train` seeds 0-19 cho đánh giá trong vòng lặp; `teacher_val` 3000-3019 để chọn lọc.
- **Việc cần làm:**
  1. Viết/hoàn thiện `scripts/pilot_evolve.py --family F --run-seed S --budget N` (đã có, kiểm tra lại).
  2. Budget cố định mỗi run (đề xuất N = 200 lượt gọi/họ/run; chủ dự án có thể chỉnh, nhưng phải **giống nhau giữa các họ và giữa các baseline ở Phase 7**).
  3. Chạy tuần tự có pacing; resumable; mock `--dry-run` để kiểm tra pipeline trước.
- **Đầu ra:** `runs/pilot/{family}/{run_id}/` gồm
  - `candidates.jsonl`: mọi candidate (code, vector màu train 20 chiều, trạng thái sandbox, operator, token, latency, thế hệ);
  - `best.json`: top-m theo điểm train (chưa dùng teacher_val);
  - `summary.json`: tổng lượt gọi, tổng token, tổng thời gian tường, tỉ lệ sandbox fail theo loại lỗi, số mẫu duy nhất theo fingerprint.
- **Nghiệm thu:** (a) chạy lại với cùng seed + mock LLM cho kết quả giống hệt; (b) tổng token/lượt/thời gian được ghi lại; (c) không có API key trong log.
- **Test:** test log schema; test resume sau khi kill giữa chừng.

### Task 6.2: Gate 2 report
- **Mục tiêu:** quyết định có đi tiếp hay không.
- **Việc cần làm** trong `scripts/gate2_report.py`:
  1. Với mỗi họ, gộp top ứng viên của các run → chọn top-1 **theo `teacher_val` (3000-3019)**.
  2. Đo **một lần** trên `test_pilot` (4000-4199) qua cơ chế khóa `.done`.
  3. Với mỗi họ báo cáo: số màu trung bình của top-1, DSATUR, RLF, R(F); **gap % so với R(F)** và so với DSATUR; bootstrap 95% CI (resample trên đồ thị, 10.000 lần); paired test (Wilcoxon) trên đồ thị.
  4. Báo cáo tỉ lệ heuristic "hành vi ≈ DSATUR/RLF" (dùng fingerprint trên probe: cùng vector số màu và/hoặc thứ tự giống tham chiếu); tỉ lệ sandbox fail theo loại; tỉ lệ ứng viên thắng R(F) trên `teacher_val` trong mỗi run.
  5. Tách riêng 4 họ mục tiêu và 2 control (control kỳ vọng ≈ R(F); nếu teacher "thắng" control một cách bất thường thì kiểm tra rò rỉ/bug).
- **Đầu ra:** `runs/gate2/report.md` + `report.json`, in ra bảng trên console.
- **Tiêu chí quyết định (ĐẶT TRƯỚC, không đổi sau khi thấy số):**
  - **PASS:** ≥ 2/4 họ mục tiêu có cải thiện ≥ 1.0% so với R(F) trên `test_pilot`, CI 95% không chứa 0, **và** valid-rate = 100%.
  - **MARGINAL:** đúng 1/4 họ PASS hoặc cải thiện 0.5-1.0%: dừng, báo chủ dự án, đề xuất điều chỉnh prompt/operator/budget (chạy lại tối đa một vòng).
  - **FAIL:** còn lại: dừng, nêu hai hướng (đổi bài toán/độ khó của họ; hoặc chuyển pivot negative-result).
- **Test:** test script với dữ liệu giả cho cả ba trạng thái PASS/MARGINAL/FAIL; test khóa `.done` từ chối chạy lần hai.
- **Lưu ý:** Agent **không** được tự chỉnh ngưỡng. Nếu thấy ngưỡng bất hợp lý, ghi đề xuất vào NOTES.md và hỏi chủ dự án.

---

## 7. PHASE 7: Baseline EoH/ReEvo (llm4ad) trên cùng sandbox

Chỉ bắt đầu sau khi Gate 2 không FAIL (có thể chạy song song với Phase 8.1/8.4/9.x, vì độc lập).

### Task 7.1: Adapter `baselines/llm4ad_task.py`
- **Mục tiêu:** để EoH/ReEvo của llm4ad tối ưu đúng bài toán của ta, chấm bằng sandbox và evaluator của ta.
- **Việc cần làm:**
  1. Đọc cấu trúc `llm4ad/` có sẵn: cách khai báo `Evaluation` (template program, task description, `evaluate_program`), cách EoH/ReEvo gọi.
  2. Tạo lớp Evaluation kế thừa chuẩn llm4ad; bên trong `evaluate_program` gọi **sandbox 4 lớp + verify.py của ta**, trả về điểm (đề xuất: `-mean_colors` trên tập train, vì llm4ad thường tối đa hóa).
  3. Template function và mô tả bài toán dùng **đúng API vectorized** (D-1) và cùng mô tả bài toán với prompt của `loop.py` (để so sánh công bằng).
  4. Cho phép chọn họ và seed qua tham số; dùng cùng tập `train` seeds 0-19 như `loop.py`.
- **Nghiệm thu:** một candidate cố định cho **cùng điểm số** khi chấm qua adapter và qua `loop.py` (sai số 0). Candidate vi phạm whitelist bị từ chối giống nhau ở hai đường.
- **Test:** `tests/test_llm4ad_adapter.py` gồm candidate hợp lệ, candidate vi phạm AST, candidate timeout, candidate không thuần (stateful).

### Task 7.2: Chạy EoH và ReEvo ở matched budget
- **Mục tiêu:** hai baseline tiến hóa chuẩn, cùng teacher LLM, cùng số lượt gọi LLM với Task 6.1.
- **Việc cần làm:**
  1. `baselines/run_eoh.py` và `baselines/run_reevo.py` với tham số `--family --run-seed --budget`.
  2. Cấu hình để **tổng lượt gọi LLM xấp xỉ** budget của 6.1 (ghi rõ cách đếm: mỗi request LLM là 1 lượt; ReEvo có thêm lượt reflection thì tính vào).
  3. Ghi log cùng định dạng 6.1 (đặt dưới `runs/baselines/{eoh|reevo}/{family}/{run_id}/`) để `gate2_report`-style so sánh trực tiếp.
  4. Chọn top-1 theo `teacher_val`, đo trên `test_pilot` **bằng cùng cơ chế khóa** (mỗi phương pháp một khóa riêng, nhưng cùng tập đồ thị).
- **Đầu ra:** bảng so sánh: our-engine vs EoH vs ReEvo vs R(F) trên 6 họ pilot, cùng budget.
- **Nghiệm thu:** tổng token/lượt/thời gian ghi đủ; không lỗi nạp chéo run cũ.
- **Test:** chạy mock LLM, kiểm tra đếm lượt gọi đúng.

### Task 7.3 (TÙY CHỌN, chỉ làm khi chủ dự án yêu cầu): CALM-lite
- Một vòng tiến hóa + fine-tune đơn giản hóa (mô phỏng ý tưởng CALM: fine-tune teacher trong lúc tiến hóa). Nếu làm, **gọi rõ là "CALM-lite (our simplified reimplementation)"** và liệt kê khác biệt so với bản gốc.

---

## 8. PHASE 8: Dữ liệu teacher quy mô lớn

### Task 8.1: Mở rộng danh sách họ/card và định nghĩa splits
- **Mục tiêu:** từ 106 họ đạt khoảng 300 card huấn luyện khả dụng và đủ các split theo proposal.
- **Việc cần làm:**
  1. Tạo `data/data_v1_2/` (KHÔNG sửa v1, v1_1). Thêm **generator biến thể tham số**: với mỗi loại cấu trúc, lấy mẫu `n`, mật độ, `k`/tham số đặc trưng trong lưới/khoảng; mỗi họ có định danh duy nhất và tham số sinh lưu đầy đủ.
  2. Định nghĩa thuộc tính dùng cho split compositional: `n_range`, `density_bin`, `degree_variance_bin`, `clustering_bin`, `structure_type`.
  3. Tạo `splits.json` với các nhóm:
     - `train_pool` (gồm `scale_free` theo v1.1);
     - `val`;
     - `test_interp` (giá trị tham số **chưa thấy nhưng nằm trong khoảng train**);
     - `test_extrap` (n lớn hơn hoặc mật độ cao hơn khoảng train);
     - `test_comp` (tổ hợp thuộc tính bị giữ ra, ví dụ dense+clustered khi train chỉ có dense+random và sparse+clustered; **liệt kê rõ các tổ hợp giữ ra trong file**);
     - `heldout_type` = `small_world`.
  4. Gắn mỗi họ với seed ranges của split tương ứng (mục 5.2).
- **Nghiệm thu:** `splits.json` + test tự động xác nhận: (a) không họ nào ở hai split khác nhóm (trừ quan hệ định nghĩa rõ); (b) **không tổ hợp thuộc tính held-out nào xuất hiện trong train**; (c) không chồng lấn seed; (d) `small_world` không xuất hiện trong train/val; (e) tổng số card huấn luyện được in ra (mục tiêu ~300; nếu ít hơn nêu lý do).
- **Test:** `tests/test_splits.py`.

### Task 8.2: Pipeline teacher `scripts/build_teacher_data.py`
- **Mục tiêu:** sinh tập (card, code, score, metadata) từ tiến hóa offline.
- **Việc cần làm (theo thứ tự):**
  1. Với mỗi card trong `train_pool`: chạy `loop.py` (nguồn 1). Nếu có kết quả Phase 7, **gộp thêm run EoH/ReEvo** làm nguồn 2 (ghi `teacher` tương ứng).
  2. Tính `base_alg(card)` và R(F) trên `teacher_val` (D-3).
  3. Giữ **top-m (m=3)** theo `teacher_val`; thêm 1-2 mẫu hạng trung làm contrast (gắn `role="contrast"`).
  4. **Dedup theo fingerprint hành vi** (vector số màu hoặc thứ tự đỉnh trên `probe` 9000-9009), không theo văn bản code.
  5. Chạy lại **toàn bộ sandbox + purity check** trên survivors.
  6. **Chấm lại trên seed mới** `teacher_rescore` (6000-6049) để lưu `score` không thiên lệch chọn lọc.
  7. Xuất `runs/teacher/{teacher_run_id}/records.jsonl` theo schema ở mục 14.1.
  8. Resumable theo card; log tổng chi phí (lượt gọi, token) theo card để cost model dùng.
- **Nghiệm thu:** in thống kê: số record, phân bố gap vs R(F), tỉ lệ record "≈DSATUR/RLF", phân bố theo loại cấu trúc/teacher/role. **Cảnh báo to (ghi vào NOTES.md và console) nếu >70% record không thắng R(F).**
- **Test:** pipeline trên 2-3 card giả với LLM mock; test dedup; test rescore dùng đúng seed range.

### Task 8.3: Lọc "chỉ giữ cái thắng" (D-9)
- Ở bước xuất SFT, tạo ba nhóm: `win` (gap vs R(F) < 0 trên rescore, tức tốt hơn), `tie` (|gap| trong ngưỡng nhiễu, đề xuất ±0.1 màu và CI chứa 0), `contrast`/`loss`.
- SFT chính dùng `win` ∪ `tie`; `contrast` dùng ở thí nghiệm ablation riêng (có/không có contrast).
- **Nghiệm thu:** bảng đếm số record mỗi nhóm theo split; nếu `win` < 300 record thì cảnh báo và đề xuất tăng budget/số card.

### Task 8.4: Card và paraphrase `scripts/make_cards.py`
- **Việc cần làm:**
  1. Template card cố định từ thuộc tính; **con số tính từ đồ thị đã sinh** (n min-max, mật độ đo được, phương sai bậc, hệ số clustering...), không gõ tay.
  2. 3-4 paraphrase/card bằng LLM, cache theo hash (card, paraphrase_idx, model); **kiểm tra tự động**: mọi con số trong card gốc phải xuất hiện nguyên vẹn trong paraphrase; không chứa tên họ/ID gây rò rỉ nhãn (ví dụ tên thuật toán tốt nhất).
  3. Tạo `card_detailed` (có tham số sinh) cho ablation, dùng định dạng riêng.
  4. Spot-check thủ công: xuất file CSV 50 mẫu ngẫu nhiên để chủ dự án duyệt.
- **Nghiệm thu:** test số học; test không rò nhãn; test cache (chạy lại không gọi LLM).

---

## 9. PHASE 9: Harness đánh giá, baseline D7, arm A0

### Task 9.1: Baseline D7 như arm chính thức (`baselines/d7_classifier.py`)
- Đưa bộ phân loại thẻ → {DSATUR, RLF} (Decision Tree / Logistic Regression) vào pipeline đánh giá chuẩn, huấn luyện **chỉ trên train_pool**.
- Đầu vào của D7 phải là đặc trưng suy ra từ **cùng thông tin thẻ** (thuộc tính có cấu trúc), đánh giá trên mọi split (interp, extrap, comp, heldout_type).
- **Nghiệm thu:** bảng D7 vs R(F) (regret) trên từng split; ghi rõ nếu D7 gần R(F) (đó là mốc khó nhất mà LLM phải vượt).

### Task 9.2: Harness `eval/eval_model.py`
- **Input:** backend sinh (checkpoint HF/LoRA, vLLM, hoặc API) + split + `k` + `--n-graphs`.
- **Quy trình:** với mỗi card, sinh `k` mẫu (temperature cấu hình được) → sandbox song song trên CPU (pool worker, tận dụng Lớp 4) → chấm trên đồ thị của split → ghi kết quả.
- **Metric đầu ra:**
  - valid-rate, sandbox-fail theo loại;
  - mean colors; **savings vs R(F)** và vs DSATUR (cả gap % và số màu);
  - **coverage@k**: tỉ lệ card có ít nhất 1 trong `k` mẫu *thắng* R(F) (thắng = tốt hơn có ý nghĩa, định nghĩa trong `eval/coverage.py` và cố định);
  - best-of-k chất lượng;
  - **pass@k ước lượng không chệch** khi sinh n > k mẫu (công thức Chen et al.);
  - tỉ lệ fingerprint trùng DSATUR/RLF; độ đa dạng trong card (khoảng cách thứ tự trung bình);
  - độ dài code, runtime heuristic trung bình (so với DSATUR vectorized).
- **Tính năng bắt buộc:** cache kết quả theo hash code (không chấm lại), checkpointing, khóa tập test (D theo quy tắc 4), xuất `per_card.jsonl` + `summary.json`.
- **Nghiệm thu:** chạy với backend giả sinh DSATUR/RLF → kết quả coverage = 0 (không thắng R(F)) và gap = 0; backend giả sinh code hỏng → valid-rate thấp được ghi đúng.
- **Test:** test coverage@k trên dữ liệu giả; test cache; test khóa test.

### Task 9.3: Arm A0 (base Qwen, best-of-k)
- Chạy trước SFT: Qwen2.5-Coder-7B-Instruct **không fine-tune**, cùng prompt như SFT (mục 14.2), k tới 256 trên val (và test khi đóng băng cấu hình). Dùng làm mốc dưới và tham chiếu coverage.
- **Nghiệm thu:** `runs/eval/A0/` có coverage@k cho k = 1,2,4,...,256 trên val; ghi lại thời gian và token.

---

## 10. PHASE 10: SFT (arm A1)

### Task 10.1: `train/build_sft.py`
- **Việc cần làm:** chuyển record (nhóm `win` ∪ `tie`, mục 8.3) thành mẫu huấn luyện theo prompt mục 14.2. Trọng số mẫu theo **hạng trong card** (hạng 1 cao nhất; công thức ghi rõ và cấu hình được).
- **Chia train/val theo card** (không theo paraphrase) để tránh rò rỉ; val chỉ dùng cho early stopping.
- **Nghiệm thu:** test không card nào xuất hiện ở cả train và val; test prompt đúng định dạng; xuất thống kê (số mẫu, độ dài token).

### Task 10.2: `train/sft.py`
- **Cấu hình:** Qwen2.5-Coder-7B-Instruct + LoRA (TRL/PEFT hoặc Unsloth; chọn công cụ phù hợp phần cứng ở mục 18), 2-3 epoch, lr khởi điểm ~1e-4, seed cấu hình được.
- **Callback early-stop theo điểm heuristic** (không chỉ token loss): mỗi N bước, sinh 2-4 mẫu/card trên tập con val (`sft_val_probe` 5000-5019 cho đồ thị chấm), chạy sandbox, tính savings vs R(F) trung bình; lưu checkpoint tốt nhất theo chỉ số này.
- **Ghi lại:** config, seed, loss, điểm heuristic theo bước, thời gian GPU (cho cost model).
- **Nghiệm thu:** smoke test 20 bước chạy xong; checkpoint nạp lại và sinh ra code được chấm bởi harness 9.2.

### Task 10.3: Đánh giá A1
- Chạy ≥ 3 seed huấn luyện. Dùng harness 9.2 trên val, và (sau khi cấu hình đóng băng) trên các test split.
- **Báo cáo bắt buộc:** coverage@k so với A0 (SFT có thêm "mẫu mới" không?), gap vs R(F) và vs D7, tỉ lệ ≈DSATUR/RLF.
- **Tiêu chí tiếp tục:** nếu A1 **không** vượt D7 và A0@k trên val ở bất kỳ họ mục tiêu nào, ghi cảnh báo và hỏi chủ dự án trước khi chạy GRPO.

---

## 11. PHASE 11: GRPO (arm A2, A3, A4)

### Task 11.1: `train/reward.py` (có unit test riêng)
Xem công thức chi tiết ở mục 14.3. Yêu cầu:
- `r_sav` chuẩn hóa như proposal nhưng **baseline là `base_alg(card)` (D-3)**, `c_ref` là kết quả TabuCol (D1') hoặc tốt nhất của teacher (chọn bằng cờ, mặc định lấy giá trị nhỏ hơn đã có).
- `r_time` dựa trên runtime đo bằng benchmark vectorized (D8), ngưỡng so với DSATUR vectorized (cấu hình được).
- `r_tb` **tùy chọn, mặc định tắt** ở bản đầu để giảm độ phức tạp.
- `r_div` theo khoảng cách thứ tự (Kendall-tau) trên 2 probe graph, **chỉ áp cho mẫu có reward trên median nhóm**.
- **Failure = -1** cho: AST fail, purity fail, timeout, RSS vượt, invalid coloring, ngoại lệ.
- **Test bắt buộc:** (a) deterministic với cùng input; (b) mọi loại fail cho -1; (c) tất cả mẫu cùng điểm → advantage 0 không NaN (xử lý std=0); (d) reward không vượt khoảng cho phép; (e) mẫu trùng hệt baseline cho `r_sav` ≈ 0; (f) nhóm được chấm trên **cùng** tập đồ thị.

### Task 11.2: `train/rollout.py`
- Mỗi step: lấy batch card (đề xuất 64 prompt/step như proposal, cấu hình được) → G = 8 mẫu/card → **sinh tập đồ thị tươi từ `rl_train_pool`** (seed lấy ngẫu nhiên có kiểm soát, ghi log) → chấm cả 8 mẫu trên **cùng tập đồ thị** bằng pool worker.
- Cache theo hash code trong cùng step; xử lý song song CPU; timeout theo mẫu, **không để một mẫu treo cả step**.
- **Nghiệm thu:** test rằng seeds RL không nằm trong train/val/test; test thông lượng tối thiểu (in ra mẫu/giây).

### Task 11.3: `train/grpo.py`
- Dùng TRL `GRPOTrainer` nếu phù hợp, hoặc tự viết vòng lặp tối giản khi TRL không đáp ứng same-graph group evaluation. LoRA, AdamW, **lr khởi điểm ~2e-5** (có thể tinh chỉnh), **KL penalty (beta) về policy SFT**, 2-5K step, early stop theo `val`.
- **Logging mỗi N step (bắt buộc):** từng thành phần reward; tỉ lệ fail theo loại; tỉ lệ ≈DSATUR/RLF; entropy; độ dài code trung bình; KL; coverage@k nhanh trên tập con val (ví dụ k=16). Đây là cơ chế phát hiện sớm **coverage collapse** và **reward hacking**.
- **Chốt chặn an toàn:** tự động dừng và lưu checkpoint nếu (a) tỉ lệ ≈DSATUR/RLF vượt ngưỡng cấu hình (đề xuất 90%) trong cửa sổ trượt, hoặc (b) tỉ lệ fail tăng đột biến, hoặc (c) reward trung bình tăng nhưng savings trên val giảm trong nhiều lần đánh giá liên tiếp.
- Resumable từ checkpoint (kể cả trạng thái optimizer và RNG).
- **Nghiệm thu:** smoke test 10 step với mock sandbox; log đủ trường.

### Task 11.4: Chạy các arm
- **A2:** GRPO-only (từ base Qwen, KL về base).
- **A3:** SFT → GRPO (KL về SFT).
- **A4:** A3 + diversity (`d` > 0) + KL mạnh hơn (`beta` lớn hơn A3).
- ≥ 3 seed mỗi arm; cùng budget bước cho công bằng. Lưu mọi checkpoint tốt nhất theo val.
- **Nghiệm thu:** `runs/grpo/{A2|A3|A4}/{seed}/` đủ log; bảng tóm tắt val.

---

## 12. PHASE 12: Ablation và phân tích

### Task 12.1: A6 (no-card / shuffled-card), **làm sớm**
- Chạy trên base (A0) và SFT (A1): (i) bỏ card, (ii) hoán card giữa các họ có `base_alg` khác nhau, (iii) hoán card giữa họ cùng `base_alg`.
- Kiểm **H4:** hiệu năng giảm trên họ mà heuristic tốt nhất khác DSATUR/RLF mặc định, không giảm ở họ mà R(F) đã gần tối ưu. Báo cáo theo nhóm họ.
- Vì kết quả có thể đổi câu chuyện bài báo, chạy ngay sau khi có A1.

### Task 12.2: A5 (refinement 5/20/50 lượt, kiểu CALM)
- Khởi tạo `loop.py` từ heuristic do model (A3/A4) sinh ra; cho phép 0/5/20/50 lượt LLM tinh chỉnh; vẽ **đường Pareto** "chất lượng vs số lượt gọi LLM" cùng với đường tiến hóa from-scratch (Phase 6/7) ở các budget tương ứng.

### Task 12.3: Coverage-vs-k
- Với A0, A1, A2, A3, A4: k = 1, 2, 4, ..., 256 trên cùng trục; coverage@k và best-of-k; bootstrap CI trên card. Kiểm **H2/H3** (GRPO-only thu hẹp coverage; diversity+KL phục hồi).

### Task 12.4: `analysis/cost_model.py`
- Input: log lượt gọi/token/latency (Task 6.1, 7.2, 8.2), thời gian GPU SFT/GRPO, chi phí suy luận (token sinh/card).
- Output: chi phí per-family của tiến hóa; chi phí một lần của amortized (teacher + SFT + GRPO); **break-even số họ**; độ nhạy theo giá API và giá GPU (bảng nhiều kịch bản).
- Phải tính **cả chi phí teacher data** vào chi phí amortized (không giấu).

### Task 12.5: Đánh giá cuối
- Chạy các test split (interp, extrap, comp, heldout_type) **một lần** cho mọi arm với cấu hình đã đóng băng; sau đó DIMACS **một lần** duy nhất, báo cáo gap tới best-known.
- **Thống kê (`analysis/stats.py`):** bootstrap CI trên card; paired test giữa arm (Wilcoxon hoặc permutation); hiệu chỉnh đa so sánh (Holm); báo cáo mọi họ kể cả thất bại.

### Task 12.6: `analysis/failure_analysis.py`
- Phân tích theo loại cấu trúc, kích thước, mật độ, clustering; liệt kê họ không giải được bằng amortization; ví dụ code lỗi điển hình; phân loại thất bại (copy baseline, bỏ qua card, quá chậm, vi phạm sandbox...).

---

## 13. PHASE 13: Tái lập và viết bài

### Task 13.1: `scripts/make_all_figures.py`
- Đọc **chỉ từ** `runs/**` đã lưu, sinh toàn bộ hình/bảng; không gọi LLM, không tính lại mô hình. Mỗi hình có file nguồn số liệu (CSV) đi kèm.

### Task 13.2: Gói release
- Generators, `splits.json`, evaluator, sandbox, record schema, README tái lập, `requirements.txt`/lock, danh sách seed. Kiểm tra bằng cách chạy một pipeline thu nhỏ từ đầu trên máy sạch (hoặc venv sạch).

### Task 13.3: Bảng tổng (bắt buộc đủ dòng)
A0, A1, A2, A3, A4, A5 (0/5/20/50), A6, **D7-classifier**, DSATUR, RLF, **R(F)**, EoH, ReEvo, TabuCol (tham chiếu non-LLM). Mỗi dòng có: mean colors, gap vs R(F), gap vs DSATUR, valid-rate, số lượt gọi LLM lúc triển khai, latency.

---

## 14. Đặc tả dùng chung

### 14.1 Schema record teacher (`records.jsonl`)

```json
{
  "id": "gnp_d0.30_n100-200_p1_t0",
  "family_id": "gnp_d0.30_n100-200",
  "card": "…(paraphrase idx=1)…",
  "card_detailed": "…",
  "attributes": {"n": [100, 200], "density": 0.30, "deg_var": "low",
                 "clustering": "low", "type": "random"},
  "paraphrase_idx": 1,
  "teacher": "our_engine | eoh | reevo",
  "teacher_run_id": "…",
  "budget": {"llm_calls": 200, "prompt_tokens": 0, "completion_tokens": 0},
  "code": "def priority(adj, colors, sat, deg, uncolored_deg):\n    ...",
  "role": "win | tie | contrast | loss",
  "rank_in_card": 1,
  "score": {
    "base_alg": "RLF",
    "R_F_rescore": 0.0,
    "avg_colors_heuristic_rescore": 0.0,
    "avg_colors_dsatur_rescore": 0.0,
    "avg_colors_rlf_rescore": 0.0,
    "gap_vs_RF_pct": 0.0,
    "valid_rate": 1.0
  },
  "fingerprint": "<hash của vector số màu / thứ tự trên probe 9000-9009>",
  "is_baseline_like": false,
  "split": "train_pool",
  "seeds": {"selection": "3000-3019", "rescore": "6000-6049", "probe": "9000-9009"},
  "git_commit": "…"
}
```
(Các giá trị trên là chỗ giữ chỗ để minh họa; chỉ ghi số đo thật.)

### 14.2 Prompt SFT/GRPO (khung, agent hoàn thiện từ code hiện có)

```
[SYSTEM] You write Python priority functions for greedy graph coloring.
[USER]
Task: write `priority(adj, colors, sat, deg, uncolored_deg)` returning a NumPy
array of scores (one per vertex). At each step the trusted colorer colors the
UNCOLORED vertex with the highest score using the smallest feasible color.
<đoạn mô tả đúng ngữ nghĩa từng tham số, trích từ framework hiện có>
Rules: no imports, only whitelisted NumPy functions, no globals, no state,
pure function.   <danh sách whitelist trích từ sandbox, không bịa>
Graph family description:
{card}
Return only the function code.
[ASSISTANT] {code}
```
Yêu cầu: prompt dùng **chung** cho SFT, GRPO, A0, A6 (chỉ thay/bỏ phần `{card}`); mọi thay đổi phải có version (`prompt_version`) ghi vào log.

### 14.3 Công thức reward (GRPO)

Với heuristic `h` trên tập đồ thị `S` (chung cho cả nhóm 8 mẫu):

```
c_base(g)  = số màu của base_alg(card) trên g            # D-3
c_ref(g)   = min(TabuCol/teacher best đã lưu cho g)      # nếu chưa có thì dùng c_base(g) - 1, cấu hình được
r_sav(h)   = mean_g clip( (c_base(g) - c_h(g)) / max(1, c_base(g) - c_ref(g)), -1, 1 )
r_time(h)  = clip( t_h / t_budget - 1, 0, 1 )            # t_budget = hệ số * thời gian DSATUR vectorized
r_div(h)   = mean pairwise Kendall-tau distance (2 probe graphs) tới các thành viên khác trong nhóm,
             CHỈ tính nếu r_pre(h) > median(r_pre trong nhóm), ngược lại 0
r_tb(h)    = (tùy chọn, mặc định 0)

r_pre(h)   = r_sav + a * r_tb - b * r_time
r(h)       = r_pre(h) + d * r_div(h)
fail       -> r = -1
A_i        = (r_i - mean(r)) / (std(r) + eps)            # advantage trong nhóm
loss       = GRPO loss + beta * KL(pi || pi_ref)
```
- `pi_ref` = policy SFT (A3/A4) hoặc base (A2).
- Hệ số `a, b, d, beta, t_budget` đưa vào config; giá trị mặc định đề xuất ban đầu: `a=0`, `b=0.1`, `d=0.1` (A4; `d=0` ở A2/A3), `beta` do chủ dự án chọn; agent không đoán mà ghi vào config rõ ràng.
- Lưu ý: thưởng khi thắng baseline nhưng **không thưởng thêm cho việc sao chép** baseline; mẫu trùng baseline có `r_sav ≈ 0` nên không bị coi là tốt.

---

## 15. Đồ thị phụ thuộc và thứ tự chạy song song

```
6.0 ─► 6.1 ─► 6.2 (GATE 2) ──┬─► 8.2 ─► 8.3 ─► 10.1 ─► 10.2 ─► 10.3 ─► 11.x ─► 12.x ─► 13.x
                              │
Làm song song được ngay       ├─► 7.1 ─► 7.2 ─────────────► (nguồn teacher thứ 2 cho 8.2, baseline cho 13.3)
(không phụ thuộc pilot):      │
  8.1, 8.4, 9.1, 9.2, 9.3,    └─► (dừng/bàn lại nếu FAIL)
  11.1 (reward + test)
```

**Thứ tự giao việc đề xuất cho agent:**
1. Lượt A (song song): 6.0 → 6.1 (chạy nền, tốn thời gian API) + 8.1 + 8.4 + 9.2 + 11.1.
2. Lượt B: 6.2 (Gate 2) + 9.1 + 9.3 + 7.1.
3. Lượt C (nếu Gate 2 không FAIL): 7.2 + 8.2 → 8.3.
4. Lượt D: 10.1 → 10.2 → 10.3 + 12.1 (A6 sớm).
5. Lượt E: 11.2 → 11.3 → 11.4.
6. Lượt F: 12.2-12.6 → 13.x.

---

## 16. Cổng quyết định (Decision Gates) và tiêu chí đặt trước

| Gate | Thời điểm | Điều kiện đi tiếp | Nếu không đạt |
|---|---|---|---|
| **Gate 2** | sau 6.2 | ≥2/4 họ mục tiêu cải thiện ≥1.0% vs R(F), CI95% không chứa 0, valid 100% | MARGINAL: chỉnh một vòng; FAIL: dừng, bàn pivot |
| **Gate Data** | sau 8.2/8.3 | ≥300 record `win` và <70% record baseline-like | tăng budget/card, thêm họ khó, thêm teacher thứ 2 |
| **Gate SFT** | sau 10.3 | A1 vượt D7 hoặc A0@k ở ≥1 nhóm họ có ý nghĩa thống kê | hỏi chủ dự án, cân nhắc đổi prompt/dữ liệu trước GRPO |
| **Gate GRPO** | trong 11.3 | không rơi vào chốt chặn an toàn (collapse/hacking) | dừng tự động, lưu checkpoint, báo cáo |
| **Pivot** | sau 12.5 | H1 đúng → bài chính; H1 sai → bài negative-result với cùng thí nghiệm | theo pivot rule của proposal |

Mọi ngưỡng ở trên do **chủ dự án** sở hữu; agent không được tự nới sau khi thấy số.

---

## 17. Rủi ro và cách phát hiện sớm

| Rủi ro | Dấu hiệu | Biện pháp |
|---|---|---|
| Model chỉ chép DSATUR/RLF | tỉ lệ fingerprint trùng cao, `r_sav ≈ 0` | lọc `win` (D-9), báo cáo tỉ lệ ≈baseline, A6, so với D7 |
| Teacher không thắng R(F) | Gate 2 FAIL/MARGINAL | đổi họ khó hơn, operator, budget, hoặc pivot |
| D7 đã đủ tốt | A1/A3 ≈ D7 | báo cáo trung thực; chuyển câu chuyện sang chi phí + khả năng tổng quát hóa |
| Reward hacking | reward tăng, savings val giảm; code lạ | sandbox + verifier độc lập, chốt chặn (11.3), kiểm tra tay mẫu top reward |
| Coverage collapse | coverage@k giảm sau GRPO | KL, diversity, báo cáo coverage-vs-k, A4 |
| Rò rỉ dữ liệu | điểm test cao bất thường | test seeds tự động, khóa test, review SEEDS.md |
| Chi phí API/GPU vượt dự kiến | cost log | quy tắc 9, `--dry-run`, chặn vượt budget |
| RL không ổn định | loss nhảy, reward dao động | nhiều seed, lr nhỏ, early stop theo val |
| Paraphrase làm sai con số | test card thất bại | kiểm tra số học tự động (8.4) |
| Môi trường Windows/GPU khác nhau | lỗi đường dẫn, đa tiến trình | `pathlib`, `spawn` an toàn, tách `train/` (D-10) |

---

## 18. Câu hỏi mở cần chủ dự án trả lời (điền vào đây trước khi giao Phase 10-11)

1. **GPU cho SFT/GRPO:** loại, VRAM, local hay cloud? (Quyết định QLoRA hay LoRA 16-bit, vLLM có dùng được không, batch bao nhiêu.)
   - Câu trả lời: `________`
2. **Teacher LLM:** tên model/nhà cung cấp, giới hạn rate, ngân sách API tổng? (Quyết định budget lượt gọi/họ và số card teacher.)
   - Câu trả lời: `________`
3. **Budget tiến hóa chuẩn mỗi họ/run** (đề xuất 200 lượt): `________`
4. **Chấp nhận bỏ CALM** ở giai đoạn đầu? (đề xuất: có) `________`
5. **Ngưỡng Gate 2** (đề xuất mục 6.2): giữ / sửa thành `________`
6. **Deadline IJCAI thực tế** (kiểm tra call chính thức): `________`
7. **Số seed huấn luyện** (đề xuất ≥3): `________`

---

## 19. Mẫu prompt giao việc cho agent

Dán nguyên khối sau, thay `<TASK>` bằng task cụ thể:

```
Bạn là coding agent trong repo "Graph Coloring". Hãy đọc kỹ PLAN_PHASE6_ONWARD.md,
đặc biệt mục 2 (Quyết định đã chốt) và mục 3 (Quy tắc chung), rồi thực hiện
DUY NHẤT task: <TASK, ví dụ "Task 9.2: Harness eval/eval_model.py">.

Yêu cầu:
1. Trước khi viết code: đọc NOTES.md và các module liên quan, tóm tắt ngắn (≤10 dòng)
   cách bạn hiểu API hiện có và kế hoạch triển khai.
2. Triển khai đầy đủ theo mục "Việc cần làm", tạo đúng file "Đầu ra".
3. Viết unit test theo mục "Test"; chạy toàn bộ pytest, 52 test cũ phải xanh.
4. Không sửa các file/thư mục đóng băng. Không đổi seed ranges. Không ghi API key.
5. Nếu gặp mâu thuẫn hoặc thiếu thông tin, DỪNG và liệt kê câu hỏi, không tự bịa
   (đặc biệt chữ ký API, whitelist sandbox, ngưỡng gate).
6. Kết thúc: thêm mục vào NOTES.md (làm gì, số liệu, quyết định, tồn đọng),
   liệt kê file đã tạo/sửa, nêu lệnh chạy lại, báo kết quả pytest.
```

---
*Hết. Mọi con số trong file này (budget 200 lượt, ngưỡng 1.0%, các hệ số `a, b, d`, khoảng seed đề xuất) là giá trị khởi điểm để chủ dự án duyệt; không phải kết quả đo.*
