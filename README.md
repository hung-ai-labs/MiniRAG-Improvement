<div align="center">

# Chọn bằng chứng có ngân sách cho MiniRAG

### Học cách đưa đúng bằng chứng vào ngữ cảnh của mô hình ngôn ngữ nhỏ

*Mã nguồn, dữ liệu và bằng chứng tái lập cho Bảng 1*

[Tóm tắt](#tóm-tắt) · [Phương pháp](#2-phương-pháp) · [Kết quả](#4-kết-quả) · [Kiểm chứng](#5-kiểm-chứng-tính-tái-lập) · [Cấu trúc](#6-cấu-trúc-thư-mục) · [Chạy lại](#7-cài-đặt-và-chạy-lại) · [Hạn chế](#9-hạn-chế-và-các-mối-đe-doạ-tới-tính-hợp-lệ)

</div>

---

## Tóm tắt

MiniRAG [1] là hệ RAG dựa trên đồ thị, thiết kế cho **mô hình ngôn ngữ nhỏ** (SLM). Sau khi truy hồi trên đồ thị, MiniRAG xếp hạng các đoạn văn bản (chunk) theo **tổng điểm của mọi thực thể hạt giống trỏ tới chunk**. Sau đó nó đưa chunk vào khối *Sources* theo thứ tự này, cho tới khi gặp chunk đầu tiên vượt ngân sách token thì dừng.

Phân tích trên tập dev cho thấy lỗi chính nằm ở **thứ tự và cách cắt**, không nằm ở khâu tìm kiếm. Trong 45 chunk đầu của danh sách xếp hạng, có 117/177 câu hỏi chứa đủ bằng chứng. Nhưng chỉ 88/177 câu giữ được đủ bằng chứng sau khi cắt ở 4.000 token.

Chúng tôi đề xuất một **bộ chọn bằng chứng** thay cho bước xếp hạng và cắt đó. Bộ chọn gồm ba thành phần:
1. **Bộ chấm F:** hồi quy logistic trên 18 đặc trưng lấy từ chính pipeline của MiniRAG (đồ thị, thời gian, cấu trúc hội thoại).
2. **Tín hiệu ngữ nghĩa:** logit của một cross-encoder zero-shot, đưa vào cùng bộ chấm.
3. **Chọn tập theo ngân sách:** tham lam theo lợi ích biên trên mỗi token, trên một hàm phủ submodular có lợi ích giảm dần theo hội thoại.

Đồ thị, index, prompt, SLM và ngân sách token **giữ nguyên**.

**Kết quả.** Đo trên 435 câu hỏi LiHua-World ngoài tập dev, 3 lượt sinh × 3 lượt chấm, so với baseline:

| SLM | ACC baseline | ACC đầy đủ | Thay đổi | Ý nghĩa thống kê |
|---|---:|---:|---:|---|
| Qwen2.5-3B | 52,92 | 58,93 | **+6,0** | p ≤ 0,02 ở cả 3 lượt sinh |
| Phi-3.5-mini | 46,03 | 55,43 | **+9,4** | p ≤ 3·10⁻⁴ ở cả 3 lượt sinh |

Tỷ lệ trả lời sai (ERR) không đổi có ý nghĩa.

**Phát hiện phụ.** vLLM 0.11.0 phục vụ Phi-3.5 sai khi có nhiều request đồng thời, do cách nó chọn hệ số LongRoPE. Lỗi này làm đồ thị do Phi dựng bị thưa. Chúng tôi chẩn đoán, vá và kiểm chứng bằng thí nghiệm A/B: tỷ lệ chunk trích xuất thất bại giảm từ 96/96 xuống 1/96.

## Điểm nổi bật

- **Đóng góp có kiểm soát.** Ba thành phần chỉ đổi khối Sources. Mọi biến khác cố định giữa các dòng của Bảng 1.
- **Không rò rỉ dữ liệu test.** Bộ chấm học trên 200 câu dev, cross-fit 5 fold theo hội thoại. Siêu tham số β, τ chọn lồng trong fold. Sau đó mọi thứ đóng băng cho 435 câu test.
- **Kiểm định thống kê theo cặp.** McNemar chính xác trên từng câu, lặp ở 3 lượt sinh độc lập, chứ không chỉ so hiệu số điểm.
- **Tái lập tới từng byte.** `python -m reproduce.verify_results` chạy offline trong khoảng 30 giây. Nó kiểm toàn bộ chuỗi: index → context → request → phản hồi gốc của server → hoá đơn API của giám khảo → bảng kết quả.

---

## 1. Bài toán và động cơ

Bảng dưới đếm số câu có **đủ bằng chứng** trong khối Sources, đo offline trên 177 câu dev có nhãn Evidence, với SLM Qwen2.5-3B:

| Khối Sources | Đủ bằng chứng / 177 |
|---|---:|
| Baseline: MiniRAG + source-time, cắt A1@4000 | 88 |
| Trần: bằng chứng **có trong 45 chunk đầu**, nếu xếp lên trước | 117 |

Như vậy **29 câu** đã có bằng chứng trong danh sách truy hồi nhưng mất ở bước cắt. Có hai nguyên nhân:
1. Điểm `kwd2chunk` cộng dồn theo **số hạt giống** trỏ tới chunk. Mỗi câu hỏi có khoảng 100 hạt giống, đa số chỉ liên quan yếu, nên chunk "hub" nhắc nhiều thực thể được đẩy lên đầu.
2. Bước cắt A1 **dừng ở chunk đầu tiên không vừa ngân sách**. Kích thước chunk rất lệch (trung vị khoảng 274 token, 10% lớn nhất từ 1.200 token trở lên). Một chunk dài có thể chặn mọi chunk ngắn phía sau.

## 2. Phương pháp

```
Câu hỏi ─► [MiniRAG] phân tích câu hỏi → thực thể hạt giống → đồ thị → kwd2chunk        (giữ nguyên)
          ─► [+ source-time] đẩy chunk có mốc Time: khớp câu hỏi                        (baseline)
          ─► 45 chunk đầu (pool)
               ├─► ① Bộ chấm F: 18 đặc trưng → p(chunk là bằng chứng)
               ├─► ② + logit cross-encoder (đặc trưng thứ 19, cùng bộ chấm)
               └─► ③ Chọn tập theo ngân sách 4.000 token                             (thay cho cắt A1)
          ─► prompt rag_response (Entities + Sources) ─► SLM ─► câu trả lời             (giữ nguyên)
```

### 2.1 Bộ chấm F, đặc trưng từ pipeline gốc

Hồi quy logistic: `StandardScaler` rồi `LogisticRegression(C=1)`, không dò siêu tham số. Mô hình chạy trên 18 đặc trưng, chia 4 nhóm:

| Nhóm | Đặc trưng |
|---|---|
| Đồ thị (10) | `native_rank`, `native_rank_log`, `in_native30`, `graph_rank`, `graph_score`, `graph_score_rel`, `in_graph`, `n_seed_paths`, `n_keyword_groups`, `is_path_head` |
| Thời gian (2) | `date_match`, `time_boosted` |
| Dense (2) | `in_dense30`, `dense_rank` |
| Cấu trúc hội thoại (4) | `is_continuation`, `chunk_order`, `doc_chunks`, `tokens` |

- **Nhãn:** chunk chứa mốc `Time:` của cột Evidence. Nhãn chỉ dùng khi huấn luyện; lúc suy luận không đọc nhãn.
- **Mã nguồn:** [`features.py`](reproduce/evidence_set/features.py), [`quick_scorer.py`](reproduce/evidence_set/quick_scorer.py).

### 2.2 Tín hiệu ngữ nghĩa

Cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2`: 22 triệu tham số, **không fine-tune**, chạy cục bộ. Logit của nó được thêm làm đặc trưng thứ 19 trong cùng hồi quy logistic.

Đặc trưng đồ thị cho biết chunk *liên quan tới thực thể nào* của câu hỏi. Cross-encoder cho biết chunk *có nói đúng điều được hỏi không*. Hai nguồn này bổ sung cho nhau, xem mục 4.3.

### 2.3 Chọn tập theo ngân sách

Cho pool *P*, xác suất *p_c* từ bộ chấm và ngân sách *B* = 4.000 token, ta tìm tập *S* ⊆ *P* với Σ_{c∈S} tok(c) ≤ *B* để cực đại:

```
F(S) = Σ_d [1 − Π_{c∈S, doc(c)=d} (1 − p_c)]                    phủ theo hội thoại (lợi ích giảm dần)
     + β · Σ_f w_f [1 − Π_{c∈S, c hỗ trợ f} (1 − p_c)]           phủ khía cạnh của câu hỏi
```

- **Khía cạnh f:** một từ khoá của câu hỏi (qua các path của thực thể hạt giống), hoặc một mốc ngày trong câu hỏi.
- **Trọng số:** w_f = log(1 + |P| / |supp(f)|), để khía cạnh hiếm được tính nhiều hơn.
- **Tính chất:** cả hai số hạng đều là hàm phủ xác suất, nên *F* đơn điệu và submodular.
- **Thuật toán:** tham lam theo (F(S∪{c}) − F(S)) / tok(c). Chunk không vừa ngân sách thì **bỏ qua và xét tiếp**, khác A1 vốn dừng hẳn.
- **Siêu tham số (β, τ):** chọn lồng trong fold trên dev. Kết quả: Qwen (2, 0), Phi (0, 0).
- **Mã nguồn:** [`select.py`](reproduce/evidence_set/select.py).

---

## 3. Thiết lập thực nghiệm

| Hạng mục | Thiết lập |
|---|---|
| Dữ liệu | LiHua-World [1]: 442 hội thoại, 637 câu hỏi (635 câu khác nhau), gồm Single / Multi / Null |
| Phân tách | **dev 200** (phân tầng, seed 13, [`logs/devset.csv`](logs/devset.csv)) chỉ để huấn luyện scorer. **Test 435** là phần còn lại, chỉ dùng một lần ở cuối. 385/435 câu test có nhãn Evidence. |
| SLM | `Qwen/Qwen2.5-3B-Instruct` và `microsoft/Phi-3.5-mini-instruct`, phục vụ bằng vLLM 0.11.0 trên Modal |
| Index | Mỗi SLM **tự dựng index của mình**, như bài báo. Embedding `all-MiniLM-L6-v2` (384 chiều), giống bài báo. |
| Sinh câu trả lời | Prompt `rag_response` của MiniRAG, `max_tokens` 1024. 3 lượt; lượt *r* dùng seed `13 + id + 100000·(r−1)`. |
| Giám khảo | `deepseek-flash`, prompt `JUDGE_PROMPT` của [`Step_2_evaluate.py`](reproduce/Step_2_evaluate.py). Chấm mù, xáo thứ tự giữa các cấu hình, 3 lượt độc lập cho mỗi câu trả lời. |
| Chỉ số | ACC, ERR, Neither (%). Trung bình trên 3 lượt sinh × 3 lượt chấm; ± là độ lệch chuẩn giữa các lượt sinh. |
| Kiểm định | McNemar chính xác hai phía, trên phán quyết đa số của mỗi lượt sinh, ghi dạng `b lên / c xuống`. Một thành phần chỉ được coi là có tác dụng khi **cùng chiều và có ý nghĩa ở cả 3 lượt**. |

**Biến được kiểm soát.** Mọi dòng của cùng một SLM dùng chung:
- index;
- đầu ra phân tích từ khoá, lấy từ cache;
- bảng Entities, prompt;
- SLM, ngân sách 4.000 token;
- giám khảo và seed.

Giữa các dòng chỉ khác **danh sách chunk trong khối Sources**.

---

## 4. Kết quả

### 4.1 Bảng 1: ablation cộng dồn trên tập test (435 câu)

**Qwen2.5-3B-Instruct**

| # | Cấu hình | Đủ bằng chứng (/385) | ACC ↑ | ERR ↓ | Neither | ACC so với baseline (lượt 1 / 2 / 3) |
|---|---|---:|---:|---:|---:|---|
| 1 | C0: MiniRAG | 167 | 47,20 ± 0,96 | 27,05 | 25,75 | 3/27 · 1/26 · 0/26 (p ≤ 8,4·10⁻⁶) |
| 2 | + source-time (*baseline*) | 196 | 52,92 ± 0,95 | 25,72 | 21,35 | — |
| 3 | + ① bộ chấm F | 229 | 56,02 ± 0,78 | 25,13 | 18,85 | 52/40 · 52/36 · 57/42 (p = 0,25 · 0,11 · 0,16) |
| 4 | + ② tín hiệu ngữ nghĩa | 238 | **59,05 ± 1,00** | 26,39 | 14,56 | 58/36 · 67/30 · 61/36 (p ≤ 0,03) |
| 5 | + ③ chọn tập (*đầy đủ*) | **241** | 58,93 ± 0,20 | 26,16 | 14,92 | 61/37 · 66/34 · 65/35 (p ≤ 0,02) |

**Phi-3.5-mini-instruct**

| # | Cấu hình | Đủ bằng chứng (/385) | ACC ↑ | ERR ↓ | Neither | ACC so với baseline (lượt 1 / 2 / 3) |
|---|---|---:|---:|---:|---:|---|
| 1 | C0: MiniRAG | 128 | 42,30 ± 0,80 | 29,22 | 28,48 | 0/18 · 1/18 · 1/15 (p ≤ 5,2·10⁻⁴) |
| 2 | + source-time (*baseline*) | 147 | 46,03 ± 0,62 | 28,02 | 25,95 | — |
| 3 | + ① bộ chấm F | 202 | 53,72 ± 0,61 | 26,39 | 19,90 | 70/37 · 64/31 · 77/42 (p ≤ 0,002) |
| 4 | + ② tín hiệu ngữ nghĩa | 211 | 55,33 ± 1,59 | 26,23 | 18,44 | 69/35 · 68/28 · 82/33 (p ≤ 0,001) |
| 5 | + ③ chọn tập (*đầy đủ*) | **212** | **55,43 ± 1,35** | 26,41 | 18,16 | 72/27 · 72/34 · 78/35 (p ≤ 3·10⁻⁴) |

Độ nhiễu của giám khảo, tính bằng độ lệch chuẩn giữa 3 lượt chấm ACC của baseline, là 0,32 (Qwen) và 0,22 (Phi).

### 4.2 Đóng góp riêng của từng thành phần (so với dòng ngay trên)

| So sánh | Qwen: ACC (lượt 1 / 2 / 3) | Phi: ACC (lượt 1 / 2 / 3) | Kết luận |
|---|---|---|---|
| ① F so với baseline | 52/40 · 52/36 · 57/42 | 70/37 · 64/31 · 77/42 | Có ý nghĩa với Phi; với Qwen cùng chiều nhưng chưa có ý nghĩa |
| ② CE so với ① | 41/31 · 43/22 · 37/27 | 29/28 · 34/27 · 40/26 | Chỉ có ý nghĩa ở 1/6 lượt |
| ③ chọn tập so với ② | 32/30 · 27/32 · 32/27 | 29/18 · 27/29 · 20/26 | **Ngang nhau** |

ERR so với dòng ngay trên không có ý nghĩa ở mọi phép so. Hiệu quả tổng (dòng 5 so với dòng 2) là chắc chắn. Phần tách riêng cho từng thành phần ②, ③ **chưa** đủ bằng chứng QA, nên chỉ được báo cáo như xu hướng.

### 4.3 Phân tích offline trên dev, cross-fit (Qwen, 0 API)

Bảng đếm số câu có đủ bằng chứng trên 177 câu, so với baseline 88.

| Cấu hình | Qwen | Phi | Ghi chú |
|---|---:|---:|---|
| Baseline (A1@4000) | 88 | 71 | |
| ① F sắp theo p → A1 | 103 | 97 | AUC trung bình 5 fold: Qwen 0,934 |
| CE một mình → A1 | 103 | 95 | Qwen: 25 lên / **10 xuống** |
| ①+② sắp theo p → A1 | **110** | 97 | Qwen: 25 lên / 3 xuống, p = 2,7·10⁻⁵; AUC 0,961 |
| ①+②+③ chọn tập | 109 | 96 | Số chunk trung vị 6 → 11, ngân sách dùng 3.689 → 3.955 token |
| *Đối chứng: xếp lại bằng BM25 → A1* | *108* | ***102*** | *không thuộc phương pháp* |
| *Trần: oracle trong pool 45* | *117* | *109* | |

**Bỏ từng nhóm đặc trưng** khỏi ①+② (Qwen dev, sắp theo p, gốc 110):

| Bỏ nhóm | cấu trúc hội thoại | dense | đồ thị | thời gian |
|---|---:|---:|---:|---:|
| Còn lại | 105 (−5) | 109 (−1) | 109 (−1) | 110 (0) |

Nhóm thời gian không đóng góp thêm, vì source-time đã khai thác thông tin này ở bước trước.

### 4.4 Theo loại câu hỏi (ACC, test)

| Cấu hình | Qwen Single (347) | Multi (43) | Null (45) | Phi Single | Multi | Null |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 53,70 | 29,72 | 69,14 | 44,28 | 34,37 | 70,62 |
| Đầy đủ | 61,51 | 28,17 | 68,40 | 56,10 | 37,21 | 67,65 |

- Mức tăng đến **gần như hoàn toàn từ câu Single**.
- Multi có n = 43 nên biến động mạnh giữa các dòng.
- Null giảm nhẹ ở Phi: khi đưa nhiều bằng chứng "gần giống" vào hơn, mô hình bớt từ chối.

### 4.5 Phát hiện phụ: lỗi LongRoPE khi phục vụ Phi-3.5 bằng vLLM

**Triệu chứng.** Index Phi ban đầu có 167/516 chunk không trích được thực thể nào. Riêng chunk từ 800 token trở lên thì 85/116 bị như vậy. Index của Qwen chỉ có 9 chunk như vậy.

**Nguyên nhân.** `phi3_long_rope_scaled_rope.py` của vLLM 0.11.0 chọn bộ hệ số *short* hay *long* bằng `torch.any(positions > 4096)` trên **cả batch**. Hệ quả:
- KV cache của một chuỗi bị trộn hai kiểu mã hoá vị trí, tuỳ vào các chuỗi đang chung batch.
- Chunk dài bị nặng nhất, vì prompt trích xuất của chúng (khoảng 4.170 token) vượt đúng mốc 4.096.

**Cách vá** ([`modal_slm_phi.py`](reproduce/modal_slm_phi.py), `ROPE_FIX`): đặt `short_factor = long_factor` qua `--hf-overrides`, giống cách llama.cpp làm. Trọng số, prompt và pipeline không đổi.

**Kiểm chứng A/B** ([`review/phi_rope_ab_20261112/`](review/phi_rope_ab_20261112/summary.json)): 96 chunk, 32 request đồng thời.

| Server | Chunk rỗng, chunk ≥ 800 token | Chunk rỗng, chunk < 300 token | Thực thể | Quan hệ |
|---|---:|---:|---:|---:|
| Gốc | 48/48 | 48/48 | 0 | 0 |
| Đã vá | **0/48** | **1/48** | 1.257 | 419 |

Index Phi dựng lại trên server đã vá có 2.856 node, 2.138 cạnh và 4/516 chunk rỗng. Index Qwen có 1.556 node, 1.509 cạnh. Mọi kết quả Phi trong README này đều đo trên server đã vá. Kết quả Phi cũ đo trên server lỗi đã bị loại.

---

## 5. Kiểm chứng tính tái lập

```bash
python -m reproduce.verify_results                   # khoảng 30 giây, offline, không cần API key
python -m reproduce.verify_results --recapture 435   # thêm bước [6] trên toàn bộ 435 câu (khoảng 1,5 giờ mỗi SLM, cần torch)
```

Script không gọi API nào và không ghi đè file kết quả nào; mọi phép tính lại chạy trong thư mục tạm.

**Chuỗi bằng chứng được kiểm**, cho từng SLM:

```
index ──[6]──► pool truy hồi ──[2]──► context 5 cấu hình ──[3]──► request vLLM ──[3]──► phản hồi gốc
                    ▲                                                                       │
             scorer ──[5]── chỉ học trên dev200                                             ▼
                                         bảng ◄──[1]── phán quyết ◄──[4]── hoá đơn DeepSeek
```

| # | Kiểm tra | Tiêu chí đạt | Ý nghĩa |
|---|---|---|---|
| 1 | TABLE | `results.json` tính lại == bản đã lưu | Bảng được tính từ phán quyết, không gõ tay |
| 2 | CONTEXT | sha256(`contexts.jsonl`) dựng lại == `build.json` | Context của 5 cấu hình là đầu ra của code này |
| 3 | ANSWER | Với 6.525 câu trả lời: sha256 của request (prompt + context + câu hỏi + seed) == `body_sha256`. `raw.json` là phản hồi gốc của vLLM (`chatcmpl-…`, `model`, `created`, `usage`) và chứa đúng câu trả lời. | Mỗi câu trả lời sinh ra từ đúng context của cấu hình đó, vào thời điểm ghi trong phản hồi |
| 4 | JUDGE | Với 15.900 phán quyết: sha256 của request chấm có hoá đơn `ok` trong `deepseek/calls.jsonl` | Mỗi phán quyết là một lần gọi API có trả phí |
| 5 | SCORER | Huấn luyện lại trên dev200 → hệ số trùng model đã dùng (Δ = 0) | Không học trên tập test |
| 6 | RECAPTURE | Chạy lại truy hồi MiniRAG trên index; LLM bị chặn; context trùng từng byte | Pool và context sinh ra từ đúng index trong gói |

**Kết quả trên máy tác giả** (đầy đủ trong [`VERIFY_LOG.txt`](VERIFY_LOG.txt)):

```
=== Qwen2.5-3B-Instruct
  [OK ] [1] TABLE tính lại == results.json: C0 47.20, BASELINE 52.92, F 56.02, CE0F 59.05, CE0F_SET 58.93
  [OK ] [2] CONTEXT dựng lại, sha256 == build.json: afaa67a8dc98e14f
  [OK ] [3] ANSWER ... 6525 câu trả lời (3 lượt sinh), created 2026-11-11 15:02 UTC -> 2026-11-11 16:38 UTC
  [OK ] [4] JUDGE mỗi phán quyết có hoá đơn DeepSeek: 15900 phán quyết
  [OK ] [5] SCORER huấn luyện lại trên dev200 == model đã dùng: max|Δhệ số| = 0
  [OK ] [6] RECAPTURE baseline: 435/435 câu trùng từng byte
=== Phi-3.5-mini-instruct (LongRoPE fix)
  [OK ] [1]–[5] ... created 2026-11-12 13:39 UTC -> 2026-11-12 15:48 UTC
KẾT LUẬN: TẤT CẢ KIỂM TRA ĐỀU ĐẠT
```

**Toàn vẹn gói.** `shasum -a 256 -c MANIFEST_SHA256.txt` kiểm mọi file so với lúc đóng gói.

**Giới hạn của kiểm chứng offline.** Các bước [1]–[6] chứng minh dữ liệu **nhất quán nội tại từ đầu đến cuối**, nhưng không loại trừ được việc `raw.json` bị viết tay. Có hai cách kiểm độc lập:
1. Đối chiếu trang usage/billing của tài khoản DeepSeek và Modal ngày 11–12/11/2026. Ở đó phải thấy 55.459 request DeepSeek thành công, 21,5 triệu token, tổng 4,46 USD.
2. Sinh lại một mẫu câu trả lời theo mục 7.

---

## 6. Cấu trúc thư mục

```
MiniRAG-HTTM/
├── minirag/                         thư viện MiniRAG (đã sửa, mục 8)
├── reproduce/
│   ├── Step_0_index.py              index (như bản gốc)
│   ├── Step_1_QA.py                 hỏi đáp (như bản gốc)
│   ├── Step_2_evaluate.py           JUDGE_PROMPT dùng cho Bảng 1
│   ├── evidence_set/                ★ phương pháp đề xuất và pipeline Bảng 1
│   │   ├── pool.py                  truy hồi gốc, ghi toàn bộ xếp hạng sau hai bước cắt
│   │   ├── features.py              18 đặc trưng (mục 2.1)
│   │   ├── quick_scorer.py          bộ chấm F / CE0+F, cross-fit 5 fold
│   │   ├── select.py                chọn tập theo ngân sách (mục 2.3)
│   │   ├── evaluate.py              đánh giá offline (mục 4.3)
│   │   ├── pipeline.py              capture → score → build → generate → judge → report
│   │   ├── run_ladder.sh            một lệnh cho cả Bảng 1 của một SLM
│   │   └── deepseek_judge.py        giám khảo, ghi hoá đơn từng lần gọi
│   ├── verify_results.py            ★ kiểm chứng (mục 5)
│   ├── modal_slm.py                 server vLLM cho Qwen2.5-3B
│   ├── modal_slm_phi.py             server vLLM cho Phi-3.5, có ROPE_FIX
│   ├── phi_longrope_fix.py          bản vá LongRoPE khi tự chạy vLLM (mục 10)
│   ├── phi_rope_ab.py               thí nghiệm A/B (mục 4.5)
│   └── index_health.py              kiểm index có bị thưa (mục 10)
├── dataset/LiHua-World/             dữ liệu và câu hỏi (như bản gốc)
├── LiHua-World-qwen-modal/          index do Qwen2.5-3B dựng
├── LiHua-World-phi-ropefix/         index do Phi-3.5 dựng, trên server đã vá
├── logs/devset.csv                  200 câu dev
├── logs/phi_ropefix/                log chạy Bảng 1 của Phi
├── review/
│   ├── evidence_set_20261111/
│   │   ├── qa_nondev435/ladder/              Qwen: contexts, answers, judgments, results.json, REPORT.md
│   │   ├── qa_nondev435/qwen_cache/          Qwen: request và phản hồi gốc của vLLM
│   │   ├── phi_ropefix/qa_nondev435/…        Phi: như trên
│   │   ├── phi_ropefix/qa_dev200/            Phi: lượt chạy thử trên dev (giám khảo Gemini)
│   │   ├── deepseek/calls.jsonl              hoá đơn mọi lần gọi DeepSeek
│   │   ├── pools_*, contexts_*, kw_cache_*   pool truy hồi gốc và cache parser (Qwen; Phi nằm trong phi_ropefix/)
│   │   ├── codex/, quick/                    dữ liệu huấn luyện dev, scorer, đánh giá offline
│   │   └── offline_eval_dev200*.json
│   └── phi_rope_ab_20261112/                 kết quả A/B
├── baseline.yaml                    cấu hình baseline đã đóng băng
├── VERIFY_LOG.txt                   đầu ra của verify_results
├── MANIFEST_SHA256.txt              sha256 của mọi file
└── requirements.lock.txt            môi trường đã dùng (Python 3.13)
```

Tên `review/evidence_set_20261111/` được giữ nguyên như lúc chạy, để code trong gói **giống hệt** code đã sinh ra kết quả.

---

## 7. Cài đặt và chạy lại

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
```

**Offline, miễn phí:**

```bash
python -m reproduce.verify_results                                               # mục 5
MINIRAG_EVSET_SLM=qwen        python -m reproduce.evidence_set.pipeline report --split nondev
MINIRAG_EVSET_SLM=phi_ropefix python -m reproduce.evidence_set.pipeline report --split nondev
MINIRAG_EVSET_SLM=qwen        python -m reproduce.evidence_set.evaluate --help   # đánh giá offline trên dev
```

**Từ đầu** (trả phí, cần GPU Modal và API key DeepSeek):
1. `modal deploy reproduce/modal_slm.py`, hoặc `modal_slm_phi.py` cho Phi.
2. Ghi URL vào `~/.config/minirag/slm_url` (Phi: `phi_ropefix_url`) và token vào `~/.config/minirag/slm_key`.
3. Ghi `DEEPSEEK_API_KEY=…` vào `.env`.
4. Chạy:
   ```bash
   zsh reproduce/evidence_set/run_ladder.sh qwen          # hoặc phi_ropefix
   ```

Mọi bước đều resume được, và câu trả lời đã có trong `qwen_cache/` sẽ không gửi lại. Muốn sinh lại thật thì chuyển `qa_nondev435/` sang chỗ khác trước khi chạy.

Chi phí đã tốn: DeepSeek 4,46 USD, gồm cả các lượt thử và lượt Phi cũ bị loại; cộng thêm giờ GPU trên Modal.

---

## 8. Thay đổi trong `minirag/` so với MiniRAG gốc

| Thay đổi | File | Loại |
|---|---|---|
| Vá lỗi O(N²) trong `ainsert`: bản gốc trích xuất lại toàn bộ corpus mỗi lần chèn một tài liệu. Từ khoảng 68.000 xuống còn 653 lời gọi. | `minirag.py` | sửa bug |
| Sửa lỗi hoa/thường khi so khớp answer-type. Trước khi sửa, khớp 0 node. | `operate.py` | sửa bug |
| Phi trả JSON dạng list làm MiniRAG crash. Trường hợp này được đưa về nhánh xử lý JSON lỗi sẵn có. | `operate.py` | sửa bug |
| Cắt Sources theo ngân sách 4.000 token (A1@4000). Có ở mọi dòng của Bảng 1. | `operate.py` | cấu hình |
| **Source-time**: đẩy chunk có mốc `Time:` khớp câu hỏi (dòng 2). | `temporal.py` | cải tiến baseline |
| Cache đầu ra parser từ khoá theo sha256 prompt, để truy hồi tất định. | `operate.py` | hạ tầng |
| Backend OpenAI-compatible (Gemini / vLLM); profile `c0` / `baseline`. | `llm/gemini.py`, `production.py` | hạ tầng |

Các module khác (`leef.py`, `path_*.py`, `p0_*`, `p2_*`, `graph_flow.py`, …) là biến thể từ các thí nghiệm trước:
- Chúng **tắt theo mặc định**.
- Profile `c0` và `baseline` đặt rỗng mọi biến điều khiển của chúng.
- `pool.py` assert rằng RRF, path pruning và extractive context đều tắt.

Vì vậy, **"C0: MiniRAG"** trong Bảng 1 là MiniRAG gốc cộng hai bản sửa bug và A1@4000.

---

## 9. Hạn chế và các mối đe doạ tới tính hợp lệ

1. **Giám khảo khác bài báo.** Chúng tôi dùng DeepSeek-flash, chấm từng câu trả lời với nhãn accurate / error / neither. Bài báo [1] dùng GPT-4o, chấm ba hệ thống trong cùng một prompt.
   - Số tuyệt đối **không so trực tiếp** với bài báo, nơi Qwen2.5-3B đạt 48,75 và Phi-3.5-mini đạt 53,29.
   - Phép so giữa các dòng vẫn có kiểm soát, vì mọi dòng dùng cùng một giám khảo.
2. **Đóng góp riêng của ② và ③ chưa được xác nhận ở mức QA.** Xem mục 4.2. Chỉ hiệu quả tổng (dòng 5 so với dòng 2) là có ý nghĩa ở mọi lượt, trên cả hai SLM.
3. **Đối chứng BM25.** Offline, xếp lại bằng BM25 là đối chứng mạnh: 108 so với 110 của chúng tôi trên Qwen, và **102 so với 97 trên Phi**. Chúng tôi chưa chạy BM25 ở mức QA trong Bảng 1, nên không thể khẳng định bộ chọn tốt hơn một bộ xếp lại từ vựng đơn giản.
4. **Theo loại câu.** Mức tăng tập trung ở Single. Multi có n = 43, không ổn định. Null giảm nhẹ ở Phi.
5. **Một tập dữ liệu.** Mới đo trên LiHua-World, và nhãn Evidence là điều kiện để huấn luyện bộ chấm. Khả năng khái quát sang dataset khác chưa được kiểm.
6. **Index Qwen thiếu một tài liệu** (441/442), do một lần index bị ngắt. Index Phi đủ 442/442. Mọi dòng của cùng một SLM dùng chung index, nên phép so giữa các dòng không bị ảnh hưởng.
7. **Không có trong gói:**
   - các dòng bộ chấm fine-tune, vì model nặng 1 GB;
   - index do Gemini dựng;
   - kết quả Phi cũ trên server lỗi;
   - log của các thí nghiệm trước đó.

---

## 10. Dùng cho dataset khác

1. **Phục vụ Phi bằng server đã vá.**
   - Trên Modal: `modal_slm_phi.py`, có `ROPE_FIX = True` theo mặc định.
   - Tự chạy vLLM: `python -m reproduce.phi_longrope_fix` in ra lệnh `vllm serve … --hf-overrides …`. Với model không dùng LongRoPE, như Qwen hay Llama, nó báo không cần vá.
   - ollama/llama.cpp không bị lỗi này.
2. **Index:** `python reproduce/Step_0_index.py --workingdir ./<index> …`, với `GEMINI_API_BASE` trỏ tới server. Xem `run_ladder.sh`.
3. **Kiểm index ngay sau khi dựng:**
   ```bash
   python -m reproduce.index_health ./<index>        # exit 1 nếu quá 5% chunk không có thực thể
   ```
   Mốc tham chiếu trên LiHua-World:

   | Index | Chunk rỗng |
   |---|---:|
   | Phi, server lỗi | 167/516 |
   | Phi, đã vá | 4/516 |
   | Qwen | 9/514 |

4. **Huấn luyện lại bộ chấm:** cần một tập dev có nhãn Evidence. Chạy `pipeline.py capture --split dev200`, rồi bước `score`.

Phi đôi khi còn viết sai định dạng record khi trích xuất, ví dụ đặt `##` bên trong ngoặc. Parser của MiniRAG khi đó bỏ record ấy. Lỗi này **chưa** sửa, để giữ nguyên pipeline đã đo; ảnh hưởng của nó nhỏ hơn nhiều so với lỗi LongRoPE.

---

## Tài liệu tham khảo

[1] T. Fan, J. Wang, X. Ren, C. Huang. *MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation.* arXiv:2501.06713, 2025. Mã nguồn: https://github.com/HKUDS/MiniRAG

```bibtex
@article{fan2025minirag,
  title   = {MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation},
  author  = {Fan, Tianyu and Wang, Jingyuan and Ren, Xubin and Huang, Chao},
  journal = {arXiv preprint arXiv:2501.06713},
  year    = {2025}
}
```

## Lời cảm ơn và giấy phép

Mã nguồn kế thừa từ [MiniRAG](https://github.com/HKUDS/MiniRAG) (HKUDS) và dataset LiHua-World của nhóm tác giả. Phát hành theo giấy phép MIT, như bản gốc (xem [`LICENSE`](LICENSE)).
