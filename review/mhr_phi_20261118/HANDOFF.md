# Bàn giao: chạy nốt ladder MultiHop-RAG × Phi-3.5-mini

Cập nhật lần cuối: 2026-11-19 15:45. C0 dừng ở 29,30 USD trên workspace cũ (đúng ngưỡng đặt trước). Người nhận dùng **workspace Modal của mình** cho phần sinh còn lại.
Đọc theo thứ tự: file này → [PROTOCOL.md](PROTOCOL.md) (đăng ký trước, không sửa) → [DEVIATIONS.md](DEVIATIONS.md)
→ [../../reproduce/mhr_phi/RUNBOOK.md](../../reproduce/mhr_phi/RUNBOOK.md) (lệnh chi tiết từng bậc).

## 1. Thí nghiệm là gì

Ladder ablation trên MultiHop-RAG (2.156 câu test × 3 seed sinh 101/202/303, dev 400 câu seed 101):
**C0 → +ST → +F → +CE → +set**. Bản Qwen2.5-3B đã xong (bảng: `review/mhr_set_20261117/BANG_SO_SANH.txt`;
chỉ +F được xác nhận). Đợt này lặp lại cho **Phi-3.5-mini-instruct**, làm như bản `phi_ropefix` của Hùng:
Phi tự dựng index, tự phân tích từ khoá, tự sinh đáp án; F/CE/set fit lại trên pool dev của Phi theo đúng quy tắc
đã đăng ký cho Qwen. Giám khảo: **gemini-3.5-flash-lite**, 3 lượt/đáp án (không đổi).

Quy tắc quyết định mỗi bậc: bậc mới hơn bậc trước **ở cả 3 seed** và McNemar chính xác **p < 0.05 ở mỗi seed**.
So sánh Phi với Qwen chỉ mô tả.

## 2. Trạng thái lúc bàn giao

| Bước | Trạng thái |
|---|---|
| Server Phi (`reproduce/modal_slm_phi.py`, ROPE_FIX, A100-80GB) | đã kiểm (vLLM 0.11.0, short_factor = long_factor trong log) |
| Index Phi `MultiHop-RAG-phi-ropefix609/` | ✅ qua cổng I: 3/1656 chunk rỗng (0,2%) — [GATE_I.json](GATE_I.json) |
| C0 dev s101 | ✅ |
| C0 test s101 | ✅ |
| C0 test s202 | từ khoá ✅, ngữ cảnh ✅ (mhr-0315 `early_exit`), đáp án ✅ đủ 2.156 trên đĩa — lần `c0 execute` tới chỉ hoàn tất biên nhận, không sinh thêm |
| C0 test s303 | chưa |
| +ST, +F, +CE, +set (Phi) | chưa |
| Chấm (Gemini) | chưa chấm gì cho Phi |

Lệnh `c0 execute` **tự nối tiếp**: bỏ qua cohort đã xong, sinh tiếp phần còn thiếu trong `answers.jsonl`.

## 3. Cài đặt trên máy mới

1. Clone repo, Python **3.13**; cài đúng phiên bản trong [ops/pip_freeze_minirag_venv.txt](ops/pip_freeze_minirag_venv.txt)
   (quan trọng: torch 2.14.0, transformers 5.16.1, sentence-transformers 6.0.1, json_repair 0.63.4, modal 1.6.2,
   scikit-learn 1.9.0). Chạy mọi lệnh từ gốc repo, `PYTHONUTF8=1`.
2. Cần có trong repo (nhánh này đã kèm): `MultiHop-RAG-phi-ropefix609/` (index Phi), `dataset/MultiHop-RAG/`,
   `data-gym-cache/` (cache tiktoken). Phía Qwen mà thang Phi **bắt buộc** đọc: `review/mhr_st_20261112/source/`
   (mã MiniRAG đóng băng, commit b640765), `review/mhr_st_20261112/inputs/{dev,test}.csv`, `review/mhr_st_20261112/embedding_cache.json`.
   Chỉ test dùng: các cache từ khoá Qwen (`inputs/kw_dev_s101.jsonl`, `s*/kw_s*.jsonl`) và `archive_c0/`. Các thư mục Qwen
   khác chỉ có bảng kết quả + PROTOCOL để đối chiếu (profile Phi chặn ghi vào mọi thư mục Qwen).
3. File `.env` **của bạn** (không bao giờ commit): `MINIRAG_SLM_KEY=<token tự đặt>` và các key Gemini
   `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, … (hoặc `GEMINI_API_KEYS` phân tách bằng dấu phẩy). Trỏ tới nó bằng
   `MINIRAG_DOTENV=<đường dẫn tuyệt đối tới .env>`.
4. Modal (workspace của bạn):
   ```
   modal token new --profile <ten>
   modal secret create minirag-slm-key MINIRAG_SLM_KEY=<cùng token như trong .env>
   MODAL_PROFILE=<ten> MODAL_IMAGE_BUILDER_VERSION=2025.06 modal deploy reproduce/modal_slm_phi.py
   ```
   Không sửa `modal_slm_phi.py`. `MODAL_IMAGE_BUILDER_VERSION=2025.06` giữ image giống bản đã dùng.
   URL sẽ là `https://<workspace>--minirag-slm-phi-ropefix-serve.us-east.modal.run`.
5. Kiểm server trước khi chạy (cold start ~1–2 phút):
   `GET /version` → `{"version":"0.11.0"}`; `GET /v1/models` có `microsoft/Phi-3.5-mini-instruct` và `minirag-slm`;
   `modal app logs minirag-slm-phi-ropefix` có `hf_overrides` với `short_factor` bắt đầu bằng `1.08` (= long_factor).
6. **Ghi vào [DEVIATIONS.md](DEVIATIONS.md)** (mục "Modal workspace / credit log"): giờ chuyển, tên workspace mới,
   kết quả kiểm server. Đây là bắt buộc để báo cáo minh bạch.

## 4. Chạy

```
set MINIRAG_DOTENV=<.env của bạn>
set PHI_SLM_URL=https://<workspace>--minirag-slm-phi-ropefix-serve.us-east.modal.run
python -B -m reproduce.mhr_phi.launch reproduce.mhr_phi.c0 execute
```
Hoặc dùng watchdog (tự chạy lại khi chết, giữ máy không ngủ, dừng khi chi phí tháng chạm `CostStop`):
`powershell -File review/mhr_phi_20261118/ops/watchdog_phi_c0.ps1 -Py <python> -Profile <ten> -PhiUrl <url> -DotEnv <.env>`.

Sau C0: làm theo RUNBOOK.md lần lượt **+ST → +F → +CE → +set**, mọi lệnh qua `reproduce.mhr_phi.launch`.
Mỗi bậc học (F, CE, set) có **bước A trên dev và đóng băng trước** khi dựng ngữ cảnh test; không đảo thứ tự.
Bước CE cần chấm cross-encoder trên GPU Modal A10G (`reproduce/mhr_ce/scores.py`, ~1 USD).

Chi phí tham khảo (A100-80GB 2,50 USD/giờ): C0 mỗi seed test ~2–2,5 USD; phần còn lại của ladder ~15 USD.
Ladder Qwen mất ~3 ngày, phần lớn là thời gian chấm Gemini (~1.650 lượt/giờ).

## 5. Quy tắc phải giữ (để còn là thí nghiệm hợp lệ)

- **Không sửa** PROTOCOL.md, mã đóng băng, kết quả Qwen. Mọi thay đổi/ sự cố → thêm mục có ngày vào DEVIATIONS.md,
  **trước** khi xem con số liên quan.
- Ba sửa đổi đã ghi trong DEVIATIONS.md đang áp dụng tự động qua profile: trình đọc JSONL tách theo "
" (`reproduce/mhr_phi/jsonl.py`); (a) câu có trả lời từ khoá làm MiniRAG sập → `early_exit` + `fail_response`
  (`reproduce/mhr_phi/kwguard.py`); (b) lượt chấm vẫn `parse_failed` sau 5 vòng → `unparsed`, tính là sai.
- Không hỏi lại SLM để "sửa" một câu xấu; không đổi seed, prompt, ngân sách 4000 token, judge.
- Laptop: cắm sạc, **không gập nắp** (Modern Standby làm treo tiến trình; watchdog chỉ giữ máy thức khi mở nắp).

## 6. Không đẩy lên git

Các thư mục `snapshot/` (bản sao index cho từng worker, ~160 MB mỗi cái, tự tạo lại khi thiếu), `answer_work/`,
`_offline_test/`, log `*.out.log`/`*.err.log`. Gợi ý thêm vào `.git/info/exclude`:
```
review/mhr_phi_20261118/**/snapshot/
review/mhr_phi_20261118/**/answer_work/
review/mhr_phi_20261118/_offline_test/
review/**/*.out.log
review/**/*.err.log
```

## 7. Khi xong

Bảng cuối nằm ở `review/mhr_phi_20261118/set/BANG_SO_SANH.txt` (+ `RESULTS.json`), cùng định dạng bảng Qwen.
Báo lại: bảng 5 bậc, quyết định từng bậc theo quy tắc 3 seed, các kiểm tra phụ (metric không qua judge, độ trôi judge,
Fleiss κ), số câu `early_exit`/`unparsed`, chi phí Modal theo giai đoạn, mọi mục mới trong DEVIATIONS.md.
