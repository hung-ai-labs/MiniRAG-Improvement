# Chấm chéo DeepSeek–Gemini: hướng dẫn chạy

Mục đích: kiểm tra xem kết luận CE → SET có giữ nguyên khi đổi giám khảo trên **cùng** các câu trả lời không.
Chỉ đổi mô hình giám khảo. Prompt, rubric, bộ đọc nhãn và cách tổng hợp của từng bộ dữ liệu giữ nguyên.

| Bộ dữ liệu | Giám khảo gốc | Giám khảo chấm thêm | Câu trả lời | Lời gọi |
|---|---|---|---|---|
| MultiHop-RAG | `gemini-3.5-flash-lite` | `deepseek-flash` | 12.099 | 36.297 |
| LiHua-World | `deepseek-flash` | `gemini-3.5-flash-lite` | 2.605 | 7.815 |

Phạm vi: cấu hình CE và SET, ba seed, ba lượt chấm cho mỗi câu trả lời khác nhau.
Ước tính DeepSeek khoảng **$7**, có ngưỡng chặn $15. Trong repo chưa có giá Gemini; ước tính khoảng 2,4 triệu token đầu vào.
Ngưỡng chặn Gemini là 10.000 HTTP request, tính cả các lần thử lại.

## Cần có

- Python 3.9 trở lên. Code chỉ dùng thư viện chuẩn, không cần cài thêm gì.
- Key `DEEPSEEK_API_KEY` và key Gemini (`GEMINI_API_KEY`, hoặc `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, …, hoặc `GOOGLE_API_KEY`).
  Code đọc key theo thứ tự: biến môi trường, rồi file `.env` ghi trong `MINIRAG_DOTENV`, rồi `.env` ở gốc repo.
  Code không in và không ghi key ra file. **Không commit `.env`.**

## Các bước (chạy từ gốc repo `MiniRAG-Improvement`)

```powershell
git pull                      # nhánh phi-multihop-ladder
$env:MINIRAG_DOTENV = "C:\duong\dan\toi\.env"     # bash: export MINIRAG_DOTENV=/duong/dan/toi/.env

python -B -m reproduce.cross_judge.test_cross_judge          # 1. test offline, phải ra OK
python -B -m reproduce.cross_judge.run plan --judge both     # 2. không gọi mạng; in số lời gọi và chi phí ước tính
python -B -m reproduce.cross_judge.run smoke --judge both    # 3. 10 lời gọi cho mỗi giám khảo
python -B -m reproduce.cross_judge.analyze --partial         #    xem nhanh: không lỗi là được
python -B -m reproduce.cross_judge.run run --judge both      # 4. chạy toàn bộ
python -B -m reproduce.cross_judge.analyze                   # 5. phân tích
```

Bước 4 ghi kết quả vào journal theo kiểu chỉ ghi thêm. Nếu bị ngắt (mất mạng, tắt máy, lỗi 429…), chạy lại đúng lệnh đó:
code tự bỏ qua các lời gọi đã có phán quyết. Có thể chạy riêng từng giám khảo bằng `--judge deepseek` hoặc `--judge gemini`.

Trên MultiHop-RAG, nếu không đọc được nhãn, code hỏi lại tối đa 3 lần như giám khảo gốc (`judge_run.py --invalid-tries 3`).
Các lần hỏi lại này được ghi `unparsed_reask` trong file `*_calls.jsonl`. Đây là hành vi bình thường.

## Khi có sự cố

- **`BudgetStop`**: đã chạm ngưỡng chi phí hoặc ngưỡng số request. Đừng tự nâng ngưỡng. Báo Tài kèm dòng lỗi và tổng chi phí
  (trường `budget_charge_usd` trong `review/cross_judge_20261124/*_calls.jsonl`). Ngưỡng nằm trong `approved_budget.json`.
- **`Missing API key(s)`**: kiểm tra lại `MINIRAG_DOTENV` hoặc tên biến trong `.env`.
- **Lỗi 401/403**: key sai hoặc hết quyền. Lỗi 429/5xx được tự thử lại; nếu vẫn dừng thì chạy lại lệnh.
- **`analyze` báo `AssertionError: ... recomputed ..., expected ...`**: số tính lại bằng giám khảo gốc không khớp với bảng chính.
  Dừng lại và báo Tài, không sửa số hay sửa code để chạy qua.
- **`analyze` báo `Missing verdicts for ...`**: bước 4 chưa chạy xong; chạy lại bước 4.

## Gửi kết quả lại

Sau bước 5, commit các file sau rồi push lên nhánh `phi-multihop-ladder`:

```powershell
git add review/cross_judge_20261124 report/generated/cross_judge.tex
git commit -m "Cross-judge DeepSeek-Gemini: CE and SET on both datasets"
git push
```

Trước khi commit, kiểm tra `git status` để chắc chắn không có `.env` hay file chứa key.
Thư mục kết quả gồm journal phán quyết, file biên nhận từng lời gọi (`*_calls.jsonl`), `plan.json`, `RESULTS.json` và `approved_budget.json`.
