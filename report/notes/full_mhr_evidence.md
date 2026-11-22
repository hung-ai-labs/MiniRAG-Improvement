# Sổ bằng chứng cho lượt chạy toàn bộ MultiHop-RAG

| Khẳng định | Nguồn cần đối chiếu | Điều kiện đo | Cách viết |
|---|---|---|---|
| Bài MiniRAG trình bày mỗi phương pháp bằng một cặp ACC và ERR | Fan và cộng sự, §3.1, Bảng 1, https://arxiv.org/html/2501.06713v2#S3.T1 | Kết quả công bố dùng thiết lập của tác giả gốc | Dùng bố cục cột ACC/ERR, ghi rõ cấu hình và thiết lập tái lập của nghiên cứu này |
| Bộ đánh giá gồm 2.556 câu | MiniRAG-Improvement/dataset/MultiHop-RAG/processed/questions.csv và hai tệp CSV đã đóng băng trong review/mhr_st_20261112/inputs/ | Kiểm số lượng, truy vấn trùng và tính đồng nhất của nội dung theo ID | Toàn bộ 2.556 câu thuộc mẫu số của mỗi cấu hình |
| 2.156 câu có phán quyết hoàn chỉnh cho cả năm cấu hình | per_question.csv trong review/mhr_{st,f,ce,set}_*/test/*_s101/judge_gemini35/ | Qwen2.5-3B-Instruct, 101 là seed chung duy nhất với 400 câu phát triển, Gemini 3.5 Flash Lite, nhãn đa số từ ba lượt chấm | Kết quả hiện có trên 2.156 câu, riêng với thiết lập một lượt sinh |
| 400 câu từng tham gia phát triển phương pháp | dev.csv, dev_offline/scores_F.jsonl, dev_offline/scores_CEF.jsonl và context_manifest.json của lượt chạy mới | F và CE lấy điểm ngoài fold. SET chọn tham số bằng vòng kiểm định chéo bên trong, loại cả fold đánh giá ngoài khỏi dữ liệu dùng để tính điểm chọn tham số | Kiểm định chéo tránh dùng nhãn của chính câu đánh giá khi khớp bộ chấm và chọn tham số theo fold. Thiết kế thành phần đã tham khảo 400 câu này |
| Chỉ công bố điểm toàn tập sau khi chấm đủ | MiniRAG-Improvement/review/mhr_full_20261119/results.json | n_pending bằng 0 và n bằng 2.556 trên từng cấu hình | Dùng ACC/ERR một lượt. Trong khi thiếu 400 câu, bảng toàn tập để trống |
| Bước F dùng tập ứng viên mở rộng | reproduce/mhr_f/run.py, mô tả đầu tệp và kiểm tra phát lại ST | Cùng chỉ mục và ngân sách phần nguồn. Khác số ứng viên được xét và cách xếp hạng | Hiệu số F với ST phản ánh cả việc mở rộng ứng viên và xếp hạng học được |

Định danh bản đóng băng dùng SHA-256 trong context_manifest.json, kể cả các tệp phán quyết gốc.
Các thử nghiệm trước đây dùng ba lượt sinh và hai bộ dữ liệu được lưu trong bản thảo phân tích tương ứng.
