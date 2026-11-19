# MultiHop-RAG — tổng quan nghiên cứu ablation

## Mục tiêu và thiết kế

Nghiên cứu đo phần đóng góp của từng thay đổi trong khâu truy xuất và chọn bằng chứng của MiniRAG. MultiHop-RAG gồm 609 bài báo. Tập chia nội bộ có 2,156 câu test và 400 câu dev lấy mẫu phân tầng theo loại câu hỏi. [Dataset](../dataset/MultiHop-RAG/README.md) · [Manifest tập chia](../dataset/MultiHop-RAG/splits/split_manifest.json)

| Loại câu hỏi | Dev | Test nội bộ |
|---|---:|---:|
| `comparison` — so sánh | 134 | 722 |
| `inference` — suy luận | 128 | 688 |
| `temporal` — thời gian | 91 | 492 |
| `null` — thiếu thông tin | 47 | 254 |

Số liệu theo [manifest tập chia](../dataset/MultiHop-RAG/splits/split_manifest.json).

| Bậc tích lũy | Thay đổi so với bậc trước | Nguồn |
|---|---|---|
| C0 | Mốc MiniRAG đóng băng từ phase 3, dùng truy xuất native ở chế độ `mini`. Đáp án Qwen lấy từ bản lưu và được chấm lại bằng judge đã chốt. | [Protocol C0/ST](mhr_st_20261112/protocol.json) · [Mã QA đóng băng](mhr_st_20261112/source/reproduce/research/phased_qa.py) |
| +ST | Ánh xạ ngày xuất bản của bài báo tới các chunk để ưu tiên truy xuất theo mốc thời gian trong câu hỏi. Câu không có mốc dùng truy xuất C0. | [Adapter ST](../reproduce/mhr_st/adapter.py) · [Protocol C0/ST](mhr_st_20261112/protocol.json) |
| +F | Mở pool theo quy tắc dev và xếp hạng lại bằng hồi quy logistic với 18 đặc trưng. A1 vẫn cắt khối Sources theo ngân sách 4000 token. | [Mã chạy F](../reproduce/mhr_f/run.py) · [Bước A của F](../reproduce/mhr_f/study.py) |
| +CE | Thêm logit cross-encoder zero-shot vào scorer F, xếp hạng lại cùng pool rồi áp dụng A1. | [Protocol CE](mhr_ce_20261116/PROTOCOL.md) · [Mã tính logit](../reproduce/mhr_ce/modal_ce.py) |
| +set | Dùng điểm CE+F để chọn tập chunk theo lợi ích biên trên mỗi token, có xét độ phủ bài báo và các nhóm từ khóa/mốc thời gian. Bỏ qua chunk vượt ngân sách còn lại. | [Protocol SET](mhr_set_20261117/PROTOCOL.md) · [Mã bộ chọn](../reproduce/mhr_set/select.py) |

SLM là Qwen2.5-3B trên Modal với vLLM 0.11.0. Các seed sinh test là `101`, `202`, `303`. Judge `gemini-3.5-flash-lite` chấm 3 lượt mỗi đáp án và lấy nhãn đa số. Với cùng câu hỏi và seed, ngữ cảnh trùng bậc trước thì tái dùng đáp án, đáp án trùng một bậc trước thì dùng chung các lượt chấm. [Protocol C0/ST](mhr_st_20261112/protocol.json) · [Quy tắc chấm ST](mhr_st_20261112/amendment_20261113_move_and_shared_verdicts.json) · [Mã chạy F](../reproduce/mhr_f/run.py) · [Protocol CE](mhr_ce_20261116/PROTOCOL.md) · [Protocol SET](mhr_set_20261117/PROTOCOL.md)

Protocol của mỗi bậc được ghi trước khi xem số liệu của bậc đó. Một bậc cải thiện so với bậc liền trước khi ACC tăng ở cả 3 seed và kiểm định McNemar chính xác cho `p < 0.05` ở từng seed. Các bước A trên dev của F, CE và SET mô tả khả năng truy xuất và dùng để đóng băng cấu hình trước khi dựng ngữ cảnh test. [Protocol C0/ST](mhr_st_20261112/protocol.json) · [Kết quả đăng ký của F](mhr_f_20261114/BANG_SO_SANH.txt) · [Protocol CE](mhr_ce_20261116/PROTOCOL.md) · [Protocol SET](mhr_set_20261117/PROTOCOL.md)

## Kết quả Qwen

| Bậc | ACC ± SD (%) | ERR (%) | Neither (%) | Recall bằng chứng (micro) | Đủ bằng chứng | Token Sources trung bình |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 57.90 ± 0.26 | 29.70 | 12.29 | 0.168 | 3.3% | 3476.8 |
| C0 + ST | 58.91 ± 0.64 | 29.50 | 11.49 | 0.221 | 6.0% | 3480.7 |
| C0 + ST + F | 63.81 ± 1.03 | 28.11 | 7.93 | 0.319 | 9.6% | 3441.8 |
| C0 + ST + F + CE | 64.70 ± 0.60 | 27.61 | 7.58 | 0.336 | 11.1% | 3442.7 |
| C0 + ST + F + CE + set | 64.97 ± 0.51 | 26.90 | 7.95 | 0.336 | 10.5% | 3215.1 |

ACC, ERR và Neither dùng nhãn đa số, còn SD là độ lệch chuẩn mẫu giữa các seed sinh. Một câu đủ bằng chứng khi mỗi fact vàng có ít nhất một chunk tương ứng trong Sources. Token Sources là tổng token nội dung chunk, lấy trung bình trên toàn bộ câu test. [Bảng tổng hợp Qwen](mhr_set_20261117/BANG_SO_SANH.txt) · [Định nghĩa trong RESULTS.json](mhr_set_20261117/RESULTS.json) · [Định nghĩa đủ bằng chứng](../reproduce/mhr_f/study.py)

| So sánh | Seed 101 (Δ ACC / p) | Seed 202 (Δ ACC / p) | Seed 303 (Δ ACC / p) | Quyết định | Nguồn |
|---|---:|---:|---:|---|---|
| ST so với C0 | +1.30 / 0.0211995 | +0.56 / 0.311231 | +1.16 / 0.023534 | Chưa xác nhận | [ST](mhr_st_20261112/BANG_SO_SANH.txt) |
| F so với ST | +5.15 / 1.5529e-05 | +6.22 / 6.44578e-08 | +3.34 / 0.00354277 | Xác nhận | [F](mhr_f_20261114/BANG_SO_SANH.txt) |
| CE so với F | +0.88 / 0.34744 | +0.32 / 0.749514 | +1.48 / 0.0954578 | Chưa xác nhận | [CE](mhr_ce_20261116/BANG_SO_SANH.txt) |
| SET so với CE | +0.19 / 0.884456 | -0.23 / 0.844733 | +0.83 / 0.389359 | Chưa xác nhận | [SET](mhr_set_20261117/BANG_SO_SANH.txt) |
| SET so với F | +1.07 / 0.339727 | +0.09 / 0.963216 | +2.32 / 0.0228173 | So sánh phụ vì CE chưa được xác nhận trên F | [SET](mhr_set_20261117/BANG_SO_SANH.txt) |
| SET so với C0 | +7.51 / 1.28908e-10 | +6.86 / 2.83531e-09 | +6.82 / 2.16987e-09 | So với mốc, tăng ở mọi seed với p dưới ngưỡng | [SET](mhr_set_20261117/BANG_SO_SANH.txt) |

Δ ACC tính bằng điểm phần trăm. `p` là giá trị McNemar chính xác theo từng seed. Các bảng của F và CE cũng có đối chiếu với C0. [Bảng F](mhr_f_20261114/BANG_SO_SANH.txt) · [Bảng CE](mhr_ce_20261116/BANG_SO_SANH.txt)

**Theo loại câu hỏi.** Đây là phân tích thăm dò, chưa hiệu chỉnh kiểm định nhiều lần. Các ô tóm tắt hướng thay đổi ACC theo seed.

| Loại | ST so với C0 | F so với ST | CE so với F | SET so với CE |
|---|---|---|---|---|
| `comparison` | Bằng nhau | Tăng ở mọi seed | Tăng ở mọi seed | Giảm ở mọi seed |
| `inference` | Tăng ở seed 101, bằng nhau ở các seed còn lại | Tăng ở mọi seed, với p dưới ngưỡng ở từng seed | Giảm ở seed 202, bằng nhau ở các seed còn lại | Giảm ở mọi seed |
| `temporal` | Tăng ở mọi seed | Có tăng và giảm | Có tăng và giảm | Có tăng, giảm và bằng nhau |
| `null` | Bằng nhau | Có tăng và giảm | Có tăng và giảm | Tăng ở mọi seed |

Số liệu ghép cặp trong `seeds.*.by_type` của [ST](mhr_st_20261112/RESULTS.json), [F](mhr_f_20261114/RESULTS.json), [CE](mhr_ce_20261116/RESULTS.json) và [SET](mhr_set_20261117/RESULTS.json).

**Bước A trên dev.** Bảng ghi số câu có đủ bằng chứng sau khi chọn Sources, trên 353 câu dev có nhãn, dùng pool của seed 101. Điểm của scorer được tính ngoài fold huấn luyện. ORACLE dùng nhãn vàng để làm mốc tham chiếu. [Protocol CE](mhr_ce_20261116/PROTOCOL.md) · [Protocol SET](mhr_set_20261117/PROTOCOL.md) · [Nhãn và các cấu hình trong mã F](../reproduce/mhr_f/study.py)

| Bước | Số câu đủ bằng chứng theo cấu hình | Nguồn |
|---|---|---|
| F | C0 10 · ST 13 · F 38 · ORACLE 155 | [Dev F](mhr_f_20261114/dev_offline/offline_eval.json) |
| CE | C0 10 · ST 13 · F 38 · CE+F 48 · CE 30 · ORACLE 155 | [Dev CE](mhr_ce_20261116/dev_offline/offline_eval.json) |
| SET | CE+F 48 · SET 48 · `SET_beta0` 49 · `SET_tau0` 51 · ORACLE 155 | [Dev SET](mhr_set_20261117/dev_offline/offline_eval.json) |

Ở bước chọn pool trước khi cắt Sources, số câu đủ bằng chứng tương ứng là `native@30` 112, `native@45` 127, `native@60` 130, `native@90` 138, `native@120` 147, `native@all` 158 và `all+dense30` 268. Quy tắc chọn mức cắt native nhỏ nhất trong sai số cho phép 8 câu so với pool native chưa cắt, nên chốt `native@all`. SET đóng băng β = `0.0` và τ = `0.05` trên dev trước khi dựng ngữ cảnh test. [Độ phủ pool F](mhr_f_20261114/dev_offline/coverage.json) · [Quy tắc chọn pool](../reproduce/mhr_f/study.py) · [Tham số SET đóng băng](mhr_set_20261117/set_params.json)

Theo tiêu chí đăng ký, chỉ +F được xác nhận cải thiện ACC so với bậc liền trước. Các so sánh ghép cặp chưa xác nhận cải thiện ở +ST, +CE hay +set. SET cao hơn C0 qua từng seed, phản ánh chênh lệch tích lũy của cả thang. [Bảng ST](mhr_st_20261112/BANG_SO_SANH.txt) · [Bảng F](mhr_f_20261114/BANG_SO_SANH.txt) · [Bảng CE](mhr_ce_20261116/BANG_SO_SANH.txt) · [Bảng SET](mhr_set_20261117/BANG_SO_SANH.txt)

## Độ tin cậy của chấm điểm

| Bậc | Fleiss κ seed 101 | Seed 202 | Seed 303 | Nguồn |
|---|---:|---:|---:|---|
| C0 | 0.912 | — | — | [Kiểm tra F seed 101](mhr_f_20261114/judge_validity_s101/output.txt) |
| ST | 0.916 | — | — | [Kiểm tra F seed 101](mhr_f_20261114/judge_validity_s101/output.txt) |
| F | 0.902 | — | — | [Kiểm tra F seed 101](mhr_f_20261114/judge_validity_s101/output.txt) |
| CE | 0.907 | 0.925 | 0.916 | [Kiểm tra phụ CE](mhr_ce_20261116/secondary_checks/output.txt) |
| SET | 0.907 | 0.924 | 0.911 | [Kiểm tra phụ trong RESULTS.json](mhr_set_20261117/RESULTS.json) |

Fleiss κ đo độ nhất quán giữa 3 lượt chấm. Dấu `—` chỉ seed nằm ngoài kiểm tra κ được báo cáo cho C0, ST và F. CE và SET dùng cột `all`, gồm cả nhãn dùng chung, và có báo cáo riêng cho đáp án mới chấm. [Ghi chú kiểm tra F](mhr_f_20261114/judge_validity_s101/NOTE.txt) · [Kiểm tra CE](mhr_ce_20261116/secondary_checks/output.txt) · [Kết quả SET](mhr_set_20261117/RESULTS.json)

| Kiểm tra độ trôi | Đồng thuận nhãn đa số khi chấm lại | Nguồn |
|---|---|---|
| Khi chấm F, lấy mẫu C0 seed 101 | 193/200 (96.5%) | [Drift F](mhr_f_20261114/judge_drift_check/RESULT.json) |
| Khi chấm CE, lấy mẫu F seed 101 | 195/200 (97.5%) | [Drift CE](mhr_ce_20261116/secondary_checks/judge_drift/RESULT.json) |
| Khi chấm SET, lấy mẫu CE seed 101 | 192/200 (tỷ lệ `0.96` trong nguồn) | [Drift SET trong RESULTS.json](mhr_set_20261117/RESULTS.json) |

Chỉ số độc lập dò đáp án vàng đã chuẩn hóa theo ranh giới từ trong câu trả lời, trên 771 câu có đáp án thực thể (loại khác `null`, đáp án vàng khác yes/no). Ở seed 101, F so với ST tăng từ 76.65 lên 82.62 với `p=0.00019`. CE so với F cho `p=0.615`, `p=0.134`, `p=1` theo thứ tự seed. SET thấp hơn CE ở mọi seed theo chỉ số này và thấp hơn F ở seed 101 và 303. Với 1131 câu yes/no, chỉ judge được dùng để chấm vì mô hình thường diễn đạt câu trả lời hàm ý. [Kiểm tra F](mhr_f_20261114/judge_validity_s101/output.txt) · [Kiểm tra CE](mhr_ce_20261116/secondary_checks/output.txt) · [Kiểm tra SET](mhr_set_20261117/RESULTS.json)

Trước khi xem nhãn tổng hợp của SET, nhóm bổ sung quy tắc ghi `unparsed` cho lượt vẫn `parse_failed` sau 5 vòng thử (15 lần hỏi). Nhãn tổng hợp `unparsed` hoặc `no_majority` không tính vào ACC, ERR hay Neither. Fleiss κ chỉ dùng câu có đủ nhãn hợp lệ ở 3 lượt, với 1 câu bị loại ở mỗi seed SET. [Sửa đổi SET](mhr_set_20261117/DEVIATIONS.md) · [Kết quả SET](mhr_set_20261117/RESULTS.json)

## Thang Phi-3.5-mini

Phi-3.5-mini-instruct là SLM thứ hai để kiểm tra mức độ lặp lại của kết quả theo cùng quy tắc. Bản sửa LongRoPE của Hùng đặt `short_factor = long_factor` bằng `--hf-overrides` để tránh lỗi đổi hệ số theo batch của vLLM 0.11.0. Phi tự trích xuất và gleaning để dựng index riêng, tạo cache từ khóa và sinh đáp án. F và CE được fit lại, còn SET chọn lại tham số trên pool dev của Phi trước khi chạy test. [Protocol Phi](mhr_phi_20261118/PROTOCOL.md) · [Ứng dụng Modal](../reproduce/modal_slm_phi.py) · [Bàn giao Phi](mhr_phi_20261118/HANDOFF.md)

| Cổng I | Số liệu ghi nhận |
|---|---|
| Pilot Phi | Qua cổng, 30 bài, 0/69 chunk không có thực thể |
| Index đầy đủ Phi | Qua cổng, 609 bài, 1656 chunk, 22841 nút, 13727 cạnh, cả 6 kiểm tra xác minh đều đạt |
| Chunk không có thực thể | Phi 3/1656 (0.2%), ngưỡng ≤ 5%. Đối chiếu Qwen 5/1656 (0.3%), 18561 nút và 14849 cạnh |
| Yêu cầu dựng index Phi | 3312 yêu cầu, 95 kết thúc do giới hạn độ dài (91 gleaning, 4 trích xuất) |

Số liệu theo [biên nhận cổng I](mhr_phi_20261118/GATE_I.json).

| Bước | Trạng thái lúc bàn giao ngày 2026-11-19, 15:45 |
|---|---|
| Server và index Phi | Server đã kiểm model, vLLM 0.11.0 và cấu hình LongRoPE. Index đầy đủ đã qua cổng I. |
| C0 dev s101 | Hoàn tất. |
| C0 test s101 | Hoàn tất. |
| C0 test s202 | Từ khóa và ngữ cảnh đã xong, `mhr-0315` ở trạng thái `early_exit`. `status.json` ghi `failed` tại pha `answers`. Theo bàn giao, 2.156 đáp án đã đủ trên đĩa và `c0 execute` sẽ hoàn tất biên nhận bằng các đáp án đã lưu. |
| C0 test s303 | Chưa chạy. |
| +ST, +F, +CE, +set | Chưa chạy. |
| Chấm Gemini | Chưa chấm Phi. |

Trạng thái theo [c0/status.json](mhr_phi_20261118/c0/status.json), [bàn giao](mhr_phi_20261118/HANDOFF.md) và [cổng I](mhr_phi_20261118/GATE_I.json).

### Sai lệch và sửa đổi của Phi

- Cache từ khóa của `mhr-0315` ở seed 202 có phần tử không phải chuỗi sau khi sửa cú pháp JSON. Quy tắc chung cho mọi bậc Phi trả `early_exit` và `fail_response` từ cache đã đóng băng. [Sai lệch Phi](mhr_phi_20261118/DEVIATIONS.md)
- Trình đọc JSONL của Phi tách theo ký tự xuống dòng `\n` để giữ nguyên ký tự U+0085 nằm trong 2 đáp án C0 seed 202 và đọc đủ bản ghi. [Sai lệch Phi](mhr_phi_20261118/DEVIATIONS.md)
- Watchdog dừng C0 ở `29.30 USD` trên workspace cũ. Bàn giao yêu cầu tiếp tục bằng workspace Modal của người nhận và ghi thời điểm chuyển cùng kết quả kiểm server. [Sai lệch Phi](mhr_phi_20261118/DEVIATIONS.md) · [Bàn giao Phi](mhr_phi_20261118/HANDOFF.md)
- Bậc SET của Phi áp dụng quy tắc `unparsed` đã bổ sung cho Qwen SET ngay từ đầu. [Protocol Phi](mhr_phi_20261118/PROTOCOL.md) · [Sửa đổi SET](mhr_set_20261117/DEVIATIONS.md)

**Chạy tiếp từ [HANDOFF.md](mhr_phi_20261118/HANDOFF.md).** Kiểm tra deployment Phi trên workspace mới, ghi chuyển đổi vào `DEVIATIONS.md`, rồi chạy `python -B -m reproduce.mhr_phi.launch reproduce.mhr_phi.c0 execute` để hoàn tất C0. Sau đó chạy +ST → +F → +CE → +set theo runbook. Làm bước A và đóng băng từng scorer hoặc bộ tham số trước khi dựng ngữ cảnh test của bậc đó. Protocol Phi chốt dùng pool do F chọn trên dev cho cả CE và SET. Nếu pool là một mức cắt nhỏ hơn `native@all`, kiểm tra hiện tại của CE và SET sẽ chặn, như mục **Blocked** của runbook ghi nhận. [Runbook](../reproduce/mhr_phi/RUNBOOK.md) · [Giải thích về pool trong protocol Phi](mhr_phi_20261118/PROTOCOL.md#clarification-2026-11-18-2300-before-any-phi-keyword-pool-or-answer-existed)

So sánh giữa hai SLM chỉ mang tính mô tả. Phi dùng mặc định vLLM (`temperature=1.0`, `top_p=1.0`) và có can thiệp LongRoPE. Qwen dùng cấu hình sampling của model (`temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition penalty=1.05`). Kiểm định ghép cặp được thực hiện giữa các bậc trong từng thang. [Protocol Phi](mhr_phi_20261118/PROTOCOL.md)

## Bản đồ thư mục

| Đường dẫn | Nội dung |
|---|---|
| [review/mhr_st_20261112/](mhr_st_20261112/) | [Protocol](mhr_st_20261112/protocol.json), [bảng C0/ST](mhr_st_20261112/BANG_SO_SANH.txt), [số liệu chi tiết](mhr_st_20261112/RESULTS.json), bản lưu C0, mã đóng băng, câu hỏi và cache embedding dùng lại cho Phi. |
| [review/mhr_f_20261114/](mhr_f_20261114/) | [Bảng F](mhr_f_20261114/BANG_SO_SANH.txt), [số liệu chi tiết](mhr_f_20261114/RESULTS.json), [model F](mhr_f_20261114/model_F.json), dev, [kiểm tra độ trôi](mhr_f_20261114/judge_drift_check/RESULT.json) và [kiểm tra judge seed 101](mhr_f_20261114/judge_validity_s101/). |
| [review/mhr_ce_20261116/](mhr_ce_20261116/) | [Protocol](mhr_ce_20261116/PROTOCOL.md), [bảng CE](mhr_ce_20261116/BANG_SO_SANH.txt), [số liệu chi tiết](mhr_ce_20261116/RESULTS.json), [model CE](mhr_ce_20261116/model_CE.json), dev và kiểm tra phụ. |
| [review/mhr_set_20261117/](mhr_set_20261117/) | [Protocol](mhr_set_20261117/PROTOCOL.md), [sai lệch](mhr_set_20261117/DEVIATIONS.md), [bảng toàn thang Qwen](mhr_set_20261117/BANG_SO_SANH.txt), [số liệu chi tiết](mhr_set_20261117/RESULTS.json), [tham số SET](mhr_set_20261117/set_params.json), dev và kiểm tra phụ. |
| [review/mhr_phi_20261118/](mhr_phi_20261118/) | [Bàn giao](mhr_phi_20261118/HANDOFF.md), [protocol](mhr_phi_20261118/PROTOCOL.md), [sai lệch](mhr_phi_20261118/DEVIATIONS.md), [cổng I](mhr_phi_20261118/GATE_I.json), trạng thái và checkpoint C0, mã dựng index và công cụ vận hành. |
| [reproduce/mhr_st/](../reproduce/mhr_st/) | Adapter source-time, replay C0, chạy ST, chấm và báo cáo. |
| [reproduce/mhr_f/](../reproduce/mhr_f/) | Thu pool, chọn pool và fit F trên dev, chạy và báo cáo test. |
| [reproduce/mhr_ce/](../reproduce/mhr_ce/) | Tính logit cross-encoder, fit và đóng băng CE+F, dựng lại Sources, chạy và báo cáo test. |
| [reproduce/mhr_set/](../reproduce/mhr_set/) | Bộ chọn tập chunk, chọn β/τ trên dev, chạy test và tổng hợp cả thang. |
| [reproduce/mhr_phi/](../reproduce/mhr_phi/) | Launcher, profile Phi, bộ chạy C0, kiểm tra phụ và [runbook](../reproduce/mhr_phi/RUNBOOK.md). |
| [MultiHop-RAG-phi-ropefix609/](../MultiHop-RAG-phi-ropefix609/) | Index Phi đầy đủ kèm [manifest](../MultiHop-RAG-phi-ropefix609/index_manifest.json), đã qua cổng I. |
| `MultiHop-RAG-qwen-full609/` | Index Qwen đóng băng dùng cho thang Qwen, để ngoài Git. [Ghi nhận bản sao index](mhr_st_20261112/amendment_20261113_move_and_shared_verdicts.json) |
| [dataset/MultiHop-RAG/](../dataset/MultiHop-RAG/) | Corpus và câu hỏi gốc, dữ liệu chuẩn hóa, nhãn vàng, ánh xạ bằng chứng sang chunk và tập chia đóng băng. [Mô tả dataset](../dataset/MultiHop-RAG/README.md) |

Các bậc Qwen có `test/<ROW>_s<seed>/answers.csv` và `test/<ROW>_s<seed>/judge_gemini35/per_question.csv`. `ROW` là `C0`, `ST`, `F`, `CE` hoặc `SET`. C0 và ST dùng chung thư mục thí nghiệm ST. [C0/ST](mhr_st_20261112/test/) · [F](mhr_f_20261114/test/) · [CE](mhr_ce_20261116/test/) · [SET](mhr_set_20261117/test/)

Các bản sao index cho worker `snapshot/` (và `snapshots/`), `answer_work/`, pool và ngữ cảnh trung gian của Qwen, index `MultiHop-RAG-qwen-full609/` và log thô được chủ động để ngoài Git. Có thể tái tạo các đầu ra này bằng mã chạy tương ứng hoặc xin bản lưu từ nhóm tác giả. [Bàn giao Phi](mhr_phi_20261118/HANDOFF.md) · [Hồ sơ lưu trữ C0/ST](mhr_st_20261112/amendment_20261113_move_and_shared_verdicts.json) · [Protocol CE](mhr_ce_20261116/PROTOCOL.md)

## Hạn chế đã biết

- Test là tập chia nội bộ, đã được dùng trong thí nghiệm RRF trước đó. Corpus dùng chung nên 1,604 trong 1,902 câu test có bằng chứng chia sẻ ít nhất một bài bằng chứng với dev, và 638 câu có tập bài bằng chứng trùng hoàn toàn với một câu dev. [Protocol C0/ST](mhr_st_20261112/protocol.json) · [Manifest tập chia](../dataset/MultiHop-RAG/splits/split_manifest.json)
- Phân tích theo loại câu hỏi mang tính thăm dò, chưa hiệu chỉnh kiểm định nhiều lần. Các kiểm tra judge đo độ nhất quán và độ trôi trên mẫu, còn câu yes/no vẫn phụ thuộc vào suy luận của judge. [Protocol CE](mhr_ce_20261116/PROTOCOL.md) · [Protocol SET](mhr_set_20261117/PROTOCOL.md) · [Kiểm tra F](mhr_f_20261114/judge_validity_s101/NOTE.txt)
- Qwen SET đã đổi workspace Modal giữa lúc sinh. Model, phiên bản vLLM, loại GPU và seed được giữ, nhưng batching của vLLM có thể tạo kết quả khác ở mức bit giữa các lần chạy. [Sai lệch SET](mhr_set_20261117/DEVIATIONS.md)
- Protocol Phi được viết sau khi biết kết quả Qwen. Khác biệt về sampling và can thiệp LongRoPE giới hạn diễn giải giữa hai SLM. Digest mã nguồn đã ghi khi dựng index Qwen chưa tái tạo được từ các commit, và app Phi không ghim `--revision` nên cần xác minh revision đã phục vụ bằng bằng chứng triển khai hoặc cache phía server. [Protocol Phi](mhr_phi_20261118/PROTOCOL.md) · [Runbook](../reproduce/mhr_phi/RUNBOOK.md)
