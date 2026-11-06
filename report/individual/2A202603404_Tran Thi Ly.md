# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603404
- **Họ và tên:** Trần Thị Lý 
- **SEED** (in ở ô đầu notebook):8436

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.778      0.875        0.889   
2. BoW + MLP (deep learning)         0.987   0.867      0.923        0.857   
3a. Encoder (MiniLM) + kNN           0.859   0.944      0.899        0.947   
3b. Encoder (MiniLM) + LogReg        0.966   0.956      0.961        0.947   
3c. Encoder (MiniLM) + MLP head      0.966   0.933      0.949        0.947   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            3.146      34467.089       0.056   
2. BoW + MLP (deep learning)         4.898      37002.065       0.188   
3a. Encoder (MiniLM) + kNN          79.374         32.780      85.459   
3b. Encoder (MiniLM) + LogReg       79.881         42.935      31.317   
3c. Encoder (MiniLM) + MLP head     80.451         39.884      30.579   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.078           0.0  
2. BoW + MLP (deep learning)          0.313           0.0  
3a. Encoder (MiniLM) + kNN          128.084           0.0  
3b. Encoder (MiniLM) + LogReg        43.208           0.0  
3c. Encoder (MiniLM) + MLP head      40.986           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
🤖 AI log (Báo cáo cá nhân)
1. Cấu hình API & An toàn Colab:
- Hỏi AI: Cách cài OPENAI_API_KEY qua Secrets, sửa lỗi 401 và bảo mật key/output khi nộp file .ipynb.
- Trả lời & Kiểm chứng: Mở tab Secrets (🔑), gán key, bật Notebook access; thực hiện Clear all outputs để không lộ vết dữ liệu trước khi tải file về.
2. Xử lý lỗi Rate Limit (Gemini API):
- Hỏi AI: Nguyên nhân và cách sửa lỗi RateLimitError 429 của Gemini 3.5 Flash Lite.
- Trả lời & Kiểm chứng: Hiểu rõ giới hạn 15 RPM của bản Free Tier
3. Bài toán tính tay (Hand Calculation):
- Hỏi AI: Công thức và lời giải 5 bài toán tính tay (sigmoid_0, f1_example, cosine_example, llm_cost_per_day...).
- Trả lời & Kiểm chứng: Tham khảo cách tính ($\sigma(0)=0.50$, $F1=0.73$, $\text{cosine}=0.60$, $\text{cost}=\$10.50$); tự tính lại từng bước ra giấy và chụp ảnh IMG_0135.HEIC nộp kèm.
4. Thực nghiệm Challenge
- Hỏi AI: Viết code thử nghiệm các ngưỡng quyết định ($0.2, 0.5, 0.8$) và góc nhìn nghiệp vụ chặn nhầm OTP ngân hàng.
- Trả lời & Kiểm chứng: Chạy code lấy chỉ số Precision/Recall/F1; chọn ngưỡng $0.8$ nhằm tối ưu Precision, tránh chặn nhầm tin nhắn OTP quan trọng.
