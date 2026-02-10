# Framework Phân tích Tài liệu OCR

Framework này được thiết kế cho việc Phân tích Tài liệu (Document Parsing), tập trung vào việc trích xuất và sửa lỗi văn bản Tiếng Việt từ hình ảnh tài liệu.

## 🏗 Kiến trúc & Quy trình
Quy trình xử lý của framework bao gồm các bước chính sau:

1.  **Trích xuất văn bản (OCR Extraction)**:
    Sử dụng mô hình OCR để trích xuất nội dung từ hình ảnh. Hiện tại hỗ trợ các phương án:
    *   **LandingADE**: Yêu cầu cấu hình `VISION_AGENT_API_KEY` (lấy từ landingai.com).
    *   **PaddleOCR**: Sử dụng thư viện PaddlePaddle.
    *   **Input**: Một danh sách các dictionary (`List[Dict]`). Tool sẽ tìm trường có tên chứa chữ `path` để xác định đường dẫn file cần xử lý. `Field` là trường mà ta cần phải xử lý. 
    *   **Output**: Trả về một danh sách các dictionary có chứa key `markdown` (ví dụ: `{'markdown': <text>}`).

2.  **Sửa lỗi văn bản (Text Correction)**:
    Áp dụng mô hình ngôn ngữ để sửa lỗi chính tả và ngữ pháp cho văn bản đầu ra của OCR.
    *   Mô hình hiện tại: **legal-tc**.
    *   **Input**: Một danh sách các dictionary (`List[Dict]`). Tool sẽ tìm trường có tên chứa chữ `markdown` để lấy nội dung văn bản cần sửa lỗi. `Field` là trường mà ta cần phải xử lý. 
    *   **Output**: Trả về một danh sách các dictionary có chứa key `markdown` (ví dụ: `{'markdown': <text>}`).
## � Cài đặt & Lưu ý Dependencies

Hiện tại do một số vấn đề về conflict trong dependencies nên bắt buộc phải cài đặt thủ công các gói sau để có thể sử dụng tool `doc_parser`:

```bash
uv pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install -U paddleocr[doc-parser]
```

## ��🔍 Quan sát & Phân tích

Trong quá trình phát triển và thử nghiệm framework, một số quan sát chính đã được ghi nhận liên quan đến sự tương tác giữa các mô hình OCR và quy trình sửa lỗi:

### 1. Nhiễu Markdown (Markdown Interference)
Các mô hình OCR xuất ra văn bản được định dạng Markdown. Việc áp dụng Corrector trực tiếp lên đầu ra thô này (chưa được làm sạch) có thể làm giảm hiệu quả của quá trình sửa lỗi, vì mô hình có thể hiểu sai cú pháp Markdown là lỗi văn bản hoặc ngược lại.

### 2. Vấn đề Ảo giác (Hallucination Issues)
Mặc dù các mô hình OCR hiện đại có khả năng tốt trong việc phân tích ký tự Tiếng Việt, chúng vẫn gặp phải vấn đề **ảo giác**—sinh ra văn bản không tồn tại trong ảnh gốc hoặc tạo ra đầu ra lặp lại không kiểm soát. 

### 3. Nhạy cảm với Chất lượng Ảnh (Augmentation)
Khi ảnh đầu vào bị suy giảm chất lượng (ví dụ: bị mờ), các mô hình OCR có xu hướng mắc lỗi chính tả nhiều hơn đáng kể.

### 4. Xung đột Phiên bản (Version Conflict)
Hiện tại chưa hỗ trợ `GLM-OCR` do xung đột với `protonx-legal-tc`. Có thể ta sẽ cân nhắc một mô hình khác để thay thế, nhưng hiện tại vẫn đang sử dụng `protonx-legal-tc`.

## 💡 Giải pháp Đề xuất

Dựa trên phân tích trên, giải pháp được khuyến nghị là **áp dụng Corrector trên Markdown đã được làm sạch**.

Thay vì đưa trực tiếp đầu ra OCR thô vào corrector, pipeline nên:
1.  Trích xuất nội dung văn bản thuần túy từ Markdown.
2.  Áp dụng Corrector Tiếng Việt để sửa lỗi chính tả (q uan trọng đối với ảnh mờ/nhiễu).

## 📊 Đánh giá & Hướng phát triển
Để đánh giá và cải thiện hệ thống sâu hơn, em dự định tham khảo các bộ benchmark tiêu chuẩn như **OmniDocBench1.5** để kiểm thử toàn diện khả năng phân tích tài liệu.
