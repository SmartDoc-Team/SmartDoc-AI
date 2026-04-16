# Tiêu chuẩn luồng xử lý RAG (RAG Standard Pipeline)

> **Mục đích:** Đây là luồng xử lý RAG (Retrieval-Augmented Generation) chuẩn mực, đơn giản, dễ duy trì và bám sát vào cấu trúc nguyên bản của dự án. Mọi thay đổi hay phát triển thêm về sau do AI thực hiện bắt buộc phải chiếu theo luồng này trước tiên.
 

**Luồng RAG Chuẩn (Baseline RAG)**. Luồng này được chia làm 2 giai đoạn chính:

## Giai đoạn 1: Data Indexing (Tiền xử lý & Lưu trữ - Offline)

**1. Data Ingestion (Thu thập và đọc văn bản)**
- Hỗ trợ các định dạng PDF, DOCX.
- Trích xuất toàn bộ văn bản thô từ tài liệu. Giữ lại thông tin (Metadata) cốt lõi như tên file nguồn (source: tai_lieu.pdf).

**2. Chunking (Cắt đoạn văn bản)**
- **Kỹ thuật:** Cắt theo độ dài cố định tính bằng chữ/token (Fixed-size Chunking) kết hợp độ trùm lặp (Overlap) giữa các đoạn liền kề để duy trì ngữ cảnh.
- **Công cụ dự kiến:** Recursive character text splitter.

**3. Embedding (Nhúng vector ngữ nghĩa)**
- Nhúng các chunk văn bản thu được thành vector số học.
- **Model:** Sử dụng mô hình đa ngôn ngữ (ví dụ: sentence-transformers hoặc bg3-m3(của ollama k phải hugging face)).

**4. Storage (Vector Database)**
- Lưu trữ toàn bộ các Vector vừa tạo kèm theo Metadata tương ứng.
- **Công cụ:** FAISS (Không cần cài cắm server phức tạp, lưu trực tiếp dưới dạng file nội bộ).

---

## Giai đoạn 2: Retrieval & Generation (Truy xuất & Sinh câu trả lời - Online)

**5. Bước Query (Nhận câu hỏi)**
- Nhận câu hỏi từ người dùng qua Streamlit UI.

**6. Query Embedding (Mã hóa câu lệnh)**
- Chuyển đổi câu hỏi của người dùng thành Vector bằng **cùng một Embedding Model** ở Giai đoạn 1.

**7. Retrieval (Truy xuất dữ liệu)**
- Tính toán độ tương đồng (nghĩa rộng) giữa Vector câu hỏi và Vector trong FAISS.
- Trích xuất **Top-K** (từ 3 đến 5) đoạn văn bản (chunks) gần nghĩa nhất.

**8. Prompt Building (Đóng gói Prompt)**
- Lắp ráp ngữ cảnh: Ghép gọn Top-K chunks thành một Context String.
- Áp dụng một định dạng Prompt chặt chẽ (Strict Format) yêu cầu LLM: *"Chỉ được phép trả lời dựa trên những ngữ cảnh được đánh kèm, tuyệt đối không bịa thông tin rác ngoài lề."*

**9. LLM Generation (Sinh nội dung)**
- Gửi Prompt cuối cùng tới Local LLM thông qua API.
- **Model:** qwen2.5:7b tích hợp qua nền tảng Ollama.

**10. Post-processing (Hậu xử lý - Tùy chọn nhẹ nhàng)**
- Dựa vào Metadata của cái Chunk được chọn, đính kèm thêm nguồn trích dẫn gốc (Citation: "Nguồn từ file ABC").
- Định dạng xuất chuẩn Markdown để hiển thị mượt trên UI.

---

### Advanced RAG (chưa cần thiết)
1. Ingestion
2. Chunking (semantic + metadata)
3. Embedding
4. Storage (FAISS)
5. Query Processing
   - Rewrite
   - Expansion
6. Retrieval
   - Dense (FAISS)
   - BM25
6.5 Re-ranking 
7. Prompt Building (strict format)
8. LLM Generation (Qwen2.5 via Ollama)
9. Post-processing
   - Citation
   - Format output
10. Evaluation (offline/online)
