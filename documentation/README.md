# SmartDoc AI - Intelligent Document Q&A System

Dự án xây dựng hệ thống hỏi đáp tài liệu thông minh sử dụng kỹ thuật RAG (Retrieval-Augmented Generation), cho phép người dùng tương tác trực tiếp với các tệp PDF thông qua ngôn ngữ tự nhiên.

## 👥 Thành Viên Nhóm

| STT | Họ tên | MSSV |
|-----|--------|------|
| 1 | Nguyễn Tuấn Vũ | 3122410483 |
| 2 | Quách Hữu Vinh | 3122410477 |
| 3 | Vũ Hoàng Chung | 3122560007 |
| 4 | Nguyễn Thanh Tú | 3121410549 |

## 1) Tổng quan dự án

1. **Mục tiêu:** 
    - Xây dựng giao diện web thân thiện cho phép người dùng tải lên tài liệu PDF
    - Tích hợp công nghệ embedding để chuyển đổi văn bản thành vector
    - Triển khai vector database (FAISS) để lưu trữ và tìm kiếm hiệu quả
    - Tích hợp mô hình ngôn ngữ lớn Qwen2.5:7b được tối ưu cho tiếng Việt
    - Xây dựng pipeline RAG hoàn chỉnh từ document loading đến answer generation
    - Tối ưu hóa hiệu suất và độ chính xác của câu trả lời
2. **Trạng thái hiện tại:** Đã hoàn thiện khung sườn dự án (Skeleton), thiết lập môi trường ảo và cài đặt các thư viện cốt lõi (LangChain, FAISS, Streamlit).

## 2) Kiến trúc tổng quan

Hệ thống được thiết kế theo mô hình multi-layer architecture là 

## 3) Tech stack
- Frontend: Streamlit 1.41.1.
- AI Framework: LangChain 0.3.16.
- LLM Engine: Ollama (Mô hình Qwen2.5:7b).
- Vector Database: FAISS 1.9.0.
- Embedding Model: Multilingual MPNet (768-dimensional).
- Document Processing: PDFPlumber.

## 4) Cấu trúc dự án 

SmartDoc-AI/
├── app.py                # File chạy chính (Streamlit UI & Logic)
├── data/                 # Thư mục chứa tài liệu PDF mẫu
├── documentation/        # Báo cáo và tài liệu hướng dẫn
├── logs/                 # Nhật ký hoạt động của hệ thống
├── venv/                 # Môi trường ảo Python
├── requirements.txt      # Danh sách thư viện cài đặt
├── .gitignore            # Cấu hình bỏ qua các file không cần thiết
└── README.md             # Hướng dẫn dự án

## 5) Hướng dẫn cài đặt & Cấu hình

1. **Clone hoặc tải project**
- git clone https://github.com/TuanVuDCT1221/SmartDoc-AI.git
- cd SMARTDOC-AI
2. **Tạo virtual environment**
- 'python -m venv venv'

- 'source venv/bin/activate' # Linux / Mac
# or
- 'venv \Scripts\activate' # Windows
3. **Cài đặt dependencies**
- 'pip install -r requirements.txt'
4. **Cài đặt Ollama**
- # Download from https :// ollama .ai
- # Then pull Qwen2.5:7b model
- 'ollama pull qwen2.5:7b'
5. **Chạy ứng dụng**
- 'streamlit run app.py'
