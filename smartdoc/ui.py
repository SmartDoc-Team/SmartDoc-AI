from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smartdoc.config import settings
from smartdoc.document_loaders import load_document
from smartdoc.ollama_client import OllamaClient
from smartdoc.rag import RAGPipeline

@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(settings)

def main() -> None:
    settings.ensure_directories()
    st.set_page_config(page_title=settings.app_name, page_icon="📄", layout="wide")
    

    if "messages" not in st.session_state:
        st.session_state.messages = [
        ]

    st.title("SmartDoc AI")
    st.caption("RAG document Q&A for PDF and DOCX with FAISS, multilingual embeddings, and Ollama Qwen2.5.")

    ollama_client = OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    healthy, status_message = ollama_client.health_check()

    # --- SIDEBAR (Giữ nguyên giao diện của bạn) ---
    with st.sidebar:
        st.subheader("System Status")
        st.write(f"Model: `{settings.ollama_model}`")
        if healthy:
            st.success(status_message)
        else:
            st.error(status_message)
        st.write(f"Embedding: `{settings.embedding_model_name}`")
        st.write(f"Chunk size: `{settings.chunk_size}`")
        st.write(f"Top-k retrieval: `{settings.retrieval_k}`")

        # Hiển thị lịch sử vắn tắt ở sidebar
        st.divider()
        st.subheader(" History")
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.caption(f"{i//2 + 1}. {msg['content'][:40]}...")

        # Các nút xóa (Q3)
        st.divider()
        if st.button("Clear History", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear Vector Store", use_container_width=True):
            get_pipeline.clear()
            st.success("Vector store cleared!")
            st.rerun()

    # --- KHU VỰC CHÍNH ---
    uploaded_file = st.file_uploader("Upload a PDF or DOCX document", type=["pdf", "docx"])
    
    # Render các bong bóng chat (Q2) bao gồm cả dữ liệu mẫu
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ô nhập liệu của bạn
    question = st.text_area("Ask a question about the uploaded document", height=120)

    # Nút bấm xử lý
    if st.button("Run SmartDoc", type="primary", disabled=uploaded_file is None and not question.strip()):
        if not question.strip():
            st.warning("Please enter a question.")
            return
            
        # Thêm câu hỏi vào lịch sử
        st.session_state.messages.append({"role": "user", "content": question.strip()})
        
        # Nếu có file thì mới xử lý RAG thật, nếu không chỉ test UI trả lời giả lập
        if uploaded_file:
            target_path = settings.data_dir / uploaded_file.name
            target_path.write_bytes(uploaded_file.getbuffer())
            try:
                document = load_document(target_path)
                with st.spinner("Thinking..."):
                    pipeline = get_pipeline()
                    result = pipeline.answer_question(document, question.strip())
                    st.session_state.messages.append({"role": "assistant", "content": result.answer})
            except Exception as exc:
                st.exception(exc)
        else:
            # Phản hồi giả lập nếu không có file (để test UI nhanh)
            st.session_state.messages.append({"role": "assistant", "content": "Bạn chưa upload tài liệu, nhưng tôi vẫn nhận được câu hỏi của bạn. Hãy upload file để tôi phân tích chính xác hơn nhé!"})
        
        st.rerun()

    st.divider()
    st.markdown("Supported file types: `PDF`, `DOCX`")

if __name__ == "__main__":
    main()