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


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _normalize_messages() -> None:
    for index, message in enumerate(st.session_state.messages):
        if "id" not in message:
            message["id"] = f"msg_{index}"


def _create_message(
    role: str,
    content: str,
    sources: list[str] | None = None,
    chunks: list[dict[str, str | int]] | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {
        "id": f"msg_{len(st.session_state.messages)}",
        "role": role,
        "content": content,
    }
    if sources is not None:
        message["sources"] = sources
    if chunks is not None:
        message["chunks"] = chunks
    return message


def _build_history_pairs() -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    messages = st.session_state.messages
    for pair_index in range(0, len(messages), 2):
        user_message = messages[pair_index]
        assistant_message = messages[pair_index + 1] if pair_index + 1 < len(messages) else {"content": ""}
        pairs.append(
            {
                "pair_index": pair_index // 2,
                "user_index": pair_index,
                "user_message": user_message,
                "assistant_message": assistant_message,
            }
        )
    return pairs


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(settings)

def main() -> None:
    settings.ensure_directories()
    st.set_page_config(page_title=settings.app_name, page_icon="📄", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    _normalize_messages()

    if "selected_message_index" not in st.session_state:
        st.session_state.selected_message_index = None

    query_params = st.query_params

    selected_param = query_params.get("selected")

    if selected_param is not None:
        try:
            st.session_state.selected_message_index = int(selected_param)
        except ValueError:
            pass

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

        # Sidebar history panel
        st.divider()
        st.subheader("History")
        st.markdown(
            """
            <style>
            .history-panel {
                max-height: 360px;
                overflow-y: auto;
                padding-right: 4px;
            }
            .history-item {
                margin-bottom: 8px;
                border: 1px solid #ddd;
                border-radius: 12px;
                background: #fff;
            }
            .history-item.selected {
                border-color: #1f77b4;
                background: #e8f4ff;
            }
            .history-link {
                display: block;
                padding: 10px 12px;
                text-decoration: none;
                color: inherit;
            }
            .history-question {
                font-size: 0.95rem;
                font-weight: 600;
                margin-bottom: 4px;
            }
            .history-answer {
                font-size: 0.9rem;
                color: #4d4d4d;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        history_pairs = _build_history_pairs()
        if history_pairs:
            history_html = ["<div class='history-panel'>"]
            for pair in history_pairs:
                user_text = _truncate_text(pair["user_message"]["content"], 30)
                assistant_text = _truncate_text(pair["assistant_message"]["content"], 50)
                preview = f"Q: {user_text}\nA: {assistant_text}"
                if st.button(preview, key=f"history_{pair['pair_index']}"):
                    st.session_state.scroll_to = pair["user_index"]
            history_html.append("</div>")
            st.markdown("\n".join(history_html), unsafe_allow_html=True)
        else:
            st.info("No chat history yet.")

        # Các nút xóa (Q3)
        st.divider()
        if st.button("Clear History", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.selected_message_index = None
            st.query_params.clear()
            st.rerun()

        if st.button("Clear Vector Store", use_container_width=True):
            get_pipeline.clear()
            st.success("Vector store cleared!")
            st.rerun()

    # --- KHU VỰC CHÍNH ---
    uploaded_file = st.file_uploader("Upload a PDF or DOCX document", type=["pdf", "docx"])
    
    # Render chat messages with anchor targets
    for message in st.session_state.messages:
        st.markdown(f'<div id="{message["id"]}"></div>', unsafe_allow_html=True)
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                sources = message["sources"]
                st.markdown("### 📌 Sources")
                for i, source_item in enumerate(sources, 1):
                    st.markdown(f"[{i}] {source_item}")
            if message.get("chunks"):
                with st.expander("🔍 View retrieved context"):
                    for chunk in message["chunks"]:
                        source_label = chunk.get("source", "unknown")
                        page_label = chunk.get("page", "?")
                        st.markdown(f"📄 {source_label} - page {page_label}")
                        st.code(chunk.get("text", "")[:500])

    # Ô nhập liệu của bạn
    question = st.text_area("Ask a question about the uploaded document", height=120)

    # Nút bấm xử lý
    if st.button("Run SmartDoc", type="primary", disabled=uploaded_file is None and not question.strip()):
        if not question.strip():
            st.warning("Please enter a question.")
            return
            
        # Thêm câu hỏi vào lịch sử
        st.session_state.messages.append(_create_message("user", question.strip()))
        
        # Nếu có file thì mới xử lý RAG thật, nếu không chỉ test UI trả lời giả lập
        if uploaded_file:
            target_path = settings.data_dir / uploaded_file.name
            target_path.write_bytes(uploaded_file.getbuffer())

            response_text = ""
            response_sources: list[str] | None = None
            response_chunks: list[dict[str, str | int]] | None = None

            try:
                document = load_document(target_path)
                with st.spinner("Thinking..."):
                    pipeline = get_pipeline()
                    result = pipeline.answer_question(
                        document,
                        question.strip(),
                        st.session_state.messages,
                    )
                    st.write(result)

                response_text = str(result.answer or "")
                if not response_text:
                    response_text = "Xin lỗi, tôi không nhận được câu trả lời từ mô hình."
                response_sources = result.sources
                response_chunks = result.chunks
            except Exception as exc:
                response_text = f"Đã xảy ra lỗi khi xử lý tài liệu: {exc}"
                st.exception(exc)
            finally:
                st.session_state.messages.append(
                    _create_message(
                        "assistant",
                        response_text,
                        sources=response_sources,
                        chunks=response_chunks,
                    )
                )
        else:
            # Phản hồi giả lập nếu không có file (để test UI nhanh)
            st.session_state.messages.append(_create_message(
                "assistant",
                "Bạn chưa upload tài liệu, nhưng tôi vẫn nhận được câu hỏi của bạn. Hãy upload file để tôi phân tích chính xác hơn nhé!",
            ))

        st.rerun()

    if "scroll_to" in st.session_state:
        target = st.session_state.scroll_to

        st.markdown(f"""
        <script>
        const el = document.getElementById("msg_{target}");
        if (el) {{
            el.scrollIntoView({{behavior: "smooth"}});
        }}
        </script>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("Supported file types: `PDF`, `DOCX`")

if __name__ == "__main__":
    main()