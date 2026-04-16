from __future__ import annotations

import hashlib
import logging
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smartdoc.config import settings
from smartdoc.document_loaders import LoadedDocument, load_document
from smartdoc.ollama_client import OllamaClient
from smartdoc.rag import RAGPipeline
from smartdoc.benchmarking import run_chunk_benchmark


logger = logging.getLogger(__name__)
MERGED_DOC_CACHE_VERSION = "v2"


def _append_ingestion_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    log_file = settings.log_dir / "ingestion.log"
    with log_file.open("a", encoding="utf-8") as file:
        file.write(log_line)


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
    latency: float | None = None,
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
    if latency is not None:
        message["latency"] = latency
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


def _uploaded_files_signature(uploaded_files: list[object]) -> str:
    hasher = hashlib.md5(usedforsecurity=False)
    for uploaded in uploaded_files:
        content = uploaded.getvalue()
        hasher.update(uploaded.name.encode("utf-8", errors="ignore"))
        hasher.update(str(len(content)).encode("utf-8"))
        hasher.update(content)
    return hasher.hexdigest()


def _merge_loaded_documents(target_docs: list[LoadedDocument]) -> LoadedDocument:
    if len(target_docs) == 1:
        return target_docs[0]

    merged_pages: list[tuple[int, str]] = []
    synthetic_page = 1

    for doc in target_docs:
        if doc.pages:
            for page_num, text in doc.pages:
                tagged_text = f"[SOURCE: {doc.source_name} | PAGE: {page_num}]\n{text}"
                merged_pages.append((synthetic_page, tagged_text))
                synthetic_page += 1
        if doc.text:
            tagged_text = f"[SOURCE: {doc.source_name} | PAGE: 1]\n{doc.text}"
            merged_pages.append((synthetic_page, tagged_text))
            synthetic_page += 1

    return LoadedDocument(
        text=None,
        pages=merged_pages if merged_pages else None,
        source_name="Multi-Documents",
        source_type="mixed",
    )


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(settings)


def main() -> None:
    settings.ensure_directories()
    st.set_page_config(page_title=settings.app_name, layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    _normalize_messages()

    if "selected_message_index" not in st.session_state:
        st.session_state.selected_message_index = None

    # Benchmark state
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None
    if "best_config" not in st.session_state:
        st.session_state.best_config = None
    if "loaded_docs_signature" not in st.session_state:
        st.session_state.loaded_docs_signature = None
    if "loaded_documents" not in st.session_state:
        st.session_state.loaded_documents = []
    if "combined_document" not in st.session_state:
        st.session_state.combined_document = None
    if "merged_doc_cache_version" not in st.session_state:
        st.session_state.merged_doc_cache_version = MERGED_DOC_CACHE_VERSION
    elif st.session_state.merged_doc_cache_version != MERGED_DOC_CACHE_VERSION:
        st.session_state.loaded_docs_signature = None
        st.session_state.loaded_documents = []
        st.session_state.combined_document = None
        st.session_state.merged_doc_cache_version = MERGED_DOC_CACHE_VERSION

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

    # --- SIDEBAR  ---
    with st.sidebar:
        st.subheader("System Status")
        st.write(f"Model: `{settings.ollama_model}`")

        st.divider()
        st.subheader("RAG Parameters Tuning")

        # Slider đồng bộ với session_state (để benchmark có thể cập nhật)
        chunk_size_key = "chunk_size_slider"
        chunk_overlap_key = "chunk_overlap_slider"
        
        # Streamlit config: Use only `st.session_state` values without `value=` parameter in slider
        # if the widget is tied to a session variable.
        if chunk_size_key not in st.session_state:
            st.session_state[chunk_size_key] = settings.chunk_size
        if chunk_overlap_key not in st.session_state:
            st.session_state[chunk_overlap_key] = settings.chunk_overlap
            
        new_chunk_size = st.slider(
            "Chunk Size",
            500, 2000,
            step=100,
            key=chunk_size_key
        )
        new_chunk_overlap = st.slider(
            "Chunk Overlap",
            50, 400,
            step=50,
            key=chunk_overlap_key
        )

        # Cập nhật settings
        settings.chunk_size = new_chunk_size
        settings.chunk_overlap = new_chunk_overlap

        if healthy:
            st.success(status_message)
        else:
            st.error(status_message)
        st.write(f"Embedding: `{settings.embedding_model_name}`")
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
                if st.button(preview, key=f"history_{pair['pair_index']}", use_container_width=True):
                    st.session_state.scroll_to = pair["user_index"]
            history_html.append("</div>")
            st.markdown("\n".join(history_html), unsafe_allow_html=True)
        else:
            st.info("No chat history yet.")

        st.divider()
        with st.popover("Clear History", use_container_width=True):
            st.warning("Bạn có chắc chắn muốn xóa toàn bộ lịch sử chat?")
            if st.button("Xác nhận xóa lịch sử", key="conf_clear_hist", type="primary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.selected_message_index = None
                st.query_params.clear()
                st.rerun()

        with st.popover("Clear Vector Store", use_container_width=True):
            st.warning("Hành động này sẽ xóa dữ liệu tài liệu hiện tại.")
            if st.button("Xác nhận xóa dữ liệu", key="conf_clear_vec", type="primary", use_container_width=True):
                get_pipeline.clear()
                st.success("Vector store cleared!")
                st.rerun()

    # --- KHU VỰC CHÍNH ---
    st.subheader("Trò chuyện với Tài liệu")
    
    uploaded_files = st.file_uploader("Upload PDF or DOCX documents", type=["pdf", "docx"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"Đã tải lên {len(uploaded_files)} tài liệu.")
    else:
        st.session_state.loaded_docs_signature = None
        st.session_state.loaded_documents = []
        st.session_state.combined_document = None

    # Render chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # --- Citation & Metadata Display ---
            if message.get("role") == "assistant" and message.get("chunks"):
                st.caption(f"Thời gian xử lý: {message.get('latency', 0):.2f}s")
                with st.expander("Context & Citations"):
                    for i, chunk in enumerate(message["chunks"], start=1):
                        st.markdown(f"**[{i}] Source**: `{chunk.get('source', 'Unknown')}` | **Page**: `{chunk.get('page', 'Unknown')}`")
                        st.text(_truncate_text(str(chunk.get("text", "")), 400))
                        st.divider()

    if question := st.chat_input("Ask a question about the uploaded documents"):
        start_time = time.time()
        
        current_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append(_create_message("user", question.strip()))
        
        with st.chat_message("user"):
            st.markdown(question)

        if not uploaded_files:
            error_msg = "Vui lòng upload ít nhất 1 file để bắt đầu!"
            st.session_state.messages.append(_create_message("assistant", error_msg))
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            st.rerun()

        target_docs: list[LoadedDocument] = st.session_state.loaded_documents
        upload_signature = _uploaded_files_signature(uploaded_files)

        if st.session_state.loaded_docs_signature != upload_signature:
            target_docs = []
            with st.status("Đang lưu và load các tài liệu...", expanded=True) as ingest_status:
                for idx, up_file in enumerate(uploaded_files, start=1):
                    target_path = settings.data_dir / up_file.name
                    target_path.write_bytes(up_file.getbuffer())

                    ingest_status.write(f"[{idx}/{len(uploaded_files)}] Saved file: {up_file.name}")
                    _append_ingestion_log(f"Saved file: {up_file.name}")

                    def on_progress(message: str, file_name: str = up_file.name) -> None:
                        progress_line = f"[{file_name}] {message}"
                        ingest_status.write(progress_line)
                        _append_ingestion_log(progress_line)

                    try:
                        document = load_document(target_path, progress_callback=on_progress)
                        target_docs.append(document)
                        ingest_status.write(f"Loaded successfully: {up_file.name}")
                        _append_ingestion_log(f"Loaded successfully: {up_file.name}")
                    except Exception as e:
                        ingest_status.write(f"Load failed: {up_file.name} -> {e}")
                        _append_ingestion_log(f"Load failed: {up_file.name} -> {e}")
                        st.error(f"Lỗi đọc file {up_file.name}: {e}")

                if target_docs:
                    ingest_status.update(label="Đã load xong tài liệu", state="complete")
                    st.session_state.loaded_documents = target_docs
                    st.session_state.loaded_docs_signature = upload_signature
                    st.session_state.combined_document = _merge_loaded_documents(target_docs)
                else:
                    ingest_status.update(label="Không thể load tài liệu", state="error")
                    st.session_state.loaded_documents = []
                    st.session_state.loaded_docs_signature = None
                    st.session_state.combined_document = None
        else:
            _append_ingestion_log("Reusing cached loaded documents for current upload set")

        combined_doc: LoadedDocument | None = st.session_state.combined_document
        if combined_doc:
            with st.chat_message("assistant"):
                with st.spinner("Đang suy nghĩ..."):
                    pipeline = get_pipeline()
                    try:
                        stream_placeholder = st.empty()
                        streamed_parts: list[str] = []

                        def on_token(delta: str) -> None:
                            streamed_parts.append(delta)
                            stream_placeholder.markdown("".join(streamed_parts) + "▌")

                        result = pipeline.answer_question(
                            document=combined_doc,
                            question=question.strip(),
                            conversation_history=current_history,
                            stream=True,
                            token_callback=on_token,
                        )
                        elapsed = round(time.time() - start_time, 2)
                        
                        ans_text = result.answer
                        confidence = "Unknown"
                        if "[CONFIDENCE: HIGH]" in ans_text:
                            confidence = "HIGH"
                            ans_text = ans_text.replace("[CONFIDENCE: HIGH]", "").strip()
                        elif "[CONFIDENCE: LOW]" in ans_text:
                            confidence = "LOW"
                            ans_text = ans_text.replace("[CONFIDENCE: LOW]", "").strip()
                            
                        stream_placeholder.markdown(ans_text)

                        logger.info("Answer generated in %.2fs | confidence=%s", elapsed, confidence)
                        _append_ingestion_log(f"Answer generated in {elapsed:.2f}s | confidence={confidence}")
                        
                        if confidence == "LOW":
                            st.warning("Độ tin cậy thấp: Mô hình cho rằng thông tin trong tài liệu có thể chưa đủ để trả lời chắc chắn.")
                        elif confidence == "HIGH":
                            st.success("Trả lời dựa sát trên Context hiển thị.")
                        
                        st.session_state.messages.append(
                            _create_message(
                                "assistant",
                                ans_text,
                                sources=result.sources,
                                chunks=result.chunks,
                                latency=elapsed
                            )
                        )
                    except Exception as exc:
                        error_text = f"Đã xảy ra lỗi hệ thống RAG: {exc}"
                        st.error(error_text)
                        st.session_state.messages.append(_create_message("assistant", error_text))
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