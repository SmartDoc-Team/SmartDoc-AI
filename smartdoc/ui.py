from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smartdoc.config import settings
from smartdoc.document_loaders import load_document
from smartdoc.ollama_client import OllamaClient
from smartdoc.rag import RAGPipeline
from smartdoc.benchmarking import run_chunk_benchmark


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _normalize_messages() -> None:
    for index, message in enumerate(st.session_state.messages):
        if "id" not in message:
            message["id"] = f"msg_{index}"


def _create_message(role: str, content: str) -> dict[str, str]:
    return {"id": f"msg_{len(st.session_state.messages)}", "role": role, "content": content}


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

    # Benchmark state
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None
    if "best_config" not in st.session_state:
        st.session_state.best_config = None

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
        if chunk_size_key not in st.session_state:
            st.session_state[chunk_size_key] = settings.chunk_size
        if chunk_overlap_key not in st.session_state:
            st.session_state[chunk_overlap_key] = settings.chunk_overlap

        new_chunk_size = st.slider(
            "Chunk Size",
            500, 2000,
            value=st.session_state[chunk_size_key],
            step=100,
            key=chunk_size_key
        )
        new_chunk_overlap = st.slider(
            "Chunk Overlap",
            50, 400,
            value=st.session_state[chunk_overlap_key],
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
    uploaded_file = st.file_uploader("Upload a PDF or DOCX document", type=["pdf", "docx"])

    # --- HIỂN THỊ BENCHMARK THẬT ---
    if uploaded_file:
        with st.expander("Benchmark & Cấu hình tối ưu"):
            st.markdown(f"** Tài liệu:** {uploaded_file.name}")

            col1, col2 = st.columns(2)
            with col1:
                sample_queries = st.number_input(
                    "Số câu hỏi mẫu",
                    min_value=5, max_value=50, value=20, step=5,
                    key="bench_sample"
                )
                run_bench = st.button("Chạy benchmark (có thể mất vài phút)", type="primary", use_container_width=True)
            with col2:
                st.markdown("**Dải cấu hình thử nghiệm:**")
                st.caption("Chunk size: 500 → 2000 (bước 200)<br>Chunk overlap: 50 → 200 (bước 50)", unsafe_allow_html=True)

            if run_bench:
                with st.spinner("Đang chạy benchmark, vui lòng chờ..."):
                    file_bytes = uploaded_file.getvalue()
                    chunk_sizes = list(range(500, 2001, 200))   # 500,700,900,...,1900
                    chunk_overlaps = list(range(50, 201, 50))   # 50,100,150,200

                    @st.cache_data(show_spinner=False)
                    def _run_benchmark(file_bytes, filename, sizes, overlaps, sample_q, retrieval_k):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = Path(tmp.name)
                        try:
                            run = run_chunk_benchmark(
                                settings=settings,
                                document_path=tmp_path,
                                chunk_sizes=sizes,
                                chunk_overlaps=overlaps,
                                sample_queries=sample_q,
                                retrieval_k=retrieval_k,
                                relevance_threshold=0.5,
                                seed=42
                            )
                            return run
                        finally:
                            tmp_path.unlink()

                    run = _run_benchmark(
                        file_bytes, uploaded_file.name,
                        chunk_sizes, chunk_overlaps,
                        sample_queries, settings.retrieval_k
                    )
                    st.session_state.benchmark_results = run
                    st.session_state.best_config = (run.best_result.chunk_size, run.best_result.chunk_overlap)
                    st.success("Benchmark hoàn tất!")
                    st.rerun()

            # Hiển thị kết quả nếu có
            if st.session_state.benchmark_results:
                run = st.session_state.benchmark_results
                st.markdown("#### Kết quả so sánh (top 10 cấu hình tốt nhất)")
                df_data = []
                for res in run.results[:10]:
                    df_data.append({
                        "Chunk size": res.chunk_size,
                        "Overlap": res.chunk_overlap,
                        "Top-k Acc": f"{res.top_k_accuracy:.2%}",
                        "MRR": f"{res.mrr:.3f}",
                        "Mean Overlap": f"{res.mean_overlap:.2%}",
                        "Chunks": res.chunk_count,
                        "Time (s)": f"{res.elapsed_seconds:.1f}"
                    })
                st.dataframe(pd.DataFrame(df_data), use_container_width=True)

                st.markdown(
                    f"**Cấu hình được đề xuất:** `chunk_size = {run.best_result.chunk_size}`, "
                    f"`chunk_overlap = {run.best_result.chunk_overlap}`"
                )
                st.markdown(f"*Độ chính xác top-{run.retrieval_k}: {run.best_result.top_k_accuracy:.2%}*")

                if st.button("Áp dụng cấu hình này cho RAG", use_container_width=True):
                    settings.chunk_size = run.best_result.chunk_size
                    settings.chunk_overlap = run.best_result.chunk_overlap
                    st.session_state.chunk_size_slider = run.best_result.chunk_size
                    st.session_state.chunk_overlap_slider = run.best_result.chunk_overlap
                    st.cache_resource.clear()
                    st.success("Đã cập nhật chunk_size và chunk_overlap! Hãy thử hỏi lại tài liệu.")
                    st.rerun()
            else:
                st.info("Chưa có kết quả benchmark. Nhấn 'Chạy benchmark' để bắt đầu.")

    # Render chat messages
    for message in st.session_state.messages:
        st.markdown(f'<div id="{message["id"]}"></div>', unsafe_allow_html=True)
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.text_area("Ask a question about the uploaded document", height=120)

    if st.button("Run SmartDoc", type="primary", disabled=uploaded_file is None and not question.strip()):
        start_time = time.time()
        if not question.strip():
            st.warning("Please enter a question.")
            return

        current_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append(_create_message("user", question.strip()))

        if uploaded_file:
            target_path = settings.data_dir / uploaded_file.name
            target_path.write_bytes(uploaded_file.getbuffer())
            try:
                document = load_document(target_path)
                with st.spinner("Thinking..."):
                    pipeline = get_pipeline()
                    result = pipeline.answer_question(
                        document=document,
                        question=question.strip(),
                        conversation_history=current_history,
                    )
                    elapsed = round(time.time() - start_time, 2)
                    final_ans = f"{result.answer}\n\n*( Xử lý trong {elapsed}s | Size: {settings.chunk_size})*"
                    st.session_state.messages.append(_create_message("assistant", final_ans))
            except Exception as exc:
                st.exception(exc)
        else:
            st.session_state.messages.append(_create_message("assistant", "Vui lòng upload file để bắt đầu!"))

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