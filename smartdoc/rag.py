from __future__ import annotations

import hashlib
import json
import os
import re
import string
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

warnings.filterwarnings("ignore", message=r"Accessing `__path__` from .*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"Accessing `__path__` from .*", category=UserWarning)

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from smartdoc.config import Settings
from smartdoc.document_loaders import LoadedDocument
from smartdoc.ollama_client import OllamaClient

try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:  # noqa: BLE001
    pass


@dataclass(slots=True)
class RetrievalResult:
    answer: str
    chunks: list[dict[str, str | int]]
    sources: list[str]
    chunk_count: int


@dataclass(slots=True)
class CachedRetrievalAssets:
    chunk_docs: list[Document]
    embeddings: np.ndarray
    index: faiss.IndexFlatIP
    bm25_index: BM25Okapi


class RAGPipeline:
    HISTORY_MESSAGE_LIMIT = 20
    SOURCE_MARKER_PATTERN = re.compile(
        r"^\s*\[SOURCE:\s*(?P<source>[^\]|]+?)(?:\s*\|\s*PAGE:\s*(?P<page>\d+))?\]\s*",
        re.IGNORECASE,
    )
    FOLLOWUP_AMBIGUOUS_PATTERNS = [
        r"\b(n[oó]i tr[êe]n|[ơo] tr[êe]n|v[ừu]a n[oó]i|[đd]i[ềe]u [đd][óo]|c[aá]i n[aà]y|c[aá]i [đd][óo]|ph[ầa]n n[aà]y|ph[ầa]n [đd][óo])\b",
        r"^(c[oò]n|v[ậa]y|th[ếe]|n[oó]|h[oọ])\b",
    ]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = self._load_embedder()
        self.reranker = self._load_reranker()
        self.ollama = OllamaClient(
            settings.ollama_base_url,
            settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
            temperature=settings.ollama_temperature,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
        )
        self._splitter_chunk_size = settings.chunk_size
        self._splitter_chunk_overlap = settings.chunk_overlap
        self._retrieval_cache: dict[str, CachedRetrievalAssets] = {}
        self._persisted_cache_keys: set[str] = set()

    def _load_embedder(self) -> SentenceTransformer:
        try:
            return SentenceTransformer(self.settings.embedding_model_name)
        except Exception as exc:  # noqa: BLE001
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            try:
                return SentenceTransformer(self.settings.embedding_model_name, local_files_only=True)
            except Exception as offline_exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to load the embedding model. Ensure internet access is available for the first run, "
                    "or pre-download the model to the local Hugging Face cache."
                ) from offline_exc

    # def answer_question(
    #     self,
    #     document: LoadedDocument,
    #     question: str,
    #     conversation_history: list[dict[str, str]] | None = None,
    # ) -> RetrievalResult:
    #     chunks = self._split_text(document.text)
    #     embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
    #     index = self._build_index(embeddings)
    #     top_chunks = self._retrieve_chunks(index, embeddings, chunks, question)
    #     prompt = self._build_prompt(document, top_chunks, question, conversation_history)
    #     answer = self.ollama.generate(prompt)
    #     self._persist_index(document, embeddings, chunks)
    #     return RetrievalResult(answer=answer, contexts=top_chunks, chunk_count=len(chunks))

    def _load_reranker(self) -> CrossEncoder:
        model_name = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-TinyBERT-L-2-v2")
        try:
            return CrossEncoder(model_name)
        except Exception as exc:  # noqa: BLE001
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            try:
                return CrossEncoder(model_name, local_files_only=True)
            except Exception as offline_exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to load the cross-encoder model. Ensure internet access is available for the first run."
                ) from offline_exc

    def answer_question(
        self,
        document: LoadedDocument,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        stream: bool = False,
        token_callback: Callable[[str], None] | None = None,
    ) -> RetrievalResult:
        standalone_question = question
        if self.settings.rewrite_followup_enabled and self._should_rewrite_question(question, conversation_history):
            standalone_question = self._rewrite_question(question, conversation_history)

        cache_key, cached_assets, is_cache_miss = self._prepare_retrieval_assets(document)
        
        # BƯỚC 2: Truy xuất dựa trên câu hỏi đã được làm rõ ngữ cảnh
        top_chunks = self._retrieve_documents(
            cached_assets.index,
            cached_assets.bm25_index,
            cached_assets.chunk_docs,
            standalone_question,
        )
        
        # BƯỚC 3: Xây dựng Prompt cuối cùng với câu hỏi gốc và ngữ cảnh
        # Lưu ý: Ta vẫn truyền 'question' gốc vào prompt cuối để AI trả lời tự nhiên nhất
        prompt = self._build_prompt(document, top_chunks, question, conversation_history)
        
        if stream:
            answer_parts: list[str] = []
            for delta in self.ollama.generate_stream(prompt):
                answer_parts.append(delta)
                if token_callback:
                    token_callback(delta)
            answer = "".join(answer_parts).strip()
            if not answer:
                raise RuntimeError("Ollama returned an empty streamed response.")
        else:
            answer = self.ollama.generate(prompt)

        if is_cache_miss:
            self._persist_index(document, cache_key, cached_assets.index, cached_assets.chunk_docs)
        
        sources = self._extract_sources(top_chunks)
        contexts_meta = [{"text": doc.page_content, "source": doc.metadata.get("source"), "page": doc.metadata.get("page")} for doc in top_chunks]
        
        return RetrievalResult(answer=answer, chunks=contexts_meta, sources=sources, chunk_count=len(cached_assets.chunk_docs))

    def _should_rewrite_question(self, question: str, conversation_history: list[dict[str, str]] | None) -> bool:
        if not conversation_history:
            return False

        normalized = re.sub(r"\s+", " ", question.strip().lower())
        if not normalized:
            return False

        token_count = len(normalized.split())
        if token_count > self.settings.rewrite_followup_max_tokens:
            return False

        for pattern in self.FOLLOWUP_AMBIGUOUS_PATTERNS:
            if re.search(pattern, normalized):
                return True

        return False

    def _rewrite_question(self, question: str, conversation_history: list[dict[str, str]] | None) -> str:
        """
        Sử dụng LLM để biến đổi câu hỏi dựa trên ngữ cảnh lịch sử.
        Giúp xử lý các câu hỏi tiếp theo (follow-up questions).
        """
        if not conversation_history or len(conversation_history) == 0:
            return question

        history_text = self._format_history(conversation_history)
        
        # Prompt chuyên dụng để tái cấu trúc câu hỏi
        rewrite_prompt = (
            "Dựa trên lịch sử hội thoại dưới đây, hãy viết lại câu hỏi mới nhất của người dùng "
            "thành một câu hỏi độc lập, đầy đủ ý nghĩa mà không cần xem lại lịch sử.\n"
            "Chỉ trả về câu hỏi đã viết lại, không thêm lời dẫn giải.\n\n"
            f"Lịch sử hội thoại:\n{history_text}\n"
            f"Câu hỏi mới nhất: {question}\n\n"
            "Câu hỏi độc lập:"
        )
        
        standalone_question = self.ollama.generate(rewrite_prompt).strip()
        # Nếu LLM trả về kết quả rỗng hoặc lỗi, fallback về câu hỏi gốc
        return standalone_question if standalone_question else question

    def _split_document(self, document: LoadedDocument) -> list[Document]:
        docs: list[Document] = []
        if document.pages:
            for page_num, text in document.pages:
                source_name, source_page, cleaned_text = self._extract_source_marker(
                    text,
                    fallback_source=document.source_name,
                    fallback_page=page_num,
                )
                if not cleaned_text.strip():
                    continue
                docs.append(Document(page_content=cleaned_text, metadata={"source": source_name, "page": source_page}))
        if document.text:
            source_name, source_page, cleaned_text = self._extract_source_marker(
                document.text,
                fallback_source=document.source_name,
                fallback_page=1,
            )
            if cleaned_text.strip():
                docs.append(Document(page_content=cleaned_text, metadata={"source": source_name, "page": source_page}))
        if not docs:
            raise ValueError("Document has no text or pages.")

        splitter = self._get_splitter()
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("No content was available after splitting the document.")
            
        for i, chunk in enumerate(chunks):
            chunk.metadata["position"] = i
            
        return chunks

    def _extract_source_marker(self, text: str, fallback_source: str, fallback_page: int) -> tuple[str, int, str]:
        match = self.SOURCE_MARKER_PATTERN.match(text)
        if not match:
            return fallback_source, fallback_page, text

        source = (match.group("source") or fallback_source).strip() or fallback_source
        page_raw = match.group("page")
        page = int(page_raw) if page_raw and page_raw.isdigit() else fallback_page
        cleaned_text = text[match.end():].lstrip()
        return source, page, cleaned_text

    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        if (
            self._splitter_chunk_size == self.settings.chunk_size
            and self._splitter_chunk_overlap == self.settings.chunk_overlap
        ):
            return self.splitter

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
        )
        self._splitter_chunk_size = self.settings.chunk_size
        self._splitter_chunk_overlap = self.settings.chunk_overlap
        return self.splitter

    def _document_cache_key(self, document: LoadedDocument) -> str:
        hasher = hashlib.md5(usedforsecurity=False)
        hasher.update(document.source_name.encode("utf-8", errors="ignore"))
        hasher.update(document.source_type.encode("utf-8", errors="ignore"))
        hasher.update(str(self.settings.chunk_size).encode("utf-8"))
        hasher.update(str(self.settings.chunk_overlap).encode("utf-8"))
        hasher.update(self.settings.embedding_model_name.encode("utf-8", errors="ignore"))

        if document.pages:
            for page_num, text in document.pages:
                hasher.update(str(page_num).encode("utf-8"))
                hasher.update(text.encode("utf-8", errors="ignore"))
        elif document.text:
            hasher.update(document.text.encode("utf-8", errors="ignore"))

        return hasher.hexdigest()

    def _prepare_retrieval_assets(self, document: LoadedDocument) -> tuple[str, CachedRetrievalAssets, bool]:
        cache_key = self._document_cache_key(document)
        cached_assets = self._retrieval_cache.get(cache_key)
        if cached_assets is not None:
            return cache_key, cached_assets, False

        chunk_docs = self._split_document(document)
        chunk_texts = [doc.page_content for doc in chunk_docs]
        embeddings = self.embedder.encode(chunk_texts, normalize_embeddings=True)
        index = self._build_index(embeddings)
        bm25_index = self._build_bm25(chunk_docs)

        cached_assets = CachedRetrievalAssets(
            chunk_docs=chunk_docs,
            embeddings=embeddings,
            index=index,
            bm25_index=bm25_index,
        )
        self._retrieval_cache[cache_key] = cached_assets
        return cache_key, cached_assets, True

    def _tokenize_text(self, text: str) -> list[str]:
        text = text.lower()
        return text.translate(str.maketrans("", "", string.punctuation)).split()

    def _build_bm25(self, documents: list[Document]) -> BM25Okapi:
        tokenized_corpus = [self._tokenize_text(doc.page_content) for doc in documents]
        return BM25Okapi(tokenized_corpus)

    def _build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.asarray(embeddings, dtype=np.float32))
        return index

    def _retrieve_documents(
        self,
        index: faiss.IndexFlatIP,
        bm25_index: BM25Okapi,
        documents: list[Document],
        question: str,
    ) -> list[Document]:
        corpus_sources = {str(doc.metadata.get("source", "unknown")) for doc in documents}
        source_count = max(1, len(corpus_sources))
        adaptive_k = max(self.settings.retrieval_k, min(8, source_count * 2))
        final_k = min(adaptive_k, len(documents))
        if final_k == 0:
            return []
            
        search_k = min(max(final_k * 4, 12), len(documents))  # Trích xuất nhiều hơn cho re-ranking

        # 1. Vector Search (Faiss)
        question_embedding = self.embedder.encode([question], normalize_embeddings=True)
        _, indices = index.search(np.asarray(question_embedding, dtype=np.float32), search_k)
        vector_results = indices[0].tolist()

        # 2. Keyword Search (BM25)
        tokenized_query = self._tokenize_text(question)
        bm25_scores = bm25_index.get_scores(tokenized_query)
        bm25_results = np.argsort(bm25_scores)[::-1][:search_k].tolist()

        # 3. Hybrid Merge with RRF (Reciprocal Rank Fusion)
        rrf_k = 60
        rrf_scores: dict[int, float] = {}

        for rank, doc_idx in enumerate(vector_results):
            if doc_idx != -1 and doc_idx < len(documents):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
                
        for rank, doc_idx in enumerate(bm25_results):
            if bm25_scores[doc_idx] > 0 and doc_idx < len(documents):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
        hybrid_top_indices = sorted_indices[:search_k]

        candidate_indices: list[int] = []
        seen_candidate_indices: set[int] = set()
        for idx in hybrid_top_indices:
            if idx in seen_candidate_indices:
                continue
            candidate_indices.append(idx)
            seen_candidate_indices.add(idx)

        source_to_indices: dict[str, list[int]] = {}
        for idx, doc in enumerate(documents):
            source = str(doc.metadata.get("source", "unknown"))
            source_to_indices.setdefault(source, []).append(idx)

        covered_sources = {str(documents[idx].metadata.get("source", "unknown")) for idx in candidate_indices}
        missing_sources = set(source_to_indices.keys()) - covered_sources

        for source in missing_sources:
            source_indices = source_to_indices[source]
            best_idx = max(source_indices, key=lambda i: float(bm25_scores[i]))
            if best_idx not in seen_candidate_indices:
                candidate_indices.append(best_idx)
                seen_candidate_indices.add(best_idx)

        hybrid_candidates = [documents[idx] for idx in candidate_indices]

        # 4. Cross-Encoder Re-ranking
        if not hybrid_candidates:
            return []
            
        cross_inputs = [[question, doc.page_content] for doc in hybrid_candidates]
        cross_scores = self.reranker.predict(cross_inputs)
        
        # Sort hybrid candidates by cross-encoder score
        reranked_pairs = sorted(zip(hybrid_candidates, cross_scores), key=lambda x: x[1], reverse=True)

        ranked_docs = [doc for doc, _ in reranked_pairs]
        unique_sources = {str(doc.metadata.get("source", "unknown")) for doc in ranked_docs}

        if len(unique_sources) <= 1:
            return ranked_docs[:final_k]

        return self._select_docs_diverse_by_source(ranked_docs, final_k)

    def _select_docs_diverse_by_source(self, ranked_docs: list[Document], final_k: int) -> list[Document]:
        source_order: list[str] = []
        source_buckets: dict[str, list[Document]] = {}

        for doc in ranked_docs:
            source = str(doc.metadata.get("source", "unknown"))
            if source not in source_buckets:
                source_buckets[source] = []
                source_order.append(source)
            source_buckets[source].append(doc)

        selected_docs: list[Document] = []
        exhausted_sources = 0

        while len(selected_docs) < final_k and exhausted_sources < len(source_order):
            exhausted_sources = 0
            for source in source_order:
                bucket = source_buckets[source]
                if not bucket:
                    exhausted_sources += 1
                    continue
                selected_docs.append(bucket.pop(0))
                if len(selected_docs) >= final_k:
                    break

        if len(selected_docs) < final_k:
            seen_ids = {id(doc) for doc in selected_docs}
            for doc in ranked_docs:
                if len(selected_docs) >= final_k:
                    break
                if id(doc) in seen_ids:
                    continue
                selected_docs.append(doc)
                seen_ids.add(id(doc))

        return selected_docs

    def _extract_sources(self, documents: list[Document]) -> list[str]:
        sources: list[str] = []
        for document in documents:
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page", 1)
            entry = f"{source} - page {page}"
            if entry not in sources:
                sources.append(entry)
        return sources

    def _format_history(self, conversation_history: list[dict[str, str]]) -> str:
        history = [msg for msg in conversation_history if msg.get("content")]
        if len(history) > self.HISTORY_MESSAGE_LIMIT:
            history = history[-self.HISTORY_MESSAGE_LIMIT:]

        lines: list[str] = []
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content')}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        document: LoadedDocument,
        contexts: list[Document],
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        context_blocks: list[str] = []
        context_sources: list[str] = []
        for doc in contexts:
            source = str(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            context_sources.append(source)
            context_blocks.append(f"[SOURCE: {source} | PAGE: {page}]\n{doc.page_content}")

        source_list = ", ".join(sorted(set(context_sources))) if context_sources else "unknown"
        joined_context = "\n\n---\n\n".join(context_blocks)
        clipped_context = joined_context[: self.settings.max_context_chars]
        
        # Xây dựng block lịch sử để AI tham chiếu ngược lại các câu trả lời đã có
        history_block = ""
        if conversation_history:
            history_block = self._format_history(conversation_history)

        prompt = (
            "You are SmartDoc AI, a helpful assistant for document question answering.\n"
            "Answer in Vietnamese unless the user asks otherwise.\n"
            "Use only the provided context and refer to the conversation history if the user mentions previous points.\n"
            "Each context block starts with [SOURCE: ... | PAGE: ...]. Use these source labels to compare documents precisely when the user asks for differences/similarities.\n"
            "If the context does NOT contain enough information to vividly answer the question, clearly state that the provided document does not contain the answer, and do not make up information.\n"
            "Always include '[CONFIDENCE: HIGH]' at the end of your answer if you are reasonably sure the context supports your answer, and include '[CONFIDENCE: LOW]' if you are unsure or the context lacks information.\n\n"
            f"Document: {document.source_name}\n"
            f"Available sources in retrieved context: {source_list}\n"
            f"Context:\n{clipped_context}\n\n"
        )

        if history_block:
            prompt += f"Conversation history (use this for context but prioritize Document content):\n{history_block}\n\n"

        prompt += f"Current Question: {question}\n\nAnswer:"
        return prompt
        
    def _persist_index(self, document: LoadedDocument, cache_key: str, index: faiss.IndexFlatIP, documents: list[Document]) -> None:
        if cache_key in self._persisted_cache_keys:
            return

        base_path = Path(self.settings.vector_store_dir) / cache_key
        faiss.write_index(index, str(base_path.with_suffix(".faiss")))

        metadata = [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "position": doc.metadata.get("position"),
                "text": doc.page_content,
            }
            for doc in documents
        ]
        base_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        base_path.with_suffix(".txt").write_text("\n\n-----\n\n".join([doc.page_content for doc in documents]), encoding="utf-8")
        self._persisted_cache_keys.add(cache_key)
