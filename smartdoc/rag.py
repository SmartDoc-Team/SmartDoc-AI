from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from smartdoc.config import Settings
from smartdoc.document_loaders import LoadedDocument
from smartdoc.ollama_client import OllamaClient


@dataclass(slots=True)
class RetrievalResult:
    answer: str
    contexts: list[str]
    chunk_count: int


class RAGPipeline:
    HISTORY_MESSAGE_LIMIT = 20

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = self._load_embedder()
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

    def answer_question(
        self,
        document: LoadedDocument,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> RetrievalResult:
        # BƯỚC 1: Xử lý follow-up question (Nối ngữ cảnh)
        # Biến câu hỏi "Nó là gì?" thành "SmartDoc AI là gì?"
        standalone_question = self._rewrite_question(question, conversation_history)
        
        chunks = self._split_text(document.text)
        embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
        index = self._build_index(embeddings)
        
        # BƯỚC 2: Truy xuất dựa trên câu hỏi đã được làm rõ ngữ cảnh
        top_chunks = self._retrieve_chunks(index, embeddings, chunks, standalone_question)
        
        # BƯỚC 3: Xây dựng Prompt cuối cùng với câu hỏi gốc và ngữ cảnh
        # Lưu ý: Ta vẫn truyền 'question' gốc vào prompt cuối để AI trả lời tự nhiên nhất
        prompt = self._build_prompt(document, top_chunks, question, conversation_history)
        
        answer = self.ollama.generate(prompt)
        self._persist_index(document, embeddings, chunks)
        
        return RetrievalResult(answer=answer, contexts=top_chunks, chunk_count=len(chunks))

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

    def _split_text(self, text: str) -> list[str]:
        chunks = [chunk.strip() for chunk in self.splitter.split_text(text) if chunk.strip()]
        if not chunks:
            raise ValueError("No content was available after splitting the document.")
        return chunks

    def _build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.asarray(embeddings, dtype=np.float32))
        return index

    def _retrieve_chunks(
        self,
        index: faiss.IndexFlatIP,
        embeddings: np.ndarray,
        chunks: list[str],
        question: str,
    ) -> list[str]:
        question_embedding = self.embedder.encode([question], normalize_embeddings=True)
        _, indices = index.search(np.asarray(question_embedding, dtype=np.float32), self.settings.retrieval_k)
        return [chunks[idx] for idx in indices[0] if 0 <= idx < len(chunks)]

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
        contexts: list[str],
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        joined_context = "\n\n---\n\n".join(contexts)
        clipped_context = joined_context[: self.settings.max_context_chars]
        
        # Xây dựng block lịch sử để AI tham chiếu ngược lại các câu trả lời đã có
        history_block = ""
        if conversation_history:
            history_block = self._format_history(conversation_history)

        prompt = (
            "You are SmartDoc AI, a helpful assistant for document question answering.\n"
            "Answer in Vietnamese unless the user asks otherwise.\n"
            "Use only the provided context and refer to the conversation history if the user mentions previous points.\n"
            "If the answer exists in the context, state it directly.\n"
            "If the answer is missing, say clearly that the document does not contain it.\n\n"
            f"Document: {document.source_name}\n"
            f"Context:\n{clipped_context}\n\n"
        )

        if history_block:
            prompt += f"Conversation history (use this for context but prioritize Document content):\n{history_block}\n\n"

        prompt += f"Current Question: {question}\n\nAnswer:"
        return prompt
        
    def _persist_index(self, document: LoadedDocument, embeddings: np.ndarray, chunks: list[str]) -> None:
        digest = hashlib.md5(document.source_name.encode("utf-8"), usedforsecurity=False).hexdigest()
        base_path = Path(self.settings.vector_store_dir) / digest
        faiss.write_index(self._build_index(embeddings), str(base_path.with_suffix(".faiss")))
        base_path.with_suffix(".txt").write_text("\n\n-----\n\n".join(chunks), encoding="utf-8")
