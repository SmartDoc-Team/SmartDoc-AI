from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from smartdoc.config import Settings
from smartdoc.document_loaders import LoadedDocument
from smartdoc.ollama_client import OllamaClient


@dataclass(slots=True)
class RetrievalResult:
    answer: str
    chunks: list[dict[str, str | int]]
    sources: list[str]
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

    @staticmethod
    def _validate_chunking_params(chunk_size: int, chunk_overlap: int) -> tuple[int, int]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return chunk_size, chunk_overlap

    def _resolve_chunking_params(
        self,
        chunk_size: int | None,
        chunk_overlap: int | None,
    ) -> tuple[int, int]:
        resolved_chunk_size = self.settings.chunk_size if chunk_size is None else int(chunk_size)
        resolved_chunk_overlap = self.settings.chunk_overlap if chunk_overlap is None else int(chunk_overlap)
        return self._validate_chunking_params(resolved_chunk_size, resolved_chunk_overlap)

    def _create_splitter(self, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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

    def answer_question(
        self,
        document: LoadedDocument,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> RetrievalResult:
        chunks = self._split_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
        index = self._build_index(embeddings)
        top_chunks = self._retrieve_chunks(index, chunks, question)
        prompt = self._build_prompt(document, top_chunks, question, conversation_history)
        answer = self.ollama.generate(prompt)
        self._persist_index(document, embeddings, chunks)
        return RetrievalResult(answer=answer, contexts=top_chunks, chunk_count=len(chunks))

    def _split_text(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        resolved_chunk_size, resolved_chunk_overlap = self._resolve_chunking_params(chunk_size, chunk_overlap)
        splitter = self._create_splitter(resolved_chunk_size, resolved_chunk_overlap)
        chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
        if not chunks:
            raise ValueError("No content was available after splitting the document.")
        return chunks

    def _build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.asarray(embeddings, dtype=np.float32))
        return index

    def _retrieve_documents(
        self,
        index: faiss.IndexFlatIP,
        chunks: list[str],
        question: str,
    ) -> list[Document]:
        question_embedding = self.embedder.encode([question], normalize_embeddings=True)
        search_k = min(self.settings.retrieval_k, 4)
        _, indices = index.search(np.asarray(question_embedding, dtype=np.float32), search_k)
        return [documents[idx] for idx in indices[0] if 0 <= idx < len(documents)]

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
        context_blocks: list[str] = [doc.page_content for doc in contexts]
        joined_context = "\n\n---\n\n".join(context_blocks)
        clipped_context = joined_context[: self.settings.max_context_chars]
        history_block = ""
        if conversation_history:
            history_block = self._format_history(conversation_history)

        prompt = (
            "You are SmartDoc AI, a helpful assistant for document question answering.\n"
            "Answer in Vietnamese unless the user asks otherwise.\n"
            "Use only the provided context. The chunks are already sorted by relevance, so prioritize earlier chunks first.\n"
            "If the answer exists in the context, state it directly and quote the section terms faithfully.\n"
            "If the answer is missing, say clearly that the document does not contain it.\n"
            "Do NOT include sources or citations in your answer. They will be displayed separately.\n\n"
            f"Document: {document.source_name} ({document.source_type})\n\n"
            f"Context:\n{clipped_context}\n\n"
        )

        if history_block:
            prompt += f"Conversation history:\n{history_block}\n\n"

        prompt += f"Question: {question}\n\nAnswer:"
        return prompt

    def _persist_index(self, document: LoadedDocument, documents: list[Document], embeddings: np.ndarray) -> None:
        digest = hashlib.md5(document.source_name.encode("utf-8"), usedforsecurity=False).hexdigest()
        base_path = Path(self.settings.vector_store_dir) / digest
        index = self._build_index(embeddings)
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
