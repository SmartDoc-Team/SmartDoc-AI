from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from smartdoc.config import Settings
from smartdoc.document_loaders import load_document
from smartdoc.rag import RAGPipeline


WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
#test: py scripts\benchmark_chunking.py --document data\sample_complex.docx --chunk-sizes 500,1000,1500,2000 --chunk-overlaps 50,100,150,200 --sample-queries 40   //any file in data folder

@dataclass(slots=True)
class BenchmarkQuery:
    question: str
    reference_text: str


@dataclass(slots=True)
class BenchmarkResult:
    chunk_size: int
    chunk_overlap: int
    top_k_accuracy: float
    mrr: float
    mean_overlap: float
    chunk_count: int
    elapsed_seconds: float


@dataclass(slots=True)
class BenchmarkRun:
    document_name: str
    sample_queries: int
    retrieval_k: int
    relevance_threshold: float
    results: list[BenchmarkResult]
    best_result: BenchmarkResult
    report_markdown: str


def parse_int_csv(raw_value: str) -> list[int]:
    values: list[int] = []
    for part in raw_value.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        values.append(int(cleaned))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def resolve_default_document(data_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".docx", ".pdf"}
    )

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No input document found in the data directory. "
        "Provide --document <path-to-pdf-or-docx> or add a .docx/.pdf file to data/."
    )


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text)}


def overlap_score(reference_text: str, candidate_text: str) -> float:
    reference_tokens = tokenize(reference_text)
    if not reference_tokens:
        return 0.0
    candidate_tokens = tokenize(candidate_text)
    if not candidate_tokens:
        return 0.0
    return len(reference_tokens & candidate_tokens) / len(reference_tokens)


def _sentence_candidates(text: str) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    candidates = [part.strip() for part in raw_parts if 40 <= len(part.strip()) <= 320]

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _fallback_candidates(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []

    fallback: list[str] = []
    window_size = 24
    stride = 12
    for start in range(0, max(1, len(words) - window_size + 1), stride):
        snippet = " ".join(words[start : start + window_size]).strip()
        if len(snippet) >= 40:
            fallback.append(snippet)
    return fallback


def build_benchmark_queries(text: str, sample_queries: int, seed: int) -> list[BenchmarkQuery]:
    candidates = _sentence_candidates(text)
    if not candidates:
        candidates = _fallback_candidates(text)
    if not candidates:
        raise ValueError("Could not generate benchmark queries from the document text.")

    rng = random.Random(seed)
    if len(candidates) >= sample_queries:
        selected = rng.sample(candidates, sample_queries)
    else:
        selected = candidates.copy()
        while len(selected) < sample_queries:
            selected.append(candidates[len(selected) % len(candidates)])

    queries: list[BenchmarkQuery] = []
    for sentence in selected:
        prompt_snippet = " ".join(sentence.split()[:18])
        question = f"Noi dung nao de cap den: '{prompt_snippet}'?"
        queries.append(BenchmarkQuery(question=question, reference_text=sentence))

    return queries


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype=np.float32))
    return index


def _document_to_text(document_text: str | None, document_pages: list[tuple[int, str]] | None) -> str:
    if document_text:
        return document_text
    if document_pages:
        return "\n\n".join(text for _, text in document_pages if text)
    return ""


def _split_for_benchmark(document_text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_documents([Document(page_content=document_text, metadata={"source": "benchmark", "page": 1})])
    return [doc.page_content for doc in docs if doc.page_content]


def evaluate_configuration(
    pipeline: RAGPipeline,
    document_text: str,
    queries: list[BenchmarkQuery],
    chunk_size: int,
    chunk_overlap: int,
    retrieval_k: int,
    relevance_threshold: float,
) -> BenchmarkResult:
    started_at = time.perf_counter()

    chunks = _split_for_benchmark(document_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No chunks generated for the benchmark configuration.")
    embeddings = pipeline.embedder.encode(chunks, normalize_embeddings=True)
    index = build_index(embeddings)

    hit_count = 0
    reciprocal_rank_sum = 0.0
    overlap_sum = 0.0

    for query in queries:
        query_embedding = pipeline.embedder.encode([query.question], normalize_embeddings=True)
        _, indices = index.search(np.asarray(query_embedding, dtype=np.float32), retrieval_k)

        first_relevant_rank: int | None = None
        best_overlap = 0.0

        for rank, idx in enumerate(indices[0], start=1):
            if idx < 0 or idx >= len(chunks):
                continue
            score = overlap_score(query.reference_text, chunks[idx])
            if score > best_overlap:
                best_overlap = score
            if first_relevant_rank is None and score >= relevance_threshold:
                first_relevant_rank = rank

        overlap_sum += best_overlap
        if first_relevant_rank is not None:
            hit_count += 1
            reciprocal_rank_sum += 1.0 / first_relevant_rank

    query_count = len(queries)
    elapsed_seconds = time.perf_counter() - started_at

    return BenchmarkResult(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k_accuracy=hit_count / query_count,
        mrr=reciprocal_rank_sum / query_count,
        mean_overlap=overlap_sum / query_count,
        chunk_count=len(chunks),
        elapsed_seconds=elapsed_seconds,
    )


def sort_results(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    return sorted(
        results,
        key=lambda item: (
            item.top_k_accuracy,
            item.mrr,
            item.mean_overlap,
            -item.elapsed_seconds,
            -item.chunk_count,
        ),
        reverse=True,
    )


def format_report_markdown(
    document_name: str,
    sample_queries: int,
    retrieval_k: int,
    relevance_threshold: float,
    results: list[BenchmarkResult],
) -> str:
    sorted_results = sort_results(results)
    best = sorted_results[0]

    lines: list[str] = [
        "# Chunking Benchmark Report",
        "",
        f"- Document: `{document_name}`",
        f"- Sample queries: `{sample_queries}`",
        f"- Retrieval k: `{retrieval_k}`",
        f"- Relevance threshold: `{relevance_threshold:.2f}`",
        "",
        "## Ranking by retrieval quality",
        "",
        "| chunk_size | chunk_overlap | top-k accuracy | MRR | mean overlap | chunk count | time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in sorted_results:
        lines.append(
            "| "
            f"{result.chunk_size} | "
            f"{result.chunk_overlap} | "
            f"{result.top_k_accuracy:.3f} | "
            f"{result.mrr:.3f} | "
            f"{result.mean_overlap:.3f} | "
            f"{result.chunk_count} | "
            f"{result.elapsed_seconds:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Recommended configuration",
            "",
            f"Best config: `chunk_size={best.chunk_size}`, `chunk_overlap={best.chunk_overlap}`.",
            (
                "Reason: this config has the strongest combined score across "
                "top-k accuracy, MRR, and mean overlap."
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def run_chunk_benchmark(
    settings: Settings,
    document_path: Path,
    chunk_sizes: list[int],
    chunk_overlaps: list[int],
    sample_queries: int,
    retrieval_k: int,
    relevance_threshold: float,
    seed: int,
) -> BenchmarkRun:
    document = load_document(document_path)
    pipeline = RAGPipeline(settings)
    full_text = _document_to_text(document.text, document.pages)
    if not full_text.strip():
        raise ValueError("Could not build benchmark corpus from the selected document.")
    queries = build_benchmark_queries(full_text, sample_queries, seed)

    results: list[BenchmarkResult] = []
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            if chunk_overlap >= chunk_size:
                continue
            result = evaluate_configuration(
                pipeline=pipeline,
                document_text=full_text,
                queries=queries,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                retrieval_k=retrieval_k,
                relevance_threshold=relevance_threshold,
            )
            results.append(result)

    if not results:
        raise ValueError("No valid (chunk_size, chunk_overlap) combinations were benchmarked.")

    sorted_results = sort_results(results)
    best_result = sorted_results[0]
    report_markdown = format_report_markdown(
        document_name=document.source_name,
        sample_queries=len(queries),
        retrieval_k=retrieval_k,
        relevance_threshold=relevance_threshold,
        results=sorted_results,
    )

    return BenchmarkRun(
        document_name=document.source_name,
        sample_queries=len(queries),
        retrieval_k=retrieval_k,
        relevance_threshold=relevance_threshold,
        results=sorted_results,
        best_result=best_result,
        report_markdown=report_markdown,
    )
