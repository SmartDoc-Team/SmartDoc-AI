from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smartdoc.benchmarking import parse_int_csv, resolve_default_document, run_chunk_benchmark
from smartdoc.config import settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark chunk_size/chunk_overlap impact on retrieval quality.")
    parser.add_argument("--document", type=Path, default=None, help="Path to benchmark document (PDF or DOCX)")
    parser.add_argument("--chunk-sizes", default="500,1000,1500,2000", help="Comma-separated chunk sizes")
    parser.add_argument("--chunk-overlaps", default="50,100,150,200", help="Comma-separated chunk overlaps")
    parser.add_argument("--sample-queries", type=int, default=40, help="Number of generated benchmark queries")
    parser.add_argument("--retrieval-k", type=int, default=settings.retrieval_k, help="Top-k retrieved chunks")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.50,
        help="Minimum overlap score for a retrieved chunk to be counted as relevant",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for query sampling")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "documentation" / "report" / "chunk_benchmark_results.md",
        help="Path to output Markdown report",
    )
    args = parser.parse_args()

    if args.sample_queries <= 0:
        raise ValueError("sample-queries must be greater than 0.")
    if args.retrieval_k <= 0:
        raise ValueError("retrieval-k must be greater than 0.")
    if not (0.0 <= args.relevance_threshold <= 1.0):
        raise ValueError("relevance-threshold must be between 0 and 1.")

    document_path = args.document if args.document is not None else resolve_default_document(settings.data_dir)

    chunk_sizes = parse_int_csv(args.chunk_sizes)
    chunk_overlaps = parse_int_csv(args.chunk_overlaps)

    benchmark_run = run_chunk_benchmark(
        settings=settings,
        document_path=document_path,
        chunk_sizes=chunk_sizes,
        chunk_overlaps=chunk_overlaps,
        sample_queries=args.sample_queries,
        retrieval_k=args.retrieval_k,
        relevance_threshold=args.relevance_threshold,
        seed=args.seed,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(benchmark_run.report_markdown, encoding="utf-8")

    print(f"Benchmark completed for {len(benchmark_run.results)} configurations.")
    print(
        "Best config: "
        f"chunk_size={benchmark_run.best_result.chunk_size}, "
        f"chunk_overlap={benchmark_run.best_result.chunk_overlap}"
    )
    print(f"Report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
