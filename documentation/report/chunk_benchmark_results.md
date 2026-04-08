# Chunking Benchmark Report

- Document: `sample_complex.docx`
- Sample queries: `40`
- Retrieval k: `3`
- Relevance threshold: `0.50`

## Ranking by retrieval quality

| chunk_size | chunk_overlap | top-k accuracy | MRR | mean overlap | chunk count | time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 150 | 0.875 | 0.812 | 0.925 | 6 | 0.80 |
| 1500 | 200 | 0.875 | 0.812 | 0.925 | 7 | 0.81 |
| 2000 | 200 | 0.875 | 0.812 | 0.925 | 6 | 0.81 |
| 2000 | 50 | 0.875 | 0.812 | 0.925 | 6 | 0.82 |
| 1500 | 100 | 0.875 | 0.812 | 0.925 | 7 | 0.83 |
| 2000 | 100 | 0.875 | 0.812 | 0.925 | 6 | 0.83 |
| 1500 | 50 | 0.875 | 0.812 | 0.925 | 7 | 0.84 |
| 1500 | 150 | 0.875 | 0.812 | 0.925 | 7 | 0.84 |
| 1000 | 150 | 0.875 | 0.812 | 0.925 | 9 | 0.90 |
| 1000 | 200 | 0.875 | 0.812 | 0.925 | 9 | 0.91 |
| 1000 | 50 | 0.875 | 0.812 | 0.925 | 8 | 0.92 |
| 1000 | 100 | 0.875 | 0.812 | 0.925 | 9 | 0.95 |
| 500 | 100 | 0.875 | 0.812 | 0.925 | 15 | 0.95 |
| 500 | 150 | 0.875 | 0.812 | 0.925 | 15 | 0.98 |
| 500 | 200 | 0.875 | 0.812 | 0.925 | 18 | 1.06 |
| 500 | 50 | 0.875 | 0.812 | 0.925 | 13 | 1.11 |

## Recommended configuration

Best config: `chunk_size=2000`, `chunk_overlap=150`.
Reason: this config has the strongest combined score across top-k accuracy, MRR, and mean overlap.
