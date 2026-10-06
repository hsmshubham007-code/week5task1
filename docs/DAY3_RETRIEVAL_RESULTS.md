# Day 3 Retrieval & Comparison Results

## Objective

Measure retrieval Recall@5 across three chunk sizes and then measure the effect of BM25 hybrid retrieval and cross-encoder reranking separately.

The evaluation uses the 25 answerable questions from the Day 1 golden set.

The five unanswerable questions are excluded because they have no valid source document and are reserved for Day 4 grounding/refusal evaluation.

---

## Evaluation Setup

- Corpus: 30 NovaTech Solutions synthetic internal-policy PDFs
- Answerable evaluation questions: 25
- Metric: Recall@5
- Embedding model: `BAAI/bge-m3`
- Vector database: Chroma
- Top K: 5
- Baseline collections:
  - `rag_chunks_250`
  - `rag_chunks_500`
  - `rag_chunks_1000`
- BM25 candidate pool: top 20
- Hybrid method: Reciprocal Rank Fusion (RRF)
- Cross-encoder candidate pool: top 20
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## Results

| Configuration | Questions | Hits@5 | Recall@5 | Explanation |
|---|---:|---:|---:|---|
| Vector — 250 tokens | 25 | 25 | 100% | Baseline vector retrieval successfully found the gold source/page for every answerable question. |
| Vector — 500 tokens | 25 | 25 | 100% | Larger chunks produced the same Recall@5 as the 250-token baseline. |
| Vector — 1000 tokens | 25 | 25 | 100% | The largest chunk configuration also retrieved every gold source within the top five. |
| 250 + BM25 hybrid | 25 | 25 | 100% | BM25 + vector fusion produced no Recall@5 improvement over vector-only retrieval. |
| 250 + cross-encoder reranker | 25 | 25 | 100% | Reranking the top 20 vector candidates produced no Recall@5 improvement. |

---

## Chunk Size Selection

All three chunk sizes achieved identical Recall@5:

- 250 tokens: 100%
- 500 tokens: 100%
- 1000 tokens: 100%

Therefore, Recall@5 does not provide a quality-based winner on this corpus.

The project retains **250-token chunks as the baseline configuration** as a deterministic tie-break for subsequent generation experiments.

This is not claimed to be superior in retrieval quality.

---

## BM25 Hybrid Result

The 250-token vector baseline achieved:

**Recall@5 = 100%**

Adding BM25 using Reciprocal Rank Fusion also achieved:

**Recall@5 = 100%**

Therefore, BM25 did not produce a measurable Recall@5 improvement on this evaluation set.

---

## Cross-Encoder Reranking Result

The 250-token vector baseline achieved:

**Recall@5 = 100%**

Adding the `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker also achieved:

**Recall@5 = 100%**

Therefore, cross-encoder reranking did not produce a measurable Recall@5 improvement on this evaluation set.

---

## Interpretation

The retrieval task is relatively easy for the current synthetic corpus. Each document contains a single page of policy information, and the golden questions directly correspond to individual policy documents.

As a result, the BGE-M3 vector retrieval baseline already achieves perfect Recall@5.

Because the baseline has reached 100%, BM25 and reranking cannot improve the Recall@5 metric further on this evaluation set.

The results should therefore be interpreted as evidence that the baseline is sufficient for this corpus, rather than evidence that BM25 or reranking are generally unnecessary.

---

## Limitations

The corpus is synthetic and consists of 30 one-page documents without tables or scanned/image-only documents.

This makes the retrieval problem simpler than a realistic internal knowledge base containing longer documents, repeated terminology, tables, scanned PDFs, and ambiguous questions.

The 25 answerable-question evaluation set is also relatively small.

Therefore, the 100% Recall@5 result should not be generalized to larger or more difficult real-world corpora.

---

## Day 3 Acceptance Status

- [x] Recall@5 measured for 250-token chunks
- [x] Recall@5 measured for 500-token chunks
- [x] Recall@5 measured for 1000-token chunks
- [x] Chunk-size comparison documented
- [x] BM25 hybrid measured separately
- [x] Cross-encoder reranking measured separately
- [x] Results saved as CSV files
- [x] Limitations documented