# Day 2 Ingestion Audit

## 1. Day 2 Objective

The objective of Day 2 was to build the document ingestion and chunking pipeline for the internal knowledge assistant.

The pipeline:

1. Reads the PDF documents.
2. Extracts text page by page.
3. Preserves source metadata.
4. Creates three independent chunk-size configurations.
5. Generates vector embeddings.
6. Stores the chunks and embeddings in persistent Chroma collections.

The three chunk-size configurations are:

* 250 tokens
* 500 tokens
* 1000 tokens

These configurations will be compared during Day 3 using Recall@5.

---

## 2. Source Corpus

The project uses a synthetic internal-company corpus because real internal documents were not available.

* Company: NovaTech Solutions
* Number of PDF documents: 30
* Total pages: 30
* Pages per document: 1
* Document format: PDF
* Document categories: HR, IT, Security, Finance, Operations, Legal, Communications, Administration
* Scanned/image-only documents: None
* Tables: None

All documents were successfully processed.

---

## 3. PDF Text Extraction

The extraction pipeline is implemented in:

`src/ingestion/extract_pdfs.py`

The pipeline processes every PDF page individually and extracts its text while preserving the following metadata:

* `source_file`
* `page`
* `section`
* `text`

Extraction output:

`data/processed/extracted_pages.jsonl`

Extraction error log:

`data/processed/extraction_errors.json`

### Extraction results

* PDF files found: 30
* Pages processed: 30
* Pages successfully extracted: 30
* Extraction issues: 0

Therefore:

**Extraction success rate = 30 / 30 = 100%**

---

## 4. Document Chunking

The chunking pipeline is implemented in:

`src/ingestion/chunk_documents.py`

Token-based chunking was used instead of character-based chunking so that the retrieval experiments can compare approximately equivalent token budgets.

Three configurations were generated:

|  Chunk size | Number of chunks | Source documents |
| ----------: | ---------------: | ---------------: |
|  250 tokens |               89 |               30 |
|  500 tokens |               59 |               30 |
| 1000 tokens |               30 |               30 |

Chunk files are stored separately:

```text
data/chunks/250/chunks.jsonl
data/chunks/500/chunks.jsonl
data/chunks/1000/chunks.jsonl
```

Each chunk preserves:

* `chunk_id`
* `source_file`
* `page`
* `section`
* `chunk_size`
* `chunk_index`
* `token_count`
* `text`

This metadata is required later so that retrieved answers can cite the exact source file and page.

---

## 5. Embedding Model Selection

The initial chunking work used the tokenizer associated with:

`sentence-transformers/all-MiniLM-L6-v2`

During validation, it was identified that the SentenceTransformer configuration has a maximum sequence length of 256 tokens.

This would make the 500-token and 1000-token configurations unsuitable for a fair embedding comparison because larger chunks could be truncated.

Therefore, the embedding model was changed to:

`BAAI/bge-m3`

The BGE-M3 configuration supports:

* Maximum sequence length: 8192 tokens
* Embedding dimension: 1024

This allows the 250-, 500-, and 1000-token chunks to be embedded without exceeding the embedding model's maximum sequence length.

The same embedding model is used for all three chunk-size configurations to keep the Day 3 comparison controlled.

---

## 6. Chroma Vector Database

The vector database is implemented using Chroma with persistent storage.

Storage location:

```text
storage/chroma
```

Three independent Chroma collections were created:

```text
rag_chunks_250
rag_chunks_500
rag_chunks_1000
```

### Vector database results

| Collection        | Chunk size | Records |
| ----------------- | ---------: | ------: |
| `rag_chunks_250`  |        250 |      89 |
| `rag_chunks_500`  |        500 |      59 |
| `rag_chunks_1000` |       1000 |      30 |

Total stored vectors:

**178**

All three collections were successfully created and verified after ingestion.

---

## 7. Metadata Verification

A Chroma record was manually inspected after ingestion.

Example metadata:

```text
{
    'page': 1,
    'source_file': 'ADMIN_Employee_ID_Policy.pdf',
    'chunk_size': 250,
    'chunk_index': 0,
    'token_count': 250,
    'section': 'NovaTech Solutions'
}
```

This confirms that the retrieval system retains the information required for source citation.

In particular, the system can identify:

* which document produced the chunk,
* which page contains the chunk,
* which chunk-size configuration was used,
* and the position of the chunk within the source page.

---

## 8. Day 2 Deliverables

The following Day 2 artifacts have been created:

```text
src/ingestion/extract_pdfs.py
src/ingestion/chunk_documents.py
src/ingestion/build_chroma.py

data/processed/extracted_pages.jsonl
data/processed/extraction_errors.json

data/chunks/250/chunks.jsonl
data/chunks/500/chunks.jsonl
data/chunks/1000/chunks.jsonl

storage/chroma/
```

---

## 9. Day 2 Verification Summary

| Requirement                   | Status   |
| ----------------------------- | -------- |
| PDF ingestion                 | Complete |
| Page-level text extraction    | Complete |
| Source filename preservation  | Complete |
| Page number preservation      | Complete |
| Section metadata preservation | Complete |
| Extraction error logging      | Complete |
| 250-token chunks              | Complete |
| 500-token chunks              | Complete |
| 1000-token chunks             | Complete |
| Embedding generation          | Complete |
| Persistent Chroma storage     | Complete |
| Metadata verification         | Complete |
| Recall@5 evaluation           | Day 3    |
| BM25 hybrid retrieval         | Day 3    |
| Cross-encoder reranking       | Day 3    |
| Grounded answer generation    | Day 4    |
| Streamlit UI                  | Day 5    |

---

## 10. Known Limitations

The current corpus consists of synthetic one-page PDF documents. It does not contain scanned/image-only documents or tables.

The 250-token and 500-token configurations can produce smaller final chunks when a document does not divide evenly into the requested chunk size. These tail chunks are retained rather than discarded so that no source text is lost.

The corpus is synthetic and this limitation must be disclosed in the final project README and evaluation report.

---

## 11. Day 2 Conclusion

Day 2 ingestion is complete.

All 30 source PDFs were successfully extracted, chunked into three configurations, embedded using the same BGE-M3 embedding model, and stored in separate persistent Chroma collections.

The resulting retrieval configurations are:

```text
250 tokens  → 89 chunks
500 tokens  → 59 chunks
1000 tokens → 30 chunks
```

These three configurations provide the baseline systems for the Day 3 retrieval experiment.

Day 3 will measure **Recall@5** for each configuration using the 25 answerable questions from the golden evaluation set. The best-performing chunk size will then be used as the baseline for separate BM25 hybrid and cross-encoder reranking experiments.
