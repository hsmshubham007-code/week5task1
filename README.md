# NovaTech Solutions RAG Evaluation System

A retrieval-augmented generation (RAG) system for answering questions from a synthetic internal-company policy corpus.

The project was developed as a five-day RAG evaluation workflow covering corpus auditing, retrieval comparison, grounded generation, evaluation, and handover.

---

## 1. Project Overview

The system answers questions using only the retrieved NovaTech Solutions policy documents.

The pipeline:

1. Loads and extracts text from PDF policy documents.
2. Splits documents into multiple chunk sizes.
3. Creates vector collections using BGE-M3 embeddings.
4. Evaluates retrieval using Recall@5.
5. Compares vector retrieval, BM25 hybrid retrieval, and cross-encoder reranking.
6. Uses the selected 250-token retrieval configuration for grounded question answering.
7. Selects supporting evidence before producing an answer.
8. Refuses to answer when sufficient evidence is unavailable.
9. Provides source-file and page citations.
10. Provides a Streamlit chat interface.
11. Provides a one-command 30-question evaluation.

---

## 2. Corpus

The evaluation corpus contains:

* 30 synthetic internal policy PDFs
* 30 pages total
* 1 page per document
* NovaTech Solutions policy documents
* Multiple policy areas including HR, IT, Security, Finance, Operations, Legal, Communications, and Administration

The corpus is synthetic and was created specifically for the Week 5 RAG evaluation assignment.

---

## 3. Golden Evaluation Set

The golden set was created before the retrieval pipeline code.

It contains exactly:

| Category     | Questions |
| ------------ | --------: |
| Answerable   |        25 |
| Unanswerable |         5 |
| Total        |        30 |

For each answerable question, the golden set records the expected answer and exact source file/page.

The five unanswerable questions are used to verify that the system refuses to answer instead of generating unsupported information.

Golden set:

```text
data/golden_set/golden_set.csv
```

---

## 4. Architecture

![RAG System Architecture](docs/Architecture.png)


```text
PDF Policy Documents
        |
        v
   PDF Extraction
        |
        v
  Text + Metadata
        |
        v
   Chunking
  /     |      \
250    500    1000 tokens
 |      |       |
 v      v       v
Chroma Chroma  Chroma
 |      |       |
 +------v-------+
        |
        v
 Retrieval Evaluation
        |
        +--> Vector Retrieval
        |
        +--> BM25 Hybrid
        |
        +--> Cross-Encoder Reranker
        |
        v
 Selected 250-token Retrieval
        |
        v
 Evidence Selection
        |
   +----+----+
   |         |
Enough?    Not enough
   |         |
   v         v
Answer     Refuse
   |
   v
Source + Page Citation
   |
   v
Streamlit UI
```

---

## 5. Embedding Model

The retrieval pipeline uses:

```text
BAAI/bge-m3
```

BGE-M3 was selected so that the 250-, 500-, and 1000-token chunk configurations could be compared using the same embedding model without the shorter context limitation of the original MiniLM configuration.

The same embedding model is used across the evaluated chunk sizes.

---

## 6. Chunking Evaluation

Three chunk sizes were evaluated:

* 250 tokens
* 500 tokens
* 1000 tokens

The selected 250-token configuration was then used for the grounded QA system.

### Chunk counts

| Chunk size  | Chunks |
| ----------- | -----: |
| 250 tokens  |     89 |
| 500 tokens  |     59 |
| 1000 tokens |     30 |

---

## 7. Retrieval Evaluation

Day 3 evaluated the 25 answerable questions using Recall@5.

| Configuration                      |     Recall@5 | Result         |
| ---------------------------------- | -----------: | -------------- |
| 250-token vector retrieval         | 25/25 (100%) | Selected       |
| 500-token vector retrieval         | 25/25 (100%) | Tie            |
| 1000-token vector retrieval        | 25/25 (100%) | Tie            |
| 250-token + BM25 hybrid            | 25/25 (100%) | No improvement |
| 250-token + cross-encoder reranker | 25/25 (100%) | No improvement |

All three chunk sizes achieved the same measured Recall@5.

The 250-token configuration was retained as a deterministic tie-breaker rather than claiming that it had higher recall.

BM25 and cross-encoder reranking also produced no Recall@5 improvement on this evaluation set.

Detailed Day 3 results are available in:

```text
docs/DAY3_RETRIEVAL_RESULTS.md
```

---

## 8. Grounded Generation

The final QA system is evidence-first.

For each question it:

1. Retrieves the top five relevant chunks.
2. Checks whether sufficient evidence exists.
3. Selects supporting evidence.
4. Produces a grounded answer when evidence is sufficient.
5. Adds a source-file/page citation.
6. Refuses when sufficient evidence is unavailable.

The refusal message is:

```text
I don't know based on the provided company documents. The available documents do not provide sufficient evidence to answer this question.
```

The system does not intentionally answer the five unanswerable golden-set questions.

---

## 9. Day 4 Generation Evaluation

The generation evaluator tests all 30 golden questions.

For answerable questions, an answer passes when:

1. The normalized generated answer exactly matches the normalized gold answer, or
2. All gold numeric facts are present and at least 60% of meaningful gold words overlap with the generated answer.

Citations are independently checked against the expected source file and page.

### Measured Day 4 results

| Metric                             |       Result |
| ---------------------------------- | -----------: |
| Answerable questions               | 25/25 (100%) |
| Correct unanswerable refusals      |   5/5 (100%) |
| Overall answer/refusal correctness | 30/30 (100%) |
| Citation correctness               | 25/25 (100%) |
| Citation validation                | 25/25 (100%) |
| Expected source in top-5           | 25/25 (100%) |

Detailed results:

```text
evals/results/day4_generation_results.csv
```

---

## 10. Day 5 Full Evaluation

Run the complete evaluation with:

```powershell
python .\evals\evaluate_all.py
```

The evaluator loads all 30 golden questions, verifies the Day 3 retrieval result, and freshly evaluates generation on all 30 questions.

Latest measured results:

| Metric                             |         Result |
| ---------------------------------- | -------------: |
| Retrieval Recall@5                 | 25/25 (100.0%) |
| Generation accuracy                | 25/25 (100.0%) |
| Correct refusals                   |   5/5 (100.0%) |
| Citation correctness               | 25/25 (100.0%) |
| Citation validation                | 25/25 (100.0%) |
| Expected source in top-5           | 25/25 (100.0%) |
| Overall answer/refusal correctness | 30/30 (100.0%) |
| Average generation latency         |      1322.25 ms |

Evaluation output is also saved to:

```text
evals/results/day5_full_evaluation.csv
```

---

## 11. Streamlit UI

Start the application with:

```powershell
python -m streamlit run .\app.py
```

The interface provides:

* Policy question chat
* Grounded answers
* Source-file/page citations
* Citation verification
* Retrieved document chunks
* Vector distances
* Selected evidence
* Evidence sufficiency
* Refusal behavior
* Response time

### UI acceptance tests

The answerable test:

```text
How many paid annual leave days does an employee receive per calendar year?
```

returned:

```text
Employees receive 18 paid annual leave days per calendar year.
[Source: HR_Leave_Policy.pdf, page 1]
```

The citation was verified against the selected evidence.

The unanswerable test:

```text
What is the company's stock options policy?
```

correctly produced a refusal because the corpus did not contain sufficient evidence.

---

## 12. Project Structure

```text
week5task1/
|
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- data/
|   |-- documents/
|   |-- processed/
|   |-- chunks/
|   |   |-- 250/
|   |   |-- 500/
|   |   `-- 1000/
|   `-- golden_set/
|       `-- golden_set.csv
|
|-- docs/
|   |-- DAY1_CORPUS_AUDIT.md
|   |-- DAY2_INGESTION_AUDIT.md
|   |-- DAY3_RETRIEVAL_RESULTS.md
|   |-- DAY4_GENERATION_AUDIT.md
|   |-- DAY5_RESULTS.md
|   `-- Architecture.png
|
|-- evals/
|   |-- evaluate_all.py
|   |-- evaluate_generation.py
|   |-- evaluate_hybrid.py
|   |-- evaluate_recall.py
|   |-- evaluate_reranker.py
|   `-- results/
|
|-- src/
|   |-- generation/
|   |   `-- grounded_qa.py
|   `-- ingestion/
|       |-- extract_pdfs.py
|       |-- chunk_documents.py
|       `-- build_chroma.py
|
|-- storage/
|   `-- chroma/
|
`-- tests/
```

---

## 13. Setup

### Requirements

* Python 3.12.x
* Git
* Internet access for downloading Hugging Face model weights on first use

### Create the virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

---

## 14. Run the Evaluation

From the project root:

```powershell
python .\evals\evaluate_all.py
```

The evaluator should report the measured retrieval and generation metrics and write:

```text
evals/results/day5_full_evaluation.csv
```

---

## 15. Run the UI

From the project root:

```powershell
python -m streamlit run .\app.py
```

Then open the local Streamlit address displayed by Streamlit.

---

## 16. Individual Evaluation Scripts

The individual evaluation scripts can also be run separately.

### Retrieval Recall

```powershell
python .\evals\evaluate_recall.py
```

### BM25 Hybrid Retrieval

```powershell
python .\evals\evaluate_hybrid.py
```

### Cross-Encoder Reranking

```powershell
python .\evals\evaluate_reranker.py
```

### Generation Evaluation

```powershell
python .\evals\evaluate_generation.py
```

### Complete Day 5 Evaluation

```powershell
python .\evals\evaluate_all.py
```

---

## 17. Day-by-Day Deliverables

### Day 1 — Corpus and Golden Set

Completed:

* 30-document corpus audit
* 30-question golden set
* 25 answerable questions
* 5 unanswerable questions
* Expected source file/page for answerable questions
* Golden set committed before retrieval pipeline code

Documentation:

```text
docs/DAY1_CORPUS_AUDIT.md
```

### Day 2 — Ingestion and Chunking

Completed:

* PDF extraction
* Metadata preservation
* 250-token chunks
* 500-token chunks
* 1000-token chunks
* Separate Chroma collections
* Extraction audit

Documentation:

```text
docs/DAY2_INGESTION_AUDIT.md
```

### Day 3 — Retrieval Comparison

Completed:

* Vector retrieval comparison
* Recall@5 measurement
* BM25 hybrid evaluation
* Cross-encoder reranker evaluation
* Selected 250-token configuration

Documentation:

```text
docs/DAY3_RETRIEVAL_RESULTS.md
```

### Day 4 — Grounded Generation

Completed:

* Evidence-first answering
* Source/page citations
* Refusal behavior
* 30-question generation evaluation
* Citation validation

Documentation:

```text
docs/DAY4_GENERATION_AUDIT.md
```

### Day 5 — UI, Evaluation and Handover

Completed:

* Streamlit chat UI
* One-command full evaluation
* Requirements file
* README
* Measured evaluation results
* UI acceptance testing

---

## 18. Limitations

This evaluation has several limitations:

1. The corpus is synthetic and was created specifically for the assignment.
2. The corpus contains 30 one-page PDF documents.
3. Retrieval results are therefore not representative of a large production document collection.
4. All evaluated questions are from the fixed 30-question golden set.
5. The three vector chunk configurations tied at 100% Recall@5, so the evaluation does not establish that 250-token chunks are universally better.
6. BM25 and cross-encoder reranking produced no measured Recall@5 improvement on this dataset.
7. The final grounded QA layer uses deterministic evidence selection and answer extraction rather than an unconstrained generative model.
8. The full evaluator uses the verified Day 3 retrieval results for the retrieval metric and freshly evaluates generation.

---

## 19. Reproducibility

The project includes the source policy documents, golden evaluation set, ingestion and chunking scripts, Chroma database builder, evaluation scripts, recorded results, dependency specifications, and day-by-day audit documentation.

### Clean-clone setup

Use Python 3.12 and Windows PowerShell. Run these commands in order:

```powershell
git clone https://github.com/hsmshubham007-code/week5task1.git
cd week5task1

py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install -r .\requirements.txt

python .\src\ingestion\extract_pdfs.py
python .\src\ingestion\chunk_documents.py
python .\src\ingestion\build_chroma.py

python .\evals\evaluate_all.py
python -m streamlit run .\app.py
```

The extraction, chunking, and database-build steps generate the local artifacts and Chroma database needed by the application. Complete these steps before running the evaluation or starting the UI in a fresh clone.

The first use of the embedding model requires internet access to download the required Hugging Face model files. The current deterministic grounded QA implementation does not require an API key.

### Evaluation output

The full evaluation is run with:

```powershell
python .\evals\evaluate_all.py
```

The evaluation output is saved to `evals/results/day5_full_evaluation.csv`.

The evaluator verifies the saved Day 3 retrieval results and freshly evaluates generation against the 30-question golden set. Therefore, the reported retrieval metric comes from the saved Day 3 result; it is not necessarily recalculated by every Day 5 evaluation run.

---

## 20. Final Measured Result

The final Day 5 evaluation achieved:

**30/30 overall answer/refusal correctness (100.0%)**

with:

* **100.0% Recall@5**
* **100.0% generation accuracy**
* **100.0% refusal accuracy**
* **100.0% citation correctness**
* **100.0% citation validation**
* **1322.25 ms average generation latency**

These values are measured results from the project's 30-question golden evaluation set, not qualitative performance claims.
