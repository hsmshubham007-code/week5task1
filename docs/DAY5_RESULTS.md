# Day 5 Results & Handover

## Objective

Day 5 completes the RAG evaluation workflow with a Streamlit user interface, a single-command evaluation script, reproducible dependency specification, and measured final results.

## Evaluation Dataset

- Corpus: 30 synthetic NovaTech Solutions internal policy PDFs
- Documents/pages: 30 documents, 30 pages
- Golden questions: 30
- Answerable questions: 25
- Unanswerable questions: 5
- Retrieval top-K: 5
- Selected retrieval configuration: 250-token chunks
- Embedding model: `BAAI/bge-m3`
- Vector database: Chroma

The 250-token configuration was retained because all three tested chunk sizes achieved the same Recall@5 of 100%. It was selected as the deterministic tie-break rather than because it produced a higher retrieval score.

## Final Measured Results

| Metric | Result |
|---|---:|
| Retrieval Recall@5 | 25/25 (100.0%) |
| Generation accuracy | 25/25 (100.0%) |
| Correct unanswerable refusals | 5/5 (100.0%) |
| Citation correctness | 25/25 (100.0%) |
| Citation validation | 25/25 (100.0%) |
| Expected source in top-5 | 25/25 (100.0%) |
| Overall answer/refusal correctness | 30/30 (100.0%) |
| Average generation latency | 1322.25 ms |

The first generation request incurred model initialization overhead of approximately 33.7 seconds; subsequent requests were substantially faster.

## Generation and Grounding

The grounded QA system uses an evidence-first deterministic answer process. It retrieves the top five Chroma results, checks whether sufficient evidence exists, selects supporting evidence, and produces an answer only when the retrieved policy text supports it.

For unanswerable questions, the system returns:

> I don't know based on the provided company documents. The available documents do not provide sufficient evidence to answer this question.

Answerable responses include the source filename and page number. Citation validation checks that the cited source/page corresponds to the expected golden-set source.

Generation correctness uses the audited Day 4 criterion: either a normalized exact match, or the expected numerical values together with at least 60% meaningful-word overlap.

## User Interface

The project includes a Streamlit application in `app.py`.

The UI provides:

- Natural-language policy questions
- Grounded answers
- Source filename and page citations
- Citation verification status
- Retrieved source chunks
- Evidence sufficiency and selected evidence
- Explicit refusal when the documents do not contain sufficient evidence
- Response-time measurement

The UI was manually tested with both an answerable policy question and an unanswerable stock-options question.

## Evaluation Command

The complete Day 5 evaluation can be run with:

```powershell
python .\evals\evaluate_all.py