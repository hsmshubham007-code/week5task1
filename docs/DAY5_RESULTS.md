# Day 5 Results & Handover

## Objective

Day 5 completes the RAG evaluation workflow with a Streamlit user interface, a one-command evaluation script, reproducible dependency specifications, and measured evaluation results.

## Evaluation Dataset

* Corpus: 30 synthetic NovaTech Solutions internal policy PDFs
* Documents/pages: 30 documents, 30 pages
* Golden questions: 30
* Answerable questions: 25
* Unanswerable questions: 5
* Retrieval top-K: 5
* Selected retrieval configuration: 250-token chunks
* Embedding model: `BAAI/bge-m3`
* Vector database: Chroma

The 250-token configuration was retained because all three tested chunk sizes achieved the same Recall@5 of 100%. It was selected as a deterministic tie-breaker, not because it achieved a higher retrieval score.

## Final Correctness Results

| Metric                             |         Result |
| ---------------------------------- | -------------: |
| Retrieval Recall@5                 | 25/25 (100.0%) |
| Generation accuracy                | 25/25 (100.0%) |
| Correct unanswerable refusals      |   5/5 (100.0%) |
| Citation correctness               | 25/25 (100.0%) |
| Citation validation                | 25/25 (100.0%) |
| Expected source in top-5           | 25/25 (100.0%) |
| Overall answer/refusal correctness | 30/30 (100.0%) |

These correctness results are measured on the fixed 30-question golden evaluation set. They should not be interpreted as a guarantee of the same accuracy on unseen questions or larger, real-world corpora.

## Latency Measurements

Generation latency varies by environment and run.

| Run                             | Average generation latency |
| ------------------------------- | -------------------------- |
| Previously recorded project run | 1322.25 ms                 |
| Clean-clone verification run    | 1418.50 ms                 |
| Latest Day 5 evaluation run     | 1508.46 ms                 |


The first generation request in the clean-clone run took approximately 36.1 seconds while the model was initializing. Subsequent requests were substantially faster. The reported average includes the initialization request, so it should not be interpreted as steady-state response latency.

## Generation and Grounding

The grounded QA system uses an evidence-first answer process. It retrieves the top five Chroma results, checks whether sufficient evidence exists, selects supporting evidence, and produces an answer only when the retrieved policy text supports it.

For unanswerable questions, the system returns:

> I don't know based on the provided company documents. The available documents do not provide sufficient evidence to answer this question.

Answerable responses include the source filename and page number. Citation validation checks the citation against the expected golden-set source and page.

Generation correctness uses the audited Day 4 criterion: either a normalized exact match, or the expected numerical values together with at least 60% meaningful-word overlap.

## User Interface

The project includes a Streamlit application in `app.py`.

The UI provides:

* Natural-language policy questions
* Grounded answers
* Source filename and page citations
* Citation verification status
* Retrieved source chunks
* Evidence sufficiency and selected evidence
* Explicit refusal when the documents do not contain sufficient evidence
* Response-time measurement

The UI was manually tested in a fresh clone with both an answerable policy question and an unanswerable stock-options question. The answerable question returned 18 paid annual leave days with a verified citation to `HR_Leave_Policy.pdf`, page 1. The unsupported stock-options question was refused.

## Reproducing the Project from a Fresh Clone

Use Python 3.12 and Windows PowerShell.

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

The extraction, chunking, and database-build steps create local generated files and the Chroma database required by the application. Run them before evaluation or starting the UI in a fresh clone.

The first model use requires internet access to download the required Hugging Face model weights. The current deterministic grounded QA implementation does not require an API key.

## Evaluation Command

After completing the ingestion and database-build steps, the complete Day 5 evaluation can be run with:

```powershell
python .\evals\evaluate_all.py
```

The evaluation output is saved to:

```text
evals/results/day5_full_evaluation.csv
```

The evaluator verifies the saved Day 3 retrieval results and freshly evaluates generation on the 30-question golden set. Therefore, the reported retrieval metric is based on the saved Day 3 result, not necessarily a fresh retrieval experiment during every Day 5 run.

## Deliverables

* `data/documents/` — synthetic policy PDFs
* `data/golden_set/golden_set.csv` — 30-question evaluation set
* `src/ingestion/` — PDF extraction, chunking, and Chroma database build scripts
* `src/generation/grounded_qa.py` — grounded question-answering implementation
* `evals/` — retrieval and generation evaluation scripts
* `evals/results/` — recorded evaluation outputs
* `docs/DAY1_CORPUS_AUDIT.md` — corpus and golden-set audit
* `docs/DAY2_INGESTION_AUDIT.md` — ingestion and chunking audit
* `docs/DAY3_RETRIEVAL_RESULTS.md` — retrieval comparison
* `docs/DAY4_GENERATION_AUDIT.md` — grounded generation audit
* `docs/DAY5_RESULTS.md` — final results and handover
* `docs/Architecture.png` — system architecture diagram
* `app.py` — Streamlit user interface
* `requirements.txt` — pinned Python dependencies
* `README.md` — setup, execution, and reproduction instructions

## Limitations

1. The corpus is synthetic and was created specifically for this assignment.
2. The corpus contains 30 one-page PDF documents.
3. All evaluated questions belong to the fixed 30-question golden set.
4. The three chunk configurations tied at 100% Recall@5, so the evaluation does not establish that 250-token chunks are universally better.
5. BM25 hybrid retrieval and cross-encoder reranking produced no measured Recall@5 improvement on this dataset.
6. The final grounded QA layer uses deterministic evidence selection and answer extraction rather than an unconstrained generative model.
7. The full Day 5 evaluator uses the saved Day 3 retrieval result for its retrieval metric and freshly evaluates generation.

## Final Result

On the fixed 30-question golden evaluation set, the recorded results were:

* **Overall answer/refusal correctness:** 30/30 (100.0%)
* **Generation accuracy:** 25/25 (100.0%)
* **Correct refusals:** 5/5 (100.0%)
* **Citation correctness:** 25/25 (100.0%)
* **Citation validation:** 25/25 (100.0%)
* **Recall@5:** 25/25 (100.0%)

A clean-clone verification confirmed that the dependencies could be installed, the corpus and vector database could be rebuilt, the evaluation could be run, and both answerable and unanswerable UI acceptance tests behaved as expected.
