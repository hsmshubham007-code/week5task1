# Day 4 Generation & Grounding Audit

## Objective

Day 4 evaluates the generation layer of the NovaTech Solutions RAG system. The system must:

* answer questions only from retrieved company-document evidence;
* avoid guessing when the documents do not contain sufficient evidence;
* provide a source filename and page citation for supported answers;
* preserve important facts such as numbers and dates from the source evidence; and
* evaluate all 30 questions in the golden set.

The Day 4 evaluation uses the existing 30-question golden set created during Day 1.

## Evaluation Inputs

* Golden set: `data/golden_set/golden_set.csv`
* Evaluation script: `evals/evaluate_generation.py`
* Grounded QA implementation: `src/generation/grounded_qa.py`
* Results: `evals/results/day4_generation_results.csv`
* Retrieval collection used by grounded QA: `rag_chunks_250`
* Retrieval top-K: 5
* Embedding model: `BAAI/bge-m3`
* Generation model: `HuggingFaceTB/SmolLM2-360M-Instruct`

The evaluation contains:

* 25 answerable questions
* 5 intentionally unanswerable questions

All 30 questions were evaluated.

## Grounding Behavior

The grounded QA implementation retrieves document evidence before generation and restricts generation to that evidence. It also validates generated content against the retrieved evidence.

For questions where the available company documents do not provide sufficient evidence, the system returns a refusal rather than inventing an answer.

The refusal used by the system is:

> I don't know based on the provided company documents. The available documents do not provide sufficient evidence to answer this question.

Source citations are added programmatically using the retrieved document filename and page number.

## Answer Correctness Criterion

The Day 4 evaluator does not require an exact string match for every answer.

An answer is considered correct when either:

1. the generated answer exactly matches the expected answer; or
2. the required gold numbers are present in the generated answer and the generated answer has at least 60% meaningful-word overlap with the expected answer.

This criterion allows minor wording differences while still requiring important factual values to be preserved.

Therefore, the reported generation accuracy below should be described as **answer correctness under the evaluation criterion**, not exact-match accuracy.

## Measured Results

### Overall results

| Metric                                                      |   Result |
| ----------------------------------------------------------- | -------: |
| Total questions evaluated                                   |    30/30 |
| Answerable questions                                        |       25 |
| Unanswerable questions                                      |        5 |
| Correct answerable answers                                  |    25/25 |
| Answer correctness                                          | **100%** |
| Correct refusals                                            |      5/5 |
| Refusal correctness                                         | **100%** |
| Correct citations for answerable questions                  |    25/25 |
| Citation correctness                                        | **100%** |
| Expected source retrieved in top-5 for answerable questions |    25/25 |
| Source retrieval rate                                       | **100%** |

### Answerable questions

All 25 answerable questions were evaluated as correct.

* Correct: **25/25**
* Accuracy: **100%**
* Correct citations: **25/25**
* Citation accuracy: **100%**
* Expected source retrieved in top-5: **25/25**
* Top-5 source retrieval rate: **100%**

### Unanswerable questions

All 5 intentionally unanswerable questions were correctly refused.

* Correct refusals: **5/5**
* Refusal accuracy: **100%**

For these questions, the expected source was not present in the top-5 retrieved results, and the system did not fabricate an answer from unrelated evidence.

## Question-Level Result

The evaluated question groups produced the following results:

| Questions | Answerability |       Answer correct |      Refusal correct |     Citation correct | Expected source in top-5 |
| --------- | ------------- | -------------------: | -------------------: | -------------------: | -----------------------: |
| Q001–Q025 | Answerable    |                25/25 |                 0/25 |                25/25 |                    25/25 |
| Q026–Q030 | Unanswerable  |                 0/5* |                  5/5 |                 5/5* |                      0/5 |
| **Total** | **30**        | **25/25 answerable** | **5/5 unanswerable** | **25/25 answerable** |     **25/25 answerable** |

`*` For unanswerable questions, an answer is not expected and citation correctness is not a substantive success criterion. The important metric for this group is refusal correctness.

## Acceptance Criteria

| Day 4 requirement                                       | Status | Evidence                           |
| ------------------------------------------------------- | ------ | ---------------------------------- |
| Evaluate all 30 golden questions                        | PASS   | 30/30 evaluated                    |
| Answer using retrieved document evidence                | PASS   | Grounded QA implementation         |
| Do not guess when evidence is insufficient              | PASS   | 5/5 unanswerable questions refused |
| Correctly answer answerable questions                   | PASS   | 25/25 = 100%                       |
| Cite source file and page                               | PASS   | 25/25 = 100%                       |
| Verify refusal behavior on all 5 unanswerable questions | PASS   | 5/5 = 100%                         |
| Record numerical evaluation results                     | PASS   | `day4_generation_results.csv`      |
| Document generation and grounding evaluation            | PASS   | This audit                         |

## Day 4 Conclusion

**Day 4 PASS.**

The grounded generation system correctly answered all 25 answerable golden-set questions under the documented answer-correctness criterion, correctly refused all 5 intentionally unanswerable questions, and provided correct source citations for all 25 supported answers.

Measured results:

* **Answer correctness: 100% (25/25)**
* **Refusal correctness: 100% (5/5)**
* **Citation correctness: 100% (25/25)**
* **Expected source retrieved in top-5: 100% (25/25)**

These results apply to the current synthetic NovaTech Solutions corpus and 30-question evaluation set. They should not be interpreted as evidence of the same performance on larger or real-world corpora.

## Limitations

1. The evaluation corpus is synthetic and intentionally structured for the assignment.
2. The golden set contains only 30 questions.
3. The corpus contains no scanned/image-only PDFs, so OCR behavior was not tested.
4. The 100% results are specific to this corpus and evaluation set and should not be generalized to production-scale or heterogeneous document collections.
5. The answer-correctness metric permits wording variation and is therefore not equivalent to exact-string accuracy.

## Reproducibility

From the project root, run:

```powershell
python .\evals\evaluate_generation.py
```

The evaluation reads the golden set and writes:

```text
evals/results/day4_generation_results.csv
```

The resulting CSV contains the question-level evaluation data used to produce the measured Day 4 results documented above.
