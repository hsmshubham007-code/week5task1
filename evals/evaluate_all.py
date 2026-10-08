from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_SET_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden_set"
    / "golden_set.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evals"
    / "results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "day5_full_evaluation.csv"
)

DAY3_RECALL_PATH = (
    RESULTS_DIR
    / "recall_250_question_results.csv"
)


# ---------------------------------------------------------------------------
# Allow imports from the project root
# ---------------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT))


from src.generation.grounded_qa import answer_question


# ---------------------------------------------------------------------------
# Scoring helpers
#
# These reproduce the audited Day 4 scoring methodology exactly.
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for factual comparison."""

    text = text.lower()

    text = re.sub(
        r"\[source:\s*[^,\]]+,\s*page\s+\d+\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"[^a-z0-9₹.,:%/-]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_numbers(text: str) -> set[str]:
    """Extract numeric facts for comparison."""

    normalized = text.lower()

    normalized = normalized.replace(
        ",",
        "",
    )

    matches = re.findall(
        r"(?:₹\s*)?\d+(?:\.\d+)?",
        normalized,
    )

    return {
        match.replace("₹", "").strip()
        for match in matches
    }


def answer_contains_gold_fact(
    generated_answer: str,
    gold_answer: str,
) -> bool:
    """
    Exact Day 4 answer-correctness criterion.

    Passes when:
      1. normalized answers match exactly, OR
      2. all gold numbers are present and at least 60% of the
         meaningful gold words overlap with the generated answer.
    """

    generated_normalized = normalize_text(
        generated_answer
    )

    gold_normalized = normalize_text(
        gold_answer
    )

    if generated_normalized == gold_normalized:
        return True

    gold_numbers = extract_numbers(
        gold_answer
    )

    generated_numbers = extract_numbers(
        generated_answer
    )

    if gold_numbers:
        if not gold_numbers.issubset(
            generated_numbers
        ):
            return False

    stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "be",
        "to",
        "of",
        "for",
        "and",
        "or",
        "when",
        "that",
        "which",
        "up",
        "more",
        "than",
        "with",
        "days",
        "day",
        "weeks",
        "week",
        "months",
        "month",
        "years",
        "year",
        "hours",
        "hour",
    }

    gold_words = {
        word
        for word in re.findall(
            r"\b[a-z]+\b",
            gold_normalized,
        )
        if word not in stop_words
    }

    generated_words = {
        word
        for word in re.findall(
            r"\b[a-z]+\b",
            generated_normalized,
        )
        if word not in stop_words
    }

    if not gold_words:
        return bool(gold_numbers)

    overlap = gold_words.intersection(
        generated_words
    )

    overlap_ratio = (
        len(overlap)
        / len(gold_words)
    )

    return overlap_ratio >= 0.60


def citation_matches_expected_source(
    result: dict,
    expected_source_file: str,
    expected_page: int | None,
) -> bool:
    """Verify the generated citation matches expected source and page."""

    if not expected_source_file:
        return False

    if expected_page is None:
        return False

    expected_key = (
        expected_source_file.strip().lower(),
        int(expected_page),
    )

    citations = re.findall(
        r"\[Source:\s*([^,\]]+),\s*page\s+(\d+)\]",
        result.get("answer", ""),
        flags=re.IGNORECASE,
    )

    if not citations:
        return False

    for source_file, page in citations:
        citation_key = (
            source_file.strip().lower(),
            int(page),
        )

        if citation_key == expected_key:
            return True

    return False


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------

def load_golden_set() -> list[dict]:
    """Load and validate the complete 30-question golden set."""

    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_SET_PATH}"
        )

    with GOLDEN_SET_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if len(rows) != 30:
        raise ValueError(
            f"Expected 30 golden questions, found {len(rows)}."
        )

    required_columns = {
        "question_id",
        "question",
        "gold_answer",
        "source_file",
        "source_page",
        "answerability",
    }

    missing = (
        required_columns
        - set(reader.fieldnames or [])
    )

    if missing:
        raise ValueError(
            "Golden set is missing required columns: "
            f"{sorted(missing)}"
        )

    return rows


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    rows: list[dict],
) -> tuple[int, int, float]:
    """
    Read the verified Day 3 250-token retrieval results.

    Day 3 evaluated the 25 answerable questions and established
    the selected 250-token configuration at 100% Recall@5.
    """

    if not DAY3_RECALL_PATH.exists():
        raise FileNotFoundError(
            "Day 3 retrieval results were not found:\n"
            f"{DAY3_RECALL_PATH}\n\n"
            "Run evaluate_recall.py first."
        )

    answerable_ids = {
        row["question_id"]
        for row in rows
        if row["answerability"].strip().lower()
        == "answerable"
    }

    hits = 0
    evaluated = 0

    with DAY3_RECALL_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            question_id = row["question_id"]

            if question_id not in answerable_ids:
                continue

            evaluated += 1

            if (
                row["hit_at_5"]
                .strip()
                .lower()
                == "true"
            ):
                hits += 1

    expected_count = len(answerable_ids)

    if evaluated != expected_count:
        raise ValueError(
            "Day 3 retrieval result count does not match "
            f"the golden set: expected {expected_count}, "
            f"found {evaluated}."
        )

    recall = (
        hits / evaluated
        if evaluated
        else 0.0
    )

    return hits, evaluated, recall


# ---------------------------------------------------------------------------
# Fresh Day 5 generation evaluation
# ---------------------------------------------------------------------------

def evaluate_generation(
    rows: list[dict],
) -> list[dict]:
    """Freshly evaluate all 30 golden questions."""

    results = []

    print()
    print("=" * 70)
    print("FRESH GENERATION EVALUATION - ALL 30 QUESTIONS")
    print("=" * 70)

    for index, row in enumerate(
        rows,
        start=1,
    ):

        question_id = row["question_id"]
        question = row["question"]
        gold_answer = row["gold_answer"]

        answerability = (
            row["answerability"]
            .strip()
            .lower()
        )

        expected_answerable = (
            answerability == "answerable"
        )

        expected_source = (
            row["source_file"]
            .strip()
        )

        source_page_raw = (
            row["source_page"]
            .strip()
        )

        expected_page = (
            int(source_page_raw)
            if source_page_raw
            else None
        )

        print()
        print(
            f"[{index}/30] "
            f"{question_id}: {question}"
        )

        start_time = time.perf_counter()

        result = answer_question(question)

        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        actual_refused = bool(
            result.get(
                "refused",
                False,
            )
        )

        generated_answer = result.get(
            "answer",
            "",
        )

        # ---------------------------------------------------------------
        # Answer/refusal correctness
        # ---------------------------------------------------------------

        if expected_answerable:

            answer_correct = (
                not actual_refused
                and answer_contains_gold_fact(
                    generated_answer,
                    gold_answer,
                )
            )

            refusal_correct = False

        else:

            answer_correct = False

            refusal_correct = actual_refused

        # ---------------------------------------------------------------
        # Citation correctness
        # ---------------------------------------------------------------

        if expected_answerable:

            citation_correct = (
                citation_matches_expected_source(
                    result,
                    expected_source,
                    expected_page,
                )
            )

        else:

            citation_correct = True

        # ---------------------------------------------------------------
        # Expected source in retrieved top-5
        # ---------------------------------------------------------------

        source_retrieved = False

        if expected_answerable:

            for document in result.get(
                "retrieved",
                [],
            ):

                retrieved_source = (
                    document.source_file
                    .strip()
                    .lower()
                )

                retrieved_page = int(
                    document.page
                )

                if (
                    retrieved_source
                    == expected_source.lower()
                    and retrieved_page
                    == expected_page
                ):
                    source_retrieved = True
                    break

        print(
            f"Refused: {actual_refused}"
        )

        print(
            f"Answer correct: "
            f"{answer_correct}"
        )

        print(
            f"Refusal correct: "
            f"{refusal_correct}"
        )

        print(
            f"Citation correct: "
            f"{citation_correct}"
        )

        print(
            f"Source retrieved in top-5: "
            f"{source_retrieved}"
        )

        print(
            f"Latency: "
            f"{elapsed_ms:.2f} ms"
        )

        results.append(
            {
                "question_id": question_id,
                "question": question,
                "answerability": answerability,
                "generated_answer": generated_answer,
                "refused": actual_refused,
                "evidence_sufficient": bool(
                    result.get(
                        "evidence_sufficient",
                        False,
                    )
                ),
                "citation_valid": bool(
                    result.get(
                        "citation_valid",
                        False,
                    )
                ),
                "citation_correct": citation_correct,
                "source_retrieved_in_top5": (
                    source_retrieved
                ),
                "answer_correct": answer_correct,
                "refusal_correct": refusal_correct,
                "latency_ms": round(
                    elapsed_ms,
                    2,
                ),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Save Day 5 results
# ---------------------------------------------------------------------------

def save_results(
    retrieval_hits: int,
    retrieval_total: int,
    retrieval_recall: float,
    generation_results: list[dict],
) -> None:
    """Write the complete Day 5 evaluation CSV."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        fieldnames = [
            "metric",
            "question_id",
            "answerability",
            "result",
            "details",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerow(
            {
                "metric": "retrieval_recall_at_5",
                "question_id": "",
                "answerability": "answerable",
                "result": (
                    f"{retrieval_hits}/"
                    f"{retrieval_total}"
                ),
                "details": (
                    f"{retrieval_recall * 100:.2f}%"
                ),
            }
        )

        for result in generation_results:

            if (
                result["answerability"]
                == "answerable"
            ):
                correctness = (
                    result["answer_correct"]
                )
            else:
                correctness = (
                    result["refusal_correct"]
                )

            writer.writerow(
                {
                    "metric": "generation",
                    "question_id": (
                        result["question_id"]
                    ),
                    "answerability": (
                        result["answerability"]
                    ),
                    "result": correctness,
                    "details": (
                        result["generated_answer"]
                    ),
                }
            )

    print()
    print(
        "Combined results saved to:"
    )
    print(RESULTS_PATH)


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_report(
    retrieval_hits: int,
    retrieval_total: int,
    retrieval_recall: float,
    generation_results: list[dict],
) -> None:

    answerable = [
        result
        for result in generation_results
        if result["answerability"]
        == "answerable"
    ]

    unanswerable = [
        result
        for result in generation_results
        if result["answerability"]
        == "unanswerable"
    ]

    answer_correct = sum(
        bool(result["answer_correct"])
        for result in answerable
    )

    refusal_correct = sum(
        bool(result["refusal_correct"])
        for result in unanswerable
    )

    citation_correct = sum(
        bool(result["citation_correct"])
        for result in answerable
    )

    citation_valid = sum(
        bool(result["citation_valid"])
        for result in answerable
    )

    source_retrieved = sum(
        bool(
            result["source_retrieved_in_top5"]
        )
        for result in answerable
    )

    overall_correct = (
        answer_correct
        + refusal_correct
    )

    overall_total = (
        len(answerable)
        + len(unanswerable)
    )

    print()
    print("=" * 70)
    print("DAY 5 - FULL 30-QUESTION EVALUATION")
    print("=" * 70)

    print()
    print(
        f"Retrieval Recall@5:       "
        f"{retrieval_hits}/{retrieval_total} "
        f"({retrieval_recall * 100:.1f}%)"
    )

    print(
        f"Generation accuracy:      "
        f"{answer_correct}/{len(answerable)} "
        f"({answer_correct / len(answerable) * 100:.1f}%)"
    )

    print(
        f"Correct refusals:         "
        f"{refusal_correct}/{len(unanswerable)} "
        f"({refusal_correct / len(unanswerable) * 100:.1f}%)"
    )

    print(
        f"Citation correctness:     "
        f"{citation_correct}/{len(answerable)} "
        f"({citation_correct / len(answerable) * 100:.1f}%)"
    )

    print(
        f"Citation validation:      "
        f"{citation_valid}/{len(answerable)} "
        f"({citation_valid / len(answerable) * 100:.1f}%)"
    )

    print(
        f"Expected source in top-5: "
        f"{source_retrieved}/{len(answerable)} "
        f"({source_retrieved / len(answerable) * 100:.1f}%)"
    )

    print(
        f"Overall answer/refusal:    "
        f"{overall_correct}/{overall_total} "
        f"({overall_correct / overall_total * 100:.1f}%)"
    )

    latencies = [
        result["latency_ms"]
        for result in generation_results
    ]

    if latencies:

        average_latency = (
            sum(latencies)
            / len(latencies)
        )

        print(
            f"Average generation latency: "
            f"{average_latency:.2f} ms"
        )

    failed_answerable = [
        result
        for result in answerable
        if not result["answer_correct"]
    ]

    failed_refusals = [
        result
        for result in unanswerable
        if not result["refusal_correct"]
    ]

    if failed_answerable:

        print()
        print(
            "FAILED ANSWERABLE QUESTIONS:"
        )

        for result in failed_answerable:

            print(
                f"- {result['question_id']}: "
                f"{result['question']}"
            )

    if failed_refusals:

        print()
        print(
            "FAILED REFUSALS:"
        )

        for result in failed_refusals:

            print(
                f"- {result['question_id']}: "
                f"{result['question']}"
            )

    print()
    print("=" * 70)

    if (
        not failed_answerable
        and not failed_refusals
    ):
        print(
            "All 30 questions passed "
            "the Day 5 correctness checks."
        )
    else:
        print(
            "Some Day 5 correctness checks failed."
        )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("DAY 5 - FULL 30-QUESTION EVALUATION")
    print("=" * 70)

    print()
    print(
        f"Golden set: {GOLDEN_SET_PATH}"
    )

    rows = load_golden_set()

    print(
        f"Questions loaded: {len(rows)}"
    )

    answerable_count = sum(
        row["answerability"]
        .strip()
        .lower()
        == "answerable"
        for row in rows
    )

    unanswerable_count = (
        len(rows)
        - answerable_count
    )

    print(
        f"Answerable: {answerable_count}"
    )

    print(
        f"Unanswerable: {unanswerable_count}"
    )

    # -----------------------------------------------------------------------
    # Verified Day 3 retrieval metric
    # -----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    (
        retrieval_hits,
        retrieval_total,
        retrieval_recall,
    ) = evaluate_retrieval(rows)

    print(
        f"\nVerified Day 3 Recall@5: "
        f"{retrieval_hits}/{retrieval_total} "
        f"({retrieval_recall * 100:.1f}%)"
    )

    # -----------------------------------------------------------------------
    # Fresh generation evaluation
    # -----------------------------------------------------------------------

    generation_results = (
        evaluate_generation(rows)
    )

    # -----------------------------------------------------------------------
    # Save and print
    # -----------------------------------------------------------------------

    save_results(
        retrieval_hits=retrieval_hits,
        retrieval_total=retrieval_total,
        retrieval_recall=retrieval_recall,
        generation_results=generation_results,
    )

    print_final_report(
        retrieval_hits=retrieval_hits,
        retrieval_total=retrieval_total,
        retrieval_recall=retrieval_recall,
        generation_results=generation_results,
    )


if __name__ == "__main__":
    main()