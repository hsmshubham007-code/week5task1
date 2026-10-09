import csv
import re
import sys
import time
from pathlib import Path
from typing import Any


# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.generation.grounded_qa import answer_question


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
    / "day4_generation_results.csv"
)


def normalize_text(
    text: str,
) -> str:
    """Normalize text for deterministic comparison."""

    text = text.lower()

    text = re.sub(
        r"\[source:.*?page\s+\d+\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"[^a-z0-9₹%.\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_numbers(
    text: str,
) -> set[str]:
    """Extract meaningful numeric values."""

    text = text.lower()

    # Repair extraction artifacts such as:
    # "1. 5 times" -> "1.5 times"
    text = re.sub(
        r"(?<=\d)\.\s+(?=\d)",
        ".",
        text,
    )

    values = set()

    # Percentages.
    for match in re.findall(
        r"\b\d+(?:\.\d+)?\s*%",
        text,
    ):
        values.add(
            re.sub(
                r"\s+",
                "",
                match,
            )
        )

    # Currency / numeric values.
    for match in re.findall(
        r"₹\s*[\d,]+(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\b",
        text,
    ):
        value = match.replace(
            ",",
            "",
        )

        value = value.replace(
            "₹",
            "",
        ).strip()

        values.add(value)

    number_words = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
        "hundred": "100",
        "thousand": "1000",
    }

    for word, number in number_words.items():
        if re.search(
            rf"\b{word}\b",
            text,
        ):
            values.add(number)

    return values


def answer_contains_gold_fact(
    generated_answer: str,
    gold_answer: str,
) -> bool:
    """
    Check whether a generated answer contains the gold fact.

    Reject explicit contradictions and negations before
    checking numeric values and word overlap.
    """

    generated_normalized = normalize_text(
        generated_answer
    )

    gold_normalized = normalize_text(
        gold_answer
    )

    # Exact matches are accepted only when they contain
    # no explicit negation.
    negation_patterns = [
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bneither\b",
        r"\bwithout\b",
        r"\bdoesn't\b",
        r"\bdon't\b",
        r"\bdidn't\b",
        r"\bisn't\b",
        r"\baren't\b",
        r"\bwasn't\b",
        r"\bweren't\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bwon't\b",
        r"\bdoes not\b",
        r"\bdo not\b",
        r"\bdid not\b",
        r"\bis not\b",
        r"\bare not\b",
        r"\bwas not\b",
        r"\bwere not\b",
    ]

    # A conservative check: if the generated answer contains
    # explicit negation but the gold answer does not, do not
    # automatically mark it correct based on word overlap.
    generated_has_negation = any(
        re.search(pattern, generated_normalized)
        for pattern in negation_patterns
    )

    gold_has_negation = any(
        re.search(pattern, gold_normalized)
        for pattern in negation_patterns
    )

    if generated_has_negation and not gold_has_negation:
        return False

    if generated_normalized == gold_normalized:
        return True

    gold_numbers = extract_numbers(
        gold_answer
    )

    generated_numbers = extract_numbers(
        generated_answer
    )

    if gold_numbers and not gold_numbers.issubset(
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
    result,
    expected_source_file: str,
    expected_page: int | None,
) -> bool:
    """
    Verify that the answer cites the expected
    source file and page.
    """

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
        result["answer"],
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


def evaluate_question(
    qa: Any,
    row: dict,
) -> dict:

    question_id = row["question_id"]
    question = row["question"]
    gold_answer = row["gold_answer"]

    expected_source = (
        row["source_file"].strip()
    )

    source_page_raw = (
        row["source_page"].strip()
    )

    expected_page = (
        int(source_page_raw)
        if source_page_raw
        else None
    )

    answerability = (
        row["answerability"]
        .strip()
        .lower()
    )

    expected_answerable = (
        answerability == "answerable"
    )

    start_time = time.perf_counter()

    result = qa(
        question
    )

    elapsed_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    actual_refused = result["refused"]

    if expected_answerable:

        answer_correct = (
            not actual_refused
            and answer_contains_gold_fact(
                result["answer"],
                gold_answer,
            )
        )

        refusal_correct = False

    else:

        answer_correct = False

        refusal_correct = actual_refused

    if expected_answerable:

        citation_correct = (
            citation_matches_expected_source(
                result,
                expected_source,
                expected_page,
            )
        )

    else:

        # Unanswerable questions have no valid
        # source citation requirement.
        citation_correct = True

    source_retrieved = False

    if expected_answerable:

        for document in result["retrieved"]:

            if (
                document.source_file.strip().lower()
                == expected_source.lower()
                and document.page == expected_page
            ):
                source_retrieved = True
                break

    return {
        "question_id": question_id,
        "question": question,
        "gold_answer": gold_answer,
        "expected_source_file": expected_source,
        "expected_source_page": (
            expected_page
            if expected_page is not None
            else ""
        ),
        "answerability": answerability,
        "generated_answer": result["answer"],
        "refused": result["refused"],
        "evidence_sufficient": (
            result["evidence_sufficient"]
        ),
        "citation_valid": (
            result["citation_valid"]
        ),
        "citation_correct": (
            citation_correct
        ),
        "source_retrieved_in_top5": (
            source_retrieved
        ),
        "answer_correct": (
            answer_correct
        ),
        "refusal_correct": (
            refusal_correct
        ),
        "latency_ms": round(
            elapsed_ms,
            2,
        ),
    }


def print_summary(
    results: list[dict],
) -> None:

    answerable_results = [
        result
        for result in results
        if result["answerability"]
        == "answerable"
    ]

    unanswerable_results = [
        result
        for result in results
        if result["answerability"]
        == "unanswerable"
    ]

    answer_correct = sum(
        result["answer_correct"]
        for result in answerable_results
    )

    refusal_correct = sum(
        result["refusal_correct"]
        for result in unanswerable_results
    )

    citation_correct = sum(
        result["citation_correct"]
        for result in answerable_results
    )

    citation_valid = sum(
        result["citation_valid"]
        for result in answerable_results
    )

    source_retrieved = sum(
        result["source_retrieved_in_top5"]
        for result in answerable_results
    )

    total = len(results)

    overall_correct = (
        answer_correct
        + refusal_correct
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DAY 4 GENERATION EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAnswerable questions: "
        f"{answer_correct}/"
        f"{len(answerable_results)} "
        f"("
        f"{answer_correct / len(answerable_results) * 100:.1f}%"
        f")"
    )

    print(
        f"Unanswerable refusals: "
        f"{refusal_correct}/"
        f"{len(unanswerable_results)} "
        f"("
        f"{refusal_correct / len(unanswerable_results) * 100:.1f}%"
        f")"
    )

    print(
        f"Overall answer/refusal correctness: "
        f"{overall_correct}/{total} "
        f"("
        f"{overall_correct / total * 100:.1f}%"
        f")"
    )

    print(
        f"Citation correctness: "
        f"{citation_correct}/"
        f"{len(answerable_results)} "
        f"("
        f"{citation_correct / len(answerable_results) * 100:.1f}%"
        f")"
    )

    print(
        f"Citation validation: "
        f"{citation_valid}/"
        f"{len(answerable_results)} "
        f"("
        f"{citation_valid / len(answerable_results) * 100:.1f}%"
        f")"
    )

    print(
        f"Expected source retrieved in top-5: "
        f"{source_retrieved}/"
        f"{len(answerable_results)} "
        f"("
        f"{source_retrieved / len(answerable_results) * 100:.1f}%"
        f")"
    )

    print(
        "\n" + "-" * 70
    )

    failed_answerable = [
        result
        for result in answerable_results
        if not result["answer_correct"]
    ]

    failed_refusals = [
        result
        for result in unanswerable_results
        if not result["refusal_correct"]
    ]

    if failed_answerable:

        print(
            "\nFAILED ANSWERABLE QUESTIONS:"
        )

        for result in failed_answerable:

            print(
                f"\n{result['question_id']}: "
                f"{result['question']}"
            )

            print(
                f"Expected: "
                f"{result['gold_answer']}"
            )

            print(
                f"Generated: "
                f"{result['generated_answer']}"
            )

            print(
                f"Evidence sufficient: "
                f"{result['evidence_sufficient']}"
            )

            print(
                f"Source retrieved in top-5: "
                f"{result['source_retrieved_in_top5']}"
            )

    else:

        print(
            "\nAll 25 answerable questions passed."
        )

    if failed_refusals:

        print(
            "\nFAILED UNANSWERABLE QUESTIONS:"
        )

        for result in failed_refusals:

            print(
                f"\n{result['question_id']}: "
                f"{result['question']}"
            )

            print(
                f"Generated: "
                f"{result['generated_answer']}"
            )

    else:

        print(
            "All 5 unanswerable questions "
            "were correctly refused."
        )

    print(
        "\n" + "=" * 70
    )


def main() -> None:

    print(
        "=" * 70
    )

    print(
        "DAY 4 GENERATION EVALUATION"
    )

    print(
        "=" * 70
    )

    if not GOLDEN_SET_PATH.exists():

        raise FileNotFoundError(
            f"Golden set not found: "
            f"{GOLDEN_SET_PATH}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nGolden set: "
        f"{GOLDEN_SET_PATH}"
    )

    with GOLDEN_SET_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        rows = list(reader)

    print(
        f"Questions loaded: "
        f"{len(rows)}"
    )

    if len(rows) != 30:

        raise ValueError(
            f"Expected 30 golden questions, "
            f"found {len(rows)}."
        )

    print(
        "\nLoading grounded QA system..."
    )

    qa = answer_question

    results = []

    for index, row in enumerate(
        rows,
        start=1,
    ):

        print(
            f"\n[{index}/30] "
            f"{row['question_id']}: "
            f"{row['question']}"
        )

        result = evaluate_question(
            qa,
            row,
        )

        results.append(
            result
        )

        print(
            f"Answer: "
            f"{result['generated_answer']}"
        )

        print(
            f"Answer correct: "
            f"{result['answer_correct']}"
        )

        print(
            f"Refusal correct: "
            f"{result['refusal_correct']}"
        )

        print(
            f"Citation correct: "
            f"{result['citation_correct']}"
        )

        print(
            f"Latency: "
            f"{result['latency_ms']:.2f} ms"
        )

    fieldnames = [
        "question_id",
        "question",
        "gold_answer",
        "expected_source_file",
        "expected_source_page",
        "answerability",
        "generated_answer",
        "refused",
        "evidence_sufficient",
        "citation_valid",
        "citation_correct",
        "source_retrieved_in_top5",
        "answer_correct",
        "refusal_correct",
        "latency_ms",
    ]

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print_summary(
        results
    )

    print(
        f"\nDetailed results saved to:\n"
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
