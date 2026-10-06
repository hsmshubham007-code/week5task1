from __future__ import annotations

import csv
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_SET = PROJECT_ROOT / "data" / "golden_set" / "golden_set.csv"
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZES = (250, 500, 1000)

TOP_K = 5


def load_answerable_questions() -> list[dict]:
    """Load only the 25 answerable questions from the golden set."""

    if not GOLDEN_SET.exists():
        raise FileNotFoundError(f"Golden set not found: {GOLDEN_SET}")

    questions = []

    with GOLDEN_SET.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "question_id",
            "question",
            "source_file",
            "source_page",
            "answerability",
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Golden set is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            if row["answerability"].strip().lower() != "answerable":
                continue

            questions.append(row)

    if len(questions) != 25:
        raise ValueError(
            f"Expected 25 answerable questions, found {len(questions)}"
        )

    return questions


def normalize_source(value: str) -> str:
    """Normalize source filename for safe comparison."""

    return Path(value.strip()).name.lower()


def is_correct_hit(result_metadata: dict, gold_source: str, gold_page: int) -> bool:
    """Check whether a retrieved result matches the gold source and page."""

    retrieved_source = normalize_source(
        str(result_metadata.get("source_file", ""))
    )

    retrieved_page = int(result_metadata.get("page", -1))

    return (
        retrieved_source == normalize_source(gold_source)
        and retrieved_page == gold_page
    )


def evaluate_collection(
    collection,
    model: SentenceTransformer,
    questions: list[dict],
    chunk_size: int,
) -> tuple[float, list[dict]]:
    """Evaluate one chunk-size collection using Recall@5."""

    question_embeddings = model.encode(
        [row["question"] for row in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    hits = 0
    details = []

    for row, embedding in zip(questions, question_embeddings):
        gold_source = row["source_file"].strip()
        gold_page = int(row["source_page"])

        result = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=TOP_K,
            include=["metadatas", "documents", "distances"],
        )

        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        matched = False
        retrieved_sources = []

        for metadata, distance in zip(metadatas, distances):
            source_file = str(metadata.get("source_file", ""))
            page = int(metadata.get("page", -1))

            retrieved_sources.append(
                f"{source_file} p.{page} (distance={distance:.4f})"
            )

            if is_correct_hit(metadata, gold_source, gold_page):
                matched = True

        if matched:
            hits += 1

        details.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "gold_source": gold_source,
                "gold_page": gold_page,
                "hit_at_5": matched,
                "retrieved": " | ".join(retrieved_sources),
            }
        )

    recall_at_5 = hits / len(questions)

    return recall_at_5, details


def save_results(
    summary: list[dict],
    details_by_chunk: dict[int, list[dict]],
) -> None:
    """Save summary and question-level evaluation results."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "day3_recall_summary.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "chunk_size",
                "questions_evaluated",
                "hits_at_5",
                "recall_at_5",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    for chunk_size, details in details_by_chunk.items():
        detail_path = (
            RESULTS_DIR / f"recall_{chunk_size}_question_results.csv"
        )

        with detail_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "question_id",
                    "question",
                    "gold_source",
                    "gold_page",
                    "hit_at_5",
                    "retrieved",
                ],
            )
            writer.writeheader()
            writer.writerows(details)

    print()
    print(f"Results saved to: {RESULTS_DIR}")


def main() -> None:
    print("=" * 70)
    print("DAY 3 - BASELINE RETRIEVAL RECALL@5")
    print("=" * 70)

    print(f"Golden set: {GOLDEN_SET}")
    print(f"Chroma directory: {CHROMA_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Top K: {TOP_K}")

    questions = load_answerable_questions()

    print()
    print(f"Answerable questions loaded: {len(questions)}")

    print()
    print("Loading BGE-M3...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("BGE-M3 loaded successfully.")
    print(f"Maximum sequence length: {model.max_seq_length}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    summary = []
    details_by_chunk = {}

    for chunk_size in CHUNK_SIZES:
        collection_name = f"rag_chunks_{chunk_size}"

        print()
        print("=" * 70)
        print(f"EVALUATING: {collection_name}")
        print("=" * 70)

        try:
            collection = client.get_collection(collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not open Chroma collection: {collection_name}"
            ) from exc

        record_count = collection.count()

        print(f"Collection records: {record_count}")

        recall_at_5, details = evaluate_collection(
            collection=collection,
            model=model,
            questions=questions,
            chunk_size=chunk_size,
        )

        hits = sum(1 for detail in details if detail["hit_at_5"])

        print()
        print(f"Chunk size:       {chunk_size}")
        print(f"Questions:        {len(questions)}")
        print(f"Hits@5:           {hits}/{len(questions)}")
        print(f"Recall@5:         {recall_at_5:.4f}")
        print(f"Recall@5 (%):     {recall_at_5 * 100:.2f}%")

        summary.append(
            {
                "experiment": f"Vector baseline - {chunk_size} tokens",
                "chunk_size": chunk_size,
                "questions_evaluated": len(questions),
                "hits_at_5": hits,
                "recall_at_5": f"{recall_at_5:.4f}",
            }
        )

        details_by_chunk[chunk_size] = details

    save_results(summary, details_by_chunk)

    print()
    print("=" * 70)
    print("DAY 3 BASELINE RESULTS")
    print("=" * 70)

    for row in summary:
        print(
            f"{row['experiment']}: "
            f"{row['hits_at_5']}/{row['questions_evaluated']} "
            f"-> Recall@5 = {row['recall_at_5']}"
        )

    best = max(summary, key=lambda row: float(row["recall_at_5"]))

    print()
    print(
        f"BEST BASELINE: {best['chunk_size']}-token chunks "
        f"with Recall@5 = {best['recall_at_5']}"
    )


if __name__ == "__main__":
    main()