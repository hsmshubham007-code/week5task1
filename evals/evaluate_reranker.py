from __future__ import annotations

import csv
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_SET = PROJECT_ROOT / "data" / "golden_set" / "golden_set.csv"
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 250
VECTOR_CANDIDATES = 20
TOP_K = 5


def load_answerable_questions() -> list[dict]:
    """Load the 25 answerable questions from the golden set."""

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
            if row["answerability"].strip().lower() == "answerable":
                questions.append(row)

    if len(questions) != 25:
        raise ValueError(
            f"Expected 25 answerable questions, found {len(questions)}"
        )

    return questions


def normalize_source(value: str) -> str:
    """Normalize a source filename for comparison."""

    return Path(value.strip()).name.lower()


def is_correct_hit(
    metadata: dict,
    gold_source: str,
    gold_page: int,
) -> bool:
    """Check whether a result matches the gold source and page."""

    retrieved_source = normalize_source(
        str(metadata.get("source_file", ""))
    )

    retrieved_page = int(metadata.get("page", -1))

    return (
        retrieved_source == normalize_source(gold_source)
        and retrieved_page == gold_page
    )


def retrieve_vector_candidates(
    collection,
    embedding_model: SentenceTransformer,
    question: str,
) -> list[dict]:
    """Retrieve the top vector candidates from Chroma."""

    embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    result = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=min(VECTOR_CANDIDATES, collection.count()),
        include=["metadatas", "documents", "distances"],
    )

    candidates = []

    for metadata, document, distance in zip(
        result["metadatas"][0],
        result["documents"][0],
        result["distances"][0],
    ):
        candidates.append(
            {
                "document": document,
                "metadata": metadata,
                "vector_distance": float(distance),
            }
        )

    return candidates


def rerank_candidates(
    question: str,
    candidates: list[dict],
    reranker: CrossEncoder,
) -> list[dict]:
    """Score vector candidates with a cross-encoder and rerank them."""

    pairs = [
        (question, candidate["document"])
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = []

    for candidate, score in zip(candidates, scores):
        reranked.append(
            {
                **candidate,
                "reranker_score": float(score),
            }
        )

    reranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return reranked[:TOP_K]


def evaluate(
    collection,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    questions: list[dict],
) -> tuple[float, list[dict]]:
    """Evaluate vector retrieval followed by cross-encoder reranking."""

    hits = 0
    details = []

    for index, row in enumerate(questions, start=1):
        print(
            f"Evaluating {index}/{len(questions)}: "
            f"{row['question_id']}"
        )

        candidates = retrieve_vector_candidates(
            collection=collection,
            embedding_model=embedding_model,
            question=row["question"],
        )

        reranked = rerank_candidates(
            question=row["question"],
            candidates=candidates,
            reranker=reranker,
        )

        gold_source = row["source_file"].strip()
        gold_page = int(row["source_page"])

        matched = False
        retrieved = []

        for rank, result in enumerate(reranked, start=1):
            metadata = result["metadata"]

            source_file = str(metadata.get("source_file", ""))
            page = int(metadata.get("page", -1))
            chunk_index = int(metadata.get("chunk_index", -1))
            score = result["reranker_score"]

            if is_correct_hit(
                metadata=metadata,
                gold_source=gold_source,
                gold_page=gold_page,
            ):
                matched = True

            retrieved.append(
                f"{rank}. {source_file} p.{page} "
                f"chunk={chunk_index} "
                f"score={score:.4f}"
            )

        if matched:
            hits += 1

        details.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "gold_source": gold_source,
                "gold_page": gold_page,
                "hit_at_5": matched,
                "retrieved": " | ".join(retrieved),
            }
        )

    recall_at_5 = hits / len(questions)

    return recall_at_5, details


def save_results(
    recall_at_5: float,
    details: list[dict],
) -> None:
    """Save reranker evaluation results."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "day3_reranker_summary.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "chunk_size",
                "reranker_model",
                "vector_candidates",
                "questions_evaluated",
                "hits_at_5",
                "recall_at_5",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "experiment": "Vector + cross-encoder reranker",
                "chunk_size": CHUNK_SIZE,
                "reranker_model": RERANKER_MODEL,
                "vector_candidates": VECTOR_CANDIDATES,
                "questions_evaluated": len(details),
                "hits_at_5": sum(
                    1
                    for detail in details
                    if detail["hit_at_5"]
                ),
                "recall_at_5": f"{recall_at_5:.4f}",
            }
        )

    detail_path = RESULTS_DIR / "reranker_question_results.csv"

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
    print(f"Summary saved to: {summary_path}")
    print(f"Question results saved to: {detail_path}")


def main() -> None:
    print("=" * 70)
    print("DAY 3 - CROSS-ENCODER RERANKER")
    print("=" * 70)

    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Reranker model: {RERANKER_MODEL}")
    print(f"Vector candidates: {VECTOR_CANDIDATES}")
    print(f"Final top K: {TOP_K}")

    questions = load_answerable_questions()

    print()
    print(f"Answerable questions: {len(questions)}")

    print()
    print("Loading BGE-M3...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("BGE-M3 loaded successfully.")

    print()
    print("Loading cross-encoder reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("Cross-encoder loaded successfully.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection_name = f"rag_chunks_{CHUNK_SIZE}"

    print()
    print(f"Opening collection: {collection_name}")

    collection = client.get_collection(collection_name)

    print(f"Collection records: {collection.count()}")

    print()
    print("Running reranker evaluation...")

    recall_at_5, details = evaluate(
        collection=collection,
        embedding_model=embedding_model,
        reranker=reranker,
        questions=questions,
    )

    hits = sum(
        1
        for detail in details
        if detail["hit_at_5"]
    )

    print()
    print("=" * 70)
    print("CROSS-ENCODER RERANKER RESULTS")
    print("=" * 70)
    print(f"Questions:    {len(questions)}")
    print(f"Hits@5:       {hits}/{len(questions)}")
    print(f"Recall@5:     {recall_at_5:.4f}")
    print(f"Recall@5:     {recall_at_5 * 100:.2f}%")

    save_results(
        recall_at_5=recall_at_5,
        details=details,
    )


if __name__ == "__main__":
    main()