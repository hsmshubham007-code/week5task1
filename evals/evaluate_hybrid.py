from __future__ import annotations

import csv
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_SET = PROJECT_ROOT / "data" / "golden_set" / "golden_set.csv"
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 250
TOP_K = 5

# Retrieve a larger candidate pool from both methods before fusion.
VECTOR_CANDIDATES = 20
BM25_CANDIDATES = 20


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


def tokenize(text: str) -> list[str]:
    """Simple whitespace/token normalization for BM25."""

    return text.lower().split()


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


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to the 0-1 range."""

    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [0.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


def build_bm25_index(collection) -> tuple[list[dict], BM25Okapi]:
    """Load all Chroma documents and build a BM25 index."""

    count = collection.count()

    if count == 0:
        raise RuntimeError("Chroma collection is empty.")

    data = collection.get(
        include=["documents", "metadatas"],
    )

    documents = data["documents"]
    metadatas = data["metadatas"]

    records = []

    for document, metadata in zip(documents, metadatas):
        records.append(
            {
                "document": document,
                "metadata": metadata,
            }
        )

    tokenized_documents = [
        tokenize(record["document"])
        for record in records
    ]

    bm25 = BM25Okapi(tokenized_documents)

    return records, bm25


def retrieve_vector_candidates(
    collection,
    model: SentenceTransformer,
    question: str,
) -> list[dict]:
    """Retrieve vector candidates from Chroma."""

    embedding = model.encode(
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


def retrieve_bm25_candidates(
    records: list[dict],
    bm25: BM25Okapi,
    question: str,
) -> list[dict]:
    """Retrieve BM25 candidates."""

    scores = bm25.get_scores(tokenize(question))

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    ranked_indices = ranked_indices[:BM25_CANDIDATES]

    candidates = []

    for index in ranked_indices:
        candidates.append(
            {
                "document": records[index]["document"],
                "metadata": records[index]["metadata"],
                "bm25_score": float(scores[index]),
            }
        )

    return candidates


def hybrid_retrieve(
    vector_candidates: list[dict],
    bm25_candidates: list[dict],
) -> list[dict]:
    """
    Fuse vector and BM25 rankings using reciprocal rank fusion.

    RRF gives each retrieval method an independent ranking signal,
    avoiding incompatible raw-score scales.
    """

    RRF_K = 60

    fused = {}

    for rank, candidate in enumerate(vector_candidates, start=1):
        metadata = candidate["metadata"]

        key = (
            str(metadata.get("source_file", "")),
            int(metadata.get("page", -1)),
            int(metadata.get("chunk_index", -1)),
        )

        if key not in fused:
            fused[key] = {
                "document": candidate["document"],
                "metadata": metadata,
                "vector_rank": rank,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }

        fused[key]["rrf_score"] += 1.0 / (RRF_K + rank)

    for rank, candidate in enumerate(bm25_candidates, start=1):
        metadata = candidate["metadata"]

        key = (
            str(metadata.get("source_file", "")),
            int(metadata.get("page", -1)),
            int(metadata.get("chunk_index", -1)),
        )

        if key not in fused:
            fused[key] = {
                "document": candidate["document"],
                "metadata": metadata,
                "vector_rank": None,
                "bm25_rank": rank,
                "rrf_score": 0.0,
            }

        else:
            fused[key]["bm25_rank"] = rank

        fused[key]["rrf_score"] += 1.0 / (RRF_K + rank)

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked[:TOP_K]


def evaluate(
    collection,
    records: list[dict],
    bm25: BM25Okapi,
    model: SentenceTransformer,
    questions: list[dict],
) -> tuple[float, list[dict]]:
    """Evaluate hybrid retrieval using Recall@5."""

    hits = 0
    details = []

    for index, row in enumerate(questions, start=1):
        print(
            f"Evaluating {index}/{len(questions)}: "
            f"{row['question_id']}"
        )

        vector_candidates = retrieve_vector_candidates(
            collection=collection,
            model=model,
            question=row["question"],
        )

        bm25_candidates = retrieve_bm25_candidates(
            records=records,
            bm25=bm25,
            question=row["question"],
        )

        hybrid_results = hybrid_retrieve(
            vector_candidates=vector_candidates,
            bm25_candidates=bm25_candidates,
        )

        gold_source = row["source_file"].strip()
        gold_page = int(row["source_page"])

        matched = False
        retrieved = []

        for rank, result in enumerate(hybrid_results, start=1):
            metadata = result["metadata"]

            source_file = str(metadata.get("source_file", ""))
            page = int(metadata.get("page", -1))
            chunk_index = int(metadata.get("chunk_index", -1))

            if is_correct_hit(
                metadata=metadata,
                gold_source=gold_source,
                gold_page=gold_page,
            ):
                matched = True

            retrieved.append(
                (
                    f"{rank}. {source_file} p.{page} "
                    f"chunk={chunk_index} "
                    f"rrf={result['rrf_score']:.6f}"
                )
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
    """Save hybrid evaluation results."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "day3_hybrid_summary.csv"

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

        writer.writerow(
            {
                "experiment": "BM25 hybrid - Reciprocal Rank Fusion",
                "chunk_size": CHUNK_SIZE,
                "questions_evaluated": len(details),
                "hits_at_5": sum(
                    1 for detail in details if detail["hit_at_5"]
                ),
                "recall_at_5": f"{recall_at_5:.4f}",
            }
        )

    detail_path = RESULTS_DIR / "hybrid_question_results.csv"

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
    print("DAY 3 - BM25 HYBRID RETRIEVAL")
    print("=" * 70)

    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Vector candidates: {VECTOR_CANDIDATES}")
    print(f"BM25 candidates: {BM25_CANDIDATES}")
    print(f"Final top K: {TOP_K}")

    questions = load_answerable_questions()

    print()
    print(f"Answerable questions: {len(questions)}")

    print()
    print("Loading BGE-M3...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection_name = f"rag_chunks_{CHUNK_SIZE}"

    print()
    print(f"Opening collection: {collection_name}")

    collection = client.get_collection(collection_name)

    print(f"Collection records: {collection.count()}")

    print()
    print("Building BM25 index...")

    records, bm25 = build_bm25_index(collection)

    print(f"BM25 documents indexed: {len(records)}")

    print()
    print("Running hybrid evaluation...")

    recall_at_5, details = evaluate(
        collection=collection,
        records=records,
        bm25=bm25,
        model=model,
        questions=questions,
    )

    hits = sum(
        1
        for detail in details
        if detail["hit_at_5"]
    )

    print()
    print("=" * 70)
    print("BM25 HYBRID RESULTS")
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