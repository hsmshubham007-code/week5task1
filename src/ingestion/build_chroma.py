from __future__ import annotations

import json
import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"

EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNK_SIZES = (250, 500, 1000)

# Small batches keep memory usage reasonable on CPU systems.
BATCH_SIZE = 8


def load_chunks(chunk_size: int) -> list[dict]:
    """Load chunk records for one chunk-size configuration."""

    path = CHUNKS_DIR / str(chunk_size) / "chunks.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc

            required_fields = {
                "chunk_id",
                "source_file",
                "page",
                "section",
                "chunk_size",
                "chunk_index",
                "token_count",
                "text",
            }

            missing = required_fields - record.keys()

            if missing:
                raise ValueError(
                    f"Missing fields {sorted(missing)} "
                    f"in {path} at line {line_number}"
                )

            records.append(record)

    return records


def build_collection(
    client: chromadb.PersistentClient,
    model: SentenceTransformer,
    chunk_size: int,
) -> None:
    """Create one Chroma collection for one chunk-size configuration."""

    collection_name = f"rag_chunks_{chunk_size}"

    print()
    print("=" * 70)
    print(f"BUILDING COLLECTION: {collection_name}")
    print("=" * 70)

    records = load_chunks(chunk_size)

    print(f"Chunk records loaded: {len(records)}")

    # Rebuild the collection so the script is safely repeatable.
    try:
        client.delete_collection(collection_name)
        print(f"Removed existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "chunk_size": chunk_size,
            "embedding_model": EMBEDDING_MODEL,
            "description": f"RAG chunks using approximately {chunk_size}-token chunks",
        },
    )

    texts = [record["text"] for record in records]
    ids = [record["chunk_id"] for record in records]

    metadatas = [
        {
            "source_file": record["source_file"],
            "page": int(record["page"]),
            "section": record["section"],
            "chunk_size": int(record["chunk_size"]),
            "chunk_index": int(record["chunk_index"]),
            "token_count": int(record["token_count"]),
        }
        for record in records
    ]

    print("Generating BGE-M3 embeddings...")
    print(f"Batch size: {BATCH_SIZE}")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(f"Embeddings generated: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    print("Writing records to Chroma...")

    for start in range(0, len(records), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(records))

        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

        print(f"  Added records {start + 1}-{end} of {len(records)}")

    stored_count = collection.count()

    print()
    print(f"Collection created: {collection_name}")
    print(f"Records stored: {stored_count}")

    if stored_count != len(records):
        raise RuntimeError(
            f"Count mismatch for {collection_name}: "
            f"expected {len(records)}, got {stored_count}"
        )


def main() -> None:
    print("=" * 70)
    print("DAY 2 - CHROMA VECTOR DATABASE BUILD")
    print("=" * 70)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chunk configurations: {CHUNK_SIZES}")
    print(f"Chroma directory: {CHROMA_DIR}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Loading BGE-M3...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("BGE-M3 loaded successfully.")
    print(f"Maximum sequence length: {model.max_seq_length}")
    print(f"Embedding dimension: {model.get_embedding_dimension()}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    print()
    print("Building collections...")

    for chunk_size in CHUNK_SIZES:
        build_collection(
            client=client,
            model=model,
            chunk_size=chunk_size,
        )

    print()
    print("=" * 70)
    print("CHROMA BUILD COMPLETE")
    print("=" * 70)

    collections = client.list_collections()

    print("Available collections:")

    for collection in collections:
        print(f"  - {collection.name}: {collection.count()} records")

    print()
    print(f"Persistent storage: {CHROMA_DIR}")


if __name__ == "__main__":
    main()