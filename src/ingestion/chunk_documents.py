from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "extracted_pages.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "chunks"

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZES = (250, 500, 1000)


def load_extracted_pages() -> list[dict]:
    """Load page-level records created by the PDF extraction step."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Extracted page file not found: {INPUT_FILE}\n"
            "Run extract_pdfs.py before running this script."
        )

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {INPUT_FILE}: {exc}"
                ) from exc

            records.append(record)

    return records


def split_text_by_tokens(
    text: str,
    tokenizer,
    chunk_size: int,
) -> list[str]:
    """
    Split text into chunks based on tokenizer token counts.

    The text is first tokenized without truncation. Each token window is
    then decoded back into text. Special tokens are not added because
    these chunks are intended to represent source-document content.
    """
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if not token_ids:
        return []

    chunks = []

    for start in range(0, len(token_ids), chunk_size):
        end = start + chunk_size
        chunk_token_ids = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def create_chunks(
    records: list[dict],
    tokenizer,
    chunk_size: int,
) -> list[dict]:
    """Create token-sized chunks while preserving source metadata."""
    chunks = []

    for record in records:
        source_file = record.get("source_file")
        page = record.get("page")
        section = record.get("section")
        text = record.get("text", "")

        if not source_file:
            raise ValueError("Record is missing required field: source_file")

        if page is None:
            raise ValueError(
                f"Record for {source_file} is missing required field: page"
            )

        if not text.strip():
            continue

        text_chunks = split_text_by_tokens(
            text=text,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
        )

        for chunk_index, chunk_text in enumerate(text_chunks):
            actual_token_count = len(
                tokenizer.encode(
                    chunk_text,
                    add_special_tokens=False,
                )
            )

            chunk_id = (
                f"{Path(source_file).stem}"
                f"_p{page}"
                f"_c{chunk_index:03d}"
                f"_s{chunk_size}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": source_file,
                    "page": page,
                    "section": section,
                    "chunk_size": chunk_size,
                    "chunk_index": chunk_index,
                    "token_count": actual_token_count,
                    "text": chunk_text,
                }
            )

    return chunks


def save_chunks(chunks: list[dict], chunk_size: int) -> Path:
    """Save one chunk-size dataset as JSONL."""
    output_dir = OUTPUT_DIR / str(chunk_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "chunks.jsonl"

    with output_file.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return output_file


def print_summary(
    chunks: list[dict],
    chunk_size: int,
    output_file: Path,
) -> None:
    """Print useful statistics for one chunk-size dataset."""
    if chunks:
        token_counts = [chunk["token_count"] for chunk in chunks]

        minimum = min(token_counts)
        maximum = max(token_counts)
        average = sum(token_counts) / len(token_counts)
    else:
        minimum = 0
        maximum = 0
        average = 0.0

    source_files = {
        chunk["source_file"]
        for chunk in chunks
    }

    print(f"\n{chunk_size}-TOKEN CHUNKS")
    print("-" * 50)
    print(f"Chunks created       : {len(chunks)}")
    print(f"Source documents     : {len(source_files)}")
    print(f"Minimum token count  : {minimum}")
    print(f"Maximum token count  : {maximum}")
    print(f"Average token count  : {average:.2f}")
    print(f"Output               : {output_file}")


def main() -> None:
    """Create and save all required chunk-size datasets."""
    print("=" * 70)
    print("DAY 2 - TOKEN-BASED DOCUMENT CHUNKING")
    print("=" * 70)
    print(f"Input file : {INPUT_FILE}")
    print(f"Tokenizer  : {TOKENIZER_NAME}")
    print(f"Chunk sizes: {', '.join(str(size) for size in CHUNK_SIZES)}")
    print()

    records = load_extracted_pages()

    print(f"Page records loaded: {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Tokenizer loaded successfully.")

    for chunk_size in CHUNK_SIZES:
        chunks = create_chunks(
            records=records,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
        )

        output_file = save_chunks(
            chunks=chunks,
            chunk_size=chunk_size,
        )

        print_summary(
            chunks=chunks,
            chunk_size=chunk_size,
            output_file=output_file,
        )

    print()
    print("=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)

    for chunk_size in CHUNK_SIZES:
        print(
            f"  {chunk_size}: "
            f"{OUTPUT_DIR / str(chunk_size) / 'chunks.jsonl'}"
        )


if __name__ == "__main__":
    main()