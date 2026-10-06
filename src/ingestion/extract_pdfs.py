from __future__ import annotations

import json
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "extracted_pages.jsonl"
ERROR_FILE = PROCESSED_DIR / "extraction_errors.json"


def clean_text(text: str) -> str:
    """Normalize extracted PDF text while preserving readable content."""
    lines = []

    for line in text.splitlines():
        cleaned = " ".join(line.split())

        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines)


def detect_section(text: str) -> str:
    """
    Try to identify a simple section heading from the extracted page text.

    The synthetic corpus uses one-page policy documents, so the first
    non-empty line is used as the section/title when available.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return "Unknown"

    first_line = lines[0]

    if len(first_line) <= 150:
        return first_line

    return "Unknown"


def extract_pdf(pdf_path: Path) -> tuple[list[dict], list[dict]]:
    """Extract every page from one PDF and return records and errors."""
    records = []
    errors = []

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        errors.append(
            {
                "source_file": pdf_path.name,
                "page": None,
                "error_type": "open_error",
                "error": str(exc),
            }
        )

        return records, errors

    try:
        for page_number, page in enumerate(document, start=1):
            try:
                raw_text = page.get_text("text")
                text = clean_text(raw_text)

                if not text:
                    errors.append(
                        {
                            "source_file": pdf_path.name,
                            "page": page_number,
                            "error_type": "no_text_extracted",
                            "error": (
                                "No text was extracted from this page. "
                                "The page may be scanned/image-only or empty."
                            ),
                        }
                    )
                    continue

                section = detect_section(text)

                records.append(
                    {
                        "source_file": pdf_path.name,
                        "page": page_number,
                        "section": section,
                        "text": text,
                    }
                )

            except Exception as exc:
                errors.append(
                    {
                        "source_file": pdf_path.name,
                        "page": page_number,
                        "error_type": "page_extraction_error",
                        "error": str(exc),
                    }
                )

    finally:
        document.close()

    return records, errors


def main() -> None:
    """Extract all PDFs in the corpus and save page-level JSONL output."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {DOCUMENTS_DIR}"
        )

    all_records = []
    all_errors = []

    print("=" * 70)
    print("DAY 2 - PDF EXTRACTION")
    print("=" * 70)
    print(f"Documents directory : {DOCUMENTS_DIR}")
    print(f"PDF files found     : {len(pdf_files)}")
    print()

    for index, pdf_path in enumerate(pdf_files, start=1):
        records, errors = extract_pdf(pdf_path)

        all_records.extend(records)
        all_errors.extend(errors)

        print(
            f"[{index:02d}/{len(pdf_files):02d}] "
            f"{pdf_path.name} -> "
            f"{len(records)} page(s) extracted"
        )

        if errors:
            print(f"    Warnings/errors: {len(errors)}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in all_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ERROR_FILE.open("w", encoding="utf-8") as file:
        json.dump(all_errors, file, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"PDF files processed : {len(pdf_files)}")
    print(f"Pages extracted     : {len(all_records)}")
    print(f"Extraction issues   : {len(all_errors)}")
    print()
    print(f"Extracted data      : {OUTPUT_FILE}")
    print(f"Error report        : {ERROR_FILE}")
    print()

    if all_errors:
        print("Extraction issues were recorded for review.")
    else:
        print("No extraction issues detected.")


if __name__ == "__main__":
    main()