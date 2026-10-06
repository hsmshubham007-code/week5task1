from __future__ import annotations

import re
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

TARGET_ADDITIONAL_WORDS = 350
SECTION_MARKER = "Additional Policy Administration"


def get_policy_name(filename: str) -> str:
    """Convert a PDF filename into a readable policy name."""
    name = Path(filename).stem

    name = re.sub(r"^[A-Z]+_", "", name)
    name = name.replace("_", " ")

    return name.strip()


def build_additional_content(filename: str) -> str:
    """Create realistic administrative content for corpus expansion."""
    policy_name = get_policy_name(filename)

    paragraphs = [
        (
            f"Policy Administration: The {policy_name} applies to employees "
            "and relevant personnel whose work is covered by this policy. "
            "Employees are expected to understand the requirements that "
            "apply to their responsibilities and follow the procedures "
            "defined by the company."
        ),
        (
            "Policy Responsibilities: Managers are responsible for "
            "communicating applicable requirements to team members and "
            "supporting consistent implementation. Employees should raise "
            "questions or request clarification through their normal "
            "management or administrative channel when a requirement is "
            "unclear."
        ),
        (
            "Records and Documentation: Where this policy requires a request, "
            "notification, approval, report, or other documented action, the "
            "relevant information should be complete and accurate. Company "
            "records associated with policy administration should be retained "
            "according to applicable internal record-handling requirements."
        ),
        (
            "Exceptions: Exceptions to this policy should not be assumed. "
            "Where an exception is necessary, the employee should obtain "
            "appropriate authorization through the responsible department "
            "before taking an action that would otherwise fall outside the "
            "normal procedure."
        ),
        (
            "Compliance: Employees are expected to comply with applicable "
            "company procedures and cooperate with reasonable administrative "
            "checks related to this policy. Repeated or deliberate failure "
            "to follow applicable requirements may be reviewed by the "
            "appropriate company function."
        ),
        (
            "Review and Updates: Company policies may be reviewed periodically "
            "to ensure that procedures remain appropriate for business needs. "
            "When an updated version is formally issued, employees should "
            "follow the current approved version and discontinue reliance on "
            "superseded instructions."
        ),
        (
            "Questions and Support: Questions concerning interpretation or "
            "implementation should be directed to the department responsible "
            "for maintaining the policy. Employees should provide sufficient "
            "context when requesting assistance so that the responsible "
            "team can provide an accurate response."
        ),
        (
            "Implementation Guidance: Normal business judgment should be used "
            "when applying this policy together with other applicable company "
            "requirements. If two requirements appear to conflict, employees "
            "should pause the affected action and seek clarification from "
            "the responsible department rather than making an unsupported "
            "assumption."
        ),
    ]

    text = "\n\n".join(paragraphs)

    while len(text.split()) < TARGET_ADDITIONAL_WORDS:
        text += (
            "\n\nAdministrative Note: This section provides general "
            "implementation guidance for the synthetic evaluation corpus. "
            "It does not replace or modify the numbered policy requirements "
            "stated earlier in the document. Employees should use the "
            "specific policy requirements as the authoritative rules for "
            "the subject covered by this document."
        )

    return text


def get_original_policy_text(page_text: str) -> str:
    """
    Recover the original policy text from a previously expanded document.

    Everything before the first expansion marker is treated as the
    authoritative original policy content.
    """
    if SECTION_MARKER in page_text:
        original_text = page_text.split(SECTION_MARKER, 1)[0].strip()
    else:
        original_text = page_text.strip()

    return original_text


def rewrite_pdf(pdf_path: Path) -> tuple[int, int]:
    """
    Rewrite the PDF using the original policy content plus exactly one
    expansion section.
    """
    document = pymupdf.open(pdf_path)

    try:
        if len(document) != 1:
            raise RuntimeError(
                f"{pdf_path.name} has {len(document)} pages. "
                "Expected exactly 1 page."
            )

        original_page = document[0]

        existing_text = original_page.get_text("text")
        original_text = get_original_policy_text(existing_text)

        if not original_text:
            raise RuntimeError(
                f"Could not recover original policy text from "
                f"{pdf_path.name}."
            )

        additional_text = build_additional_content(pdf_path.name)

        combined_text = (
            f"{original_text}\n\n"
            f"{SECTION_MARKER}\n\n"
            f"{additional_text}"
        )

        original_word_count = len(original_text.split())
        final_word_count = len(combined_text.split())

        page_width = original_page.rect.width
        page_height = original_page.rect.height

        document.delete_page(0)

        page = document.new_page(
            width=page_width,
            height=page_height,
        )

        text_rect = pymupdf.Rect(
            45,
            40,
            page_width - 45,
            page_height - 40,
        )

        inserted = page.insert_textbox(
            text_rect,
            combined_text,
            fontsize=7.5,
            fontname="helv",
            lineheight=1.15,
            align=pymupdf.TEXT_ALIGN_LEFT,
            color=(0, 0, 0),
        )

        if inserted < 0:
            raise RuntimeError(
                f"Expanded content did not fit on page 1 of "
                f"{pdf_path.name}. Overflow: {abs(inserted):.2f}"
            )

        temporary_path = pdf_path.with_suffix(".tmp.pdf")

        document.save(
            temporary_path,
            garbage=4,
            deflate=True,
        )

    finally:
        document.close()

    temporary_path.replace(pdf_path)

    return original_word_count, final_word_count


def main() -> None:
    """Regenerate all 30 PDFs from their original policy text."""
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if len(pdf_files) != 30:
        raise RuntimeError(
            f"Expected exactly 30 PDF files, found {len(pdf_files)}."
        )

    print("=" * 70)
    print("REGENERATING SYNTHETIC RAG CORPUS")
    print("=" * 70)
    print(f"Documents found: {len(pdf_files)}")
    print(
        "The original policy text will be preserved and only one "
        "expansion section will be added."
    )
    print()

    total_original_words = 0
    total_final_words = 0

    for index, pdf_path in enumerate(pdf_files, start=1):
        original_words, final_words = rewrite_pdf(pdf_path)

        total_original_words += original_words
        total_final_words += final_words

        print(
            f"[{index:02d}/30] {pdf_path.name}"
            f" | {original_words} -> {final_words} words"
        )

    print()
    print("=" * 70)
    print("CORPUS REGENERATION COMPLETE")
    print("=" * 70)
    print(f"Documents updated    : {len(pdf_files)}")
    print(f"Original total words : {total_original_words}")
    print(f"Final total words    : {total_final_words}")
    print()
    print(
        "Each PDF contains the original policy plus one expansion "
        "section, all on page 1."
    )


if __name__ == "__main__":
    main()