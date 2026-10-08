from __future__ import annotations

import time

import streamlit as st

from src.generation.grounded_qa import answer_question


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NovaTech Policy Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================================
# PAGE HEADER
# ============================================================================

st.title("📚 NovaTech Solutions Policy Assistant")

st.markdown(
    """
Ask questions about the NovaTech Solutions internal company policies.

The assistant answers **only from the retrieved policy documents**.
If the documents do not provide sufficient evidence, it will refuse to answer.
"""
)

st.divider()


# ============================================================================
# SESSION STATE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def display_sources(result: dict) -> None:
    """Display the retrieved source documents."""

    retrieved = result.get("retrieved", [])

    with st.expander(
        "📄 Retrieved sources",
        expanded=False,
    ):
        if not retrieved:
            st.write("No documents were retrieved.")
            return

        st.write("Top retrieved policy chunks:")

        for document in retrieved:
            source_file = document.source_file
            page = document.page
            section = document.section
            distance = document.distance
            rank = document.rank

            st.markdown(
                f"**#{rank} — {source_file} — page {page}**"
            )

            if section:
                st.caption(
                    f"Section: {section}"
                )

            st.caption(
                f"Vector distance: {distance:.4f}"
            )

            st.markdown(
                f"> {document.text}"
            )

            st.divider()


def display_evidence(result: dict) -> None:
    """Display the evidence selected for the final answer."""

    evidence = result.get(
        "selected_evidence",
        "",
    )

    score = result.get(
        "evidence_score",
        0.0,
    )

    sufficient = result.get(
        "evidence_sufficient",
        False,
    )

    with st.expander(
        "🔎 Selected evidence",
        expanded=False,
    ):
        st.metric(
            "Evidence score",
            f"{score:.4f}",
        )

        st.write(
            f"Evidence sufficient: **{sufficient}**"
        )

        if evidence:
            st.markdown(
                f"> {evidence}"
            )
        else:
            st.info(
                "No sufficient evidence was selected."
            )


def display_result(result: dict) -> None:
    """Display the answer, citation, and supporting evidence."""

    answer = result.get(
        "answer",
        "",
    )

    refused = result.get(
        "refused",
        False,
    )

    citation_valid = result.get(
        "citation_valid",
        False,
    )

    source_file = result.get(
        "source_file",
        "",
    )

    source_page = result.get(
        "source_page",
        "",
    )

    if refused:
        st.warning(
            "⚠️ The assistant refused to answer because "
            "the available company documents do not provide "
            "sufficient evidence."
        )
    else:
        st.success(answer)

        if source_file:
            st.caption(
                f"Source: {source_file}, page {source_page}"
            )

        if citation_valid:
            st.caption(
                "✅ Citation verified against the selected evidence."
            )

    display_sources(result)

    if not refused:
        display_evidence(result)


# ============================================================================
# PREVIOUS CHAT HISTORY
# ============================================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":
            st.markdown(
                message["content"]
            )

        else:
            display_result(
                message["result"]
            )


# ============================================================================
# CHAT INPUT
# ============================================================================

question = st.chat_input(
    "Ask a question about the company policies..."
)


# ============================================================================
# PROCESS QUESTION
# ============================================================================

if question:

    question = question.strip()

    if not question:
        st.stop()

    # ------------------------------------------------------------------------
    # Display user message
    # ------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # ------------------------------------------------------------------------
    # Run grounded QA
    # ------------------------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching company policies and generating a grounded answer..."
        ):

            start_time = time.perf_counter()

            try:

                result = answer_question(
                    question
                )

                elapsed_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000

            except Exception as exc:

                elapsed_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000

                st.error(
                    "The policy assistant encountered an error."
                )

                with st.expander(
                    "Technical error details",
                    expanded=False,
                ):
                    st.code(
                        f"{type(exc).__name__}: {exc}"
                    )

                st.stop()

        # --------------------------------------------------------------------
        # Display result
        # --------------------------------------------------------------------

        display_result(
            result
        )

        st.caption(
            f"Response time: "
            f"{elapsed_ms / 1000:.2f} seconds"
        )

    # ------------------------------------------------------------------------
    # Save assistant result to conversation history
    # ------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "result": result,
        }
    )