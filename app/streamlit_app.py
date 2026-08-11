"""Streamlit demo UI for the Hito 1 pipeline: classify -> route -> draft/escalate -> report.

Exists because `scripts/run_pipeline.py` only produces terminal text --
nobody reviewing this project is going to clone the repo, install
Ollama, and read stdout. This renders the exact same
`orchestrator.process_reviews()` output as an interactive page, and adds
one new thing scripts can't: a human approve/reject action on each
draft, logged via `draft_decisions.record_draft_decision`.

Scope, deliberately: "Reject" only records the decision (with an
optional note). It does NOT regenerate the draft -- there is no
reject-then-regenerate loop wired up yet. See docs/DESIGN.md's
LangGraph entry: that loop is the concrete trigger condition for
introducing a graph orchestrator, and this UI is what would drive it,
but building the loop itself is separate, not-yet-started work.

Per CLAUDE.md's non-negotiable rule, nothing here can publish a reply
either -- "Approve" only marks a draft as human-reviewed in the local
log, exactly like "Reject" does; there is still no send/publish
capability anywhere in this codebase.

Run with:
    pip install -e ".[ui]"
    python -m streamlit run app/streamlit_app.py

(``python -m streamlit`` rather than the bare ``streamlit`` command --
pip's install location for the console script isn't always on PATH,
especially on Windows; see README.md's Usage section.)
"""

from __future__ import annotations

import streamlit as st

from aurapulse.aggregation import summarize_other_aspect_usage
from aurapulse.draft_decisions import record_draft_decision
from aurapulse.orchestrator import Hito1Batch, process_reviews
from aurapulse.pipeline_io import (
    DEFAULT_INPUT_PATH,
    DEFAULT_REVIEWS_PATH,
    load_demo_data,
    load_real_data,
)
from aurapulse.schemas import DraftDecision, DraftResponse, ReviewAnalysis, Sentiment

st.set_page_config(page_title="AuraPulse", page_icon="📋", layout="wide")


def _init_state() -> None:
    st.session_state.setdefault("batch", None)
    st.session_state.setdefault("analyses", [])
    st.session_state.setdefault("review_texts", {})
    st.session_state.setdefault("decisions", {})  # review_id -> "approved" | "rejected"


def _run_pipeline(use_demo: bool) -> None:
    """Load data, run process_reviews(), stash results in session state."""
    if use_demo:
        analyses, review_texts = load_demo_data()
    else:
        analyses, review_texts = load_real_data(DEFAULT_INPUT_PATH, DEFAULT_REVIEWS_PATH)

    if not analyses:
        st.error("No classified reviews to process.")
        return

    with st.spinner("Routing reviews (draft generation calls the local Ollama model, this can take a while)..."):
        batch = process_reviews(analyses, review_texts)

    st.session_state["batch"] = batch
    st.session_state["analyses"] = analyses
    st.session_state["review_texts"] = review_texts
    st.session_state["decisions"] = {}


def _render_sidebar() -> None:
    st.sidebar.title("AuraPulse")
    st.sidebar.caption(
        "Zero-cost, local-LLM pipeline: classify -> route -> draft/escalate -> report. "
        "Never publishes a reply -- drafts are always for human review."
    )
    use_demo = st.sidebar.radio(
        "Data source",
        ["Demo dataset (8 deterministic fake reviews)", "Real classified data"],
        index=0,
    ) == "Demo dataset (8 deterministic fake reviews)"

    if not use_demo:
        st.sidebar.caption(f"Reads `{DEFAULT_INPUT_PATH}` + `{DEFAULT_REVIEWS_PATH}`.")

    if st.sidebar.button("Run pipeline", type="primary"):
        try:
            _run_pipeline(use_demo)
        except FileNotFoundError as exc:
            st.sidebar.error(str(exc))

    st.sidebar.caption("Requires a running local Ollama server (`ollama serve`) -- draft generation always calls it.")


def _render_business_reports(batch: Hito1Batch) -> None:
    st.header("Business reports")
    for report in batch.business_reports:
        with st.expander(f"{report.business_id} ({report.review_count} reviews)", expanded=True):
            cols = st.columns(3)
            cols[0].metric("Positive", report.sentiment_counts.get(Sentiment.POSITIVE, 0))
            cols[1].metric("Neutral", report.sentiment_counts.get(Sentiment.NEUTRAL, 0))
            cols[2].metric("Negative", report.sentiment_counts.get(Sentiment.NEGATIVE, 0))

            if report.aspect_summaries:
                st.dataframe(
                    [
                        {
                            "aspect": s.aspect.value,
                            "mentions": s.total_mentions,
                            "positive": s.positive_count,
                            "neutral": s.neutral_count,
                            "negative": s.negative_count,
                            "% negative": f"{s.negative_share:.0%}",
                        }
                        for s in report.aspect_summaries
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No aspects mentioned.")

            for flag in report.inconsistent_aspects:
                st.warning(f"⚠️ {flag}")


def _decision_key(review_id: str) -> str:
    return f"decision::{review_id}"


def _render_draft(draft: DraftResponse, review_text: str) -> None:
    decisions: dict[str, str] = st.session_state["decisions"]
    decided = decisions.get(draft.review_id)

    st.markdown(f"**[{draft.review_id}]** ({draft.business_id})")
    st.caption(f"Original review: {review_text}")
    st.info(draft.draft_text)

    if decided == "approved":
        st.success("✅ Approved for human sending (logged, never auto-sent).")
        return
    if decided == "rejected":
        st.error("❌ Rejected (logged).")
        return

    with st.form(key=_decision_key(draft.review_id)):
        feedback = st.text_input("Optional note (why reject / what to change)", key=f"feedback::{draft.review_id}")
        col_approve, col_reject = st.columns(2)
        approve_clicked = col_approve.form_submit_button("✅ Approve")
        reject_clicked = col_reject.form_submit_button("❌ Reject")

    if approve_clicked or reject_clicked:
        approved = approve_clicked
        record_draft_decision(
            DraftDecision(
                review_id=draft.review_id,
                business_id=draft.business_id,
                approved=approved,
                feedback=feedback or None,
            )
        )
        decisions[draft.review_id] = "approved" if approved else "rejected"
        st.rerun()


def _render_drafts(batch: Hito1Batch, review_texts: dict[str, str]) -> None:
    st.header(f"Draft replies — human review required ({len(batch.drafts)})")
    if batch.draft_failures:
        st.warning(
            f"Draft generation FAILED for {len(batch.draft_failures)} review(s): "
            f"{batch.draft_failures}. Ollama may be unreachable (try: `ollama serve`)."
        )
    if not batch.drafts:
        st.caption("No drafts generated for this batch.")
        return

    for draft in batch.drafts:
        _render_draft(draft, review_texts.get(draft.review_id, "(text unavailable)"))
        st.divider()


def _render_escalations(batch: Hito1Batch) -> None:
    st.header(f"Escalations ({len(batch.escalations)})")
    if not batch.escalations:
        st.caption("No reviews escalated for this batch.")
        return
    st.dataframe(
        [{"review_id": e.review_id, "business_id": e.business_id, "reason": e.reason} for e in batch.escalations],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Escalations are also appended to `data/escalations/escalations.jsonl` by "
        "`scripts/run_pipeline.py` -- this view only reflects the current in-app run."
    )


def _render_other_aspect_summary(analyses: list[ReviewAnalysis]) -> None:
    summary = summarize_other_aspect_usage(analyses)
    st.header("Aspect enum coverage")
    st.write(f"`other` used in {summary.other_mentions}/{summary.total_mentions} aspect mentions "
             f"({summary.other_share:.1%}).")
    if summary.other_details:
        with st.expander("What fell through to `other`"):
            for detail in summary.other_details[:20]:
                st.write(f"- {detail}")


def main() -> None:
    """Entry point for `streamlit run app/streamlit_app.py`."""
    _init_state()
    _render_sidebar()

    batch: Hito1Batch | None = st.session_state["batch"]
    if batch is None:
        st.title("AuraPulse")
        st.write("Choose a data source in the sidebar and click **Run pipeline** to see results.")
        return

    _render_business_reports(batch)
    _render_drafts(batch, st.session_state["review_texts"])
    _render_escalations(batch)
    _render_other_aspect_summary(st.session_state["analyses"])


if __name__ == "__main__":
    main()
