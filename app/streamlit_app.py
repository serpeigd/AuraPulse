"""Streamlit demo UI for the Hito 1 pipeline: classify -> route -> draft/escalate -> report.

Exists because `scripts/run_pipeline.py` only produces terminal text --
nobody reviewing this project is going to clone the repo, install
Ollama, and read stdout. This renders the same routing/aggregation
logic as an interactive page, plus the thing a script fundamentally
can't do: a human rejecting a draft, having it regenerated with their
feedback, and rejecting again -- the reject/regenerate loop, driven by
`aurapulse.draft_graph`'s LangGraph state machine (see docs/DESIGN.md
for why this loop specifically was the trigger condition for
introducing a graph orchestrator, when plain routing wasn't).

Deliberately does NOT reuse `orchestrator.process_reviews()` for
DRAFT_RESPONSE reviews -- that function eagerly generates one draft per
review in a single pass, which would mean generating a draft twice (once
via process_reviews, once via the graph's own first `generate_draft`
node) for no reason. Instead this does its own thin per-review routing
loop, reusing `routing.decide_route`, `response_draft.flag_for_escalation`,
and `aggregation.aggregate_reviews` directly -- all pure/deterministic,
none of them the thing that changed. `run_pipeline.py` (the
non-interactive script) is untouched and still uses `process_reviews()`
as before; the interactive loop only makes sense with a human present
to click Approve/Reject.

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

import logging

import streamlit as st
from langgraph.types import Command, RunnableConfig

from aurapulse.aggregation import (
    BusinessReport,
    aggregate_reviews,
    summarize_other_aspect_usage,
)
from aurapulse.classifier import ClassificationError
from aurapulse.draft_decisions import record_draft_decision
from aurapulse.draft_graph import build_draft_graph, draft_outcome, initial_state
from aurapulse.pipeline_io import (
    DEFAULT_INPUT_PATH,
    DEFAULT_REVIEWS_PATH,
    load_demo_data,
    load_real_data,
)
from aurapulse.response_draft import flag_for_escalation
from aurapulse.routing import decide_route
from aurapulse.schemas import (
    DraftDecision,
    EscalationFlag,
    ReviewAnalysis,
    Route,
    Sentiment,
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title="AuraPulse", page_icon="📋", layout="wide")


def _init_state() -> None:
    st.session_state.setdefault("graph", None)
    st.session_state.setdefault("business_reports", None)
    st.session_state.setdefault("escalations", [])
    st.session_state.setdefault("draft_review_ids", [])
    st.session_state.setdefault("draft_failures", [])
    st.session_state.setdefault("analyses", [])
    st.session_state.setdefault("review_texts", {})


def _graph_config(review_id: str) -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": review_id})


def _run_pipeline(use_demo: bool) -> None:
    """Load data, route every review, and kick off the draft graph for DRAFT_RESPONSE ones."""
    if use_demo:
        analyses, review_texts = load_demo_data()
    else:
        analyses, review_texts = load_real_data(DEFAULT_INPUT_PATH, DEFAULT_REVIEWS_PATH)

    if not analyses:
        st.error("No classified reviews to process.")
        return

    graph = build_draft_graph()
    escalations: list[EscalationFlag] = []
    draft_review_ids: list[str] = []
    draft_failures: list[str] = []

    with st.spinner("Routing reviews (draft generation calls the local Ollama model, this can take a while)..."):
        for analysis in analyses:
            route = decide_route(analysis)
            if route == Route.AGGREGATE:
                continue
            if route == Route.ESCALATE:
                escalations.append(flag_for_escalation(analysis))
                continue
            # route == Route.DRAFT_RESPONSE: run the graph's first generate_draft ->
            # human_review, which pauses (interrupt()) once the first draft is ready.
            text = review_texts[analysis.review_id]
            try:
                graph.invoke(
                    initial_state(analysis.review_id, analysis.business_id, text, analysis),
                    config=_graph_config(analysis.review_id),
                )
            except ClassificationError as exc:
                draft_failures.append(analysis.review_id)
                logger.warning("review_id=%s: draft generation failed, skipping: %s", analysis.review_id, exc)
                continue
            draft_review_ids.append(analysis.review_id)

        business_reports = aggregate_reviews(analyses)

    st.session_state["graph"] = graph
    st.session_state["business_reports"] = business_reports
    st.session_state["escalations"] = escalations
    st.session_state["draft_review_ids"] = draft_review_ids
    st.session_state["draft_failures"] = draft_failures
    st.session_state["analyses"] = analyses
    st.session_state["review_texts"] = review_texts


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


def _render_business_reports(business_reports: list[BusinessReport]) -> None:
    st.header("Business reports")
    for report in business_reports:
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


def _render_draft(review_id: str, review_text: str) -> None:
    """Render one review's current draft-graph state and, if still pending, its Approve/Reject form.

    Reads live state via ``graph.get_state`` rather than a snapshot passed
    in -- after a resume, the graph may already be sitting on a *new*
    draft (a regeneration), or on a terminal outcome, and this always
    reflects whichever is current.
    """
    graph = st.session_state["graph"]
    config = _graph_config(review_id)
    values = graph.get_state(config).values
    business_id = values["business_id"]
    attempt, max_attempts = values["attempt"], values["max_attempts"]
    outcome = draft_outcome(values)

    st.markdown(f"**[{review_id}]** ({business_id}) — draft {attempt}/{max_attempts}")
    st.caption(f"Original review: {review_text}")
    st.info(values["draft_text"])

    if outcome == "approved":
        st.success("✅ Approved for human sending (logged, never auto-sent).")
        return
    if outcome == "needs_human_rewrite":
        st.error(
            f"❌ Rejected {max_attempts} time(s) in a row — the model isn't converging on this one. "
            "Needs a human-written reply instead of another regeneration."
        )
        return

    reject_label = "❌ Reject & regenerate" if attempt < max_attempts else "❌ Reject (last automatic attempt)"
    with st.form(key=f"decision::{review_id}::{attempt}"):
        feedback = st.text_input(
            "Optional note (why reject / what to change)", key=f"feedback::{review_id}::{attempt}"
        )
        col_approve, col_reject = st.columns(2)
        approve_clicked = col_approve.form_submit_button("✅ Approve")
        reject_clicked = col_reject.form_submit_button(reject_label)

    if not (approve_clicked or reject_clicked):
        return

    approved = approve_clicked
    record_draft_decision(
        DraftDecision(
            review_id=review_id, business_id=business_id, approved=approved, feedback=(feedback or None)
        )
    )
    spinner_msg = "Regenerating draft with your feedback..." if reject_clicked and attempt < max_attempts else None
    if spinner_msg:
        with st.spinner(spinner_msg):
            graph.invoke(Command(resume={"approved": approved, "feedback": feedback or None}), config=config)
    else:
        graph.invoke(Command(resume={"approved": approved, "feedback": feedback or None}), config=config)
    st.rerun()


def _render_drafts(review_ids: list[str], draft_failures: list[str], review_texts: dict[str, str]) -> None:
    st.header(f"Draft replies — human review required ({len(review_ids)})")
    if draft_failures:
        st.warning(
            f"Draft generation FAILED for {len(draft_failures)} review(s): "
            f"{draft_failures}. Ollama may be unreachable (try: `ollama serve`)."
        )
    if not review_ids:
        st.caption("No drafts generated for this batch.")
        return

    for review_id in review_ids:
        _render_draft(review_id, review_texts.get(review_id, "(text unavailable)"))
        st.divider()


def _render_escalations(escalations: list[EscalationFlag]) -> None:
    st.header(f"Escalations ({len(escalations)})")
    if not escalations:
        st.caption("No reviews escalated for this batch.")
        return
    st.dataframe(
        [{"review_id": e.review_id, "business_id": e.business_id, "reason": e.reason} for e in escalations],
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

    business_reports: list[BusinessReport] | None = st.session_state["business_reports"]
    if business_reports is None:
        st.title("AuraPulse")
        st.write("Choose a data source in the sidebar and click **Run pipeline** to see results.")
        return

    _render_business_reports(business_reports)
    _render_drafts(st.session_state["draft_review_ids"], st.session_state["draft_failures"], st.session_state["review_texts"])
    _render_escalations(st.session_state["escalations"])
    _render_other_aspect_summary(st.session_state["analyses"])


if __name__ == "__main__":
    main()
