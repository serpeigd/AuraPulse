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

A third data source, "Instant demo", exists for the Streamlit Cloud
deployment specifically: a cloud container can't reach a local Ollama
server, so the live routes above simply don't work there. The frozen
snapshot (`app/frozen_demo.json`, produced by `scripts/freeze_demo_run.py`)
replays one real, previously-captured run instead -- see docs/DESIGN.md's
"Streamlit Cloud replay mode" entry for why this is a real run's actual
output, not invented placeholder text, and why it can't support the
reject/regenerate loop (nothing to regenerate against without a
reachable model).

Onboarding, added after the first deployed version proved confusing to
a first-time visitor: `_ollama_reachable()` cheaply probes whether a
local model is actually available and defaults the sidebar to whichever
data source will actually work here. When that resolves to the instant
demo (the cloud case), the pipeline auto-runs on page load -- a visitor
sees real results immediately instead of an empty page with a button to
find first. Live options never auto-run (they're slow, real model
calls); those still need an explicit click. `_run_disabled_reason()`
disables "Run pipeline" outright (not just a warning) whenever the
selected data source is known in advance to fail -- Ollama unreachable
for a live option, or a missing input file -- so nobody can click into
a broken run.

The instant demo's own tab, `_render_pipeline_walkthrough`, is its
flagship view: every one of the 8 sample reviews shown with its actual
sentiment/aspects, the route the pipeline picked for it, and why --
aggregated, drafted, or escalated -- not just the 5 that got a draft.

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

import json
import logging
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from langgraph.types import Command, RunnableConfig

from aurapulse.aggregation import (
    BusinessReport,
    OtherAspectSummary,
    aggregate_reviews,
    summarize_other_aspect_usage,
)
from aurapulse.classifier import DEFAULT_OLLAMA_HOST, ClassificationError
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
    Route,
    Sentiment,
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title="AuraPulse", page_icon="📋", layout="wide")

FROZEN_SNAPSHOT_PATH = Path(__file__).parent / "frozen_demo.json"

_FROZEN = "🧊 Instant demo"
_DEMO_LIVE = "⚡ Live demo (sample reviews)"
_REAL_LIVE = "📁 Live demo (your own data)"
_DATA_SOURCES = [_FROZEN, _DEMO_LIVE, _REAL_LIVE]


def _init_state() -> None:
    st.session_state.setdefault("mode", None)  # "live" | "frozen"
    st.session_state.setdefault("auto_run_done", False)
    st.session_state.setdefault("graph", None)
    st.session_state.setdefault("business_reports", None)
    st.session_state.setdefault("escalations", [])
    st.session_state.setdefault("draft_review_ids", [])
    st.session_state.setdefault("draft_failures", [])
    st.session_state.setdefault("frozen_drafts", [])
    st.session_state.setdefault("frozen_reviews", [])
    st.session_state.setdefault("other_summary", None)
    st.session_state.setdefault("review_texts", {})


@st.cache_data(ttl=15, show_spinner=False)
def _ollama_reachable() -> bool:
    """Cheap TCP probe so the sidebar can default to whatever will actually work here.

    Cached briefly -- Streamlit reruns the whole script on every widget
    interaction, and re-probing a socket on each one would be wasteful.
    A closed/refused connection just means "not reachable"; this never
    raises, so a slow or misbehaving network can't crash the page.
    """
    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    parsed = urlparse(host)
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 11434), timeout=1.0):
            return True
    except OSError:
        return False


def _graph_config(review_id: str) -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": review_id})


def _run_pipeline_live(use_demo: bool) -> None:
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
        other_summary = summarize_other_aspect_usage(analyses)

    st.session_state["mode"] = "live"
    st.session_state["graph"] = graph
    st.session_state["business_reports"] = business_reports
    st.session_state["escalations"] = escalations
    st.session_state["draft_review_ids"] = draft_review_ids
    st.session_state["draft_failures"] = draft_failures
    st.session_state["other_summary"] = other_summary
    st.session_state["review_texts"] = review_texts


def _run_pipeline_frozen() -> None:
    """Load the committed snapshot instead of calling anything live -- see module docstring."""
    if not FROZEN_SNAPSHOT_PATH.exists():
        st.error(
            f"{FROZEN_SNAPSHOT_PATH} not found. Generate it with `python scripts/freeze_demo_run.py` "
            "(needs a local Ollama server, one-time)."
        )
        return

    data = json.loads(FROZEN_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    st.session_state["mode"] = "frozen"
    st.session_state["business_reports"] = [BusinessReport.model_validate(r) for r in data["business_reports"]]
    st.session_state["escalations"] = [EscalationFlag.model_validate(e) for e in data["escalations"]]
    st.session_state["other_summary"] = OtherAspectSummary.model_validate(data["other_aspect_summary"])
    st.session_state["frozen_drafts"] = data["drafts"]
    st.session_state["frozen_reviews"] = data.get("reviews", [])


def _run_disabled_reason(data_source: str, ollama_ok: bool) -> str | None:
    """None if Run pipeline would work for this selection; otherwise why it's disabled."""
    if data_source == _FROZEN:
        if not FROZEN_SNAPSHOT_PATH.exists():
            return f"⚠️ `{FROZEN_SNAPSHOT_PATH.name}` not found — run `python scripts/freeze_demo_run.py` first."
        return None
    if not ollama_ok:
        return (
            f"⚠️ No local Ollama server detected here — this would fail. Pick **{_FROZEN}** "
            "instead, or run this app locally with `ollama serve` running (see README)."
        )
    if data_source == _REAL_LIVE and not (DEFAULT_INPUT_PATH.exists() and DEFAULT_REVIEWS_PATH.exists()):
        return f"⚠️ Missing `{DEFAULT_INPUT_PATH}` / `{DEFAULT_REVIEWS_PATH}` — run `scripts/build_subset.py` first."
    return None


def _render_sidebar(ollama_ok: bool) -> tuple[str, bool]:
    st.sidebar.title("📋 AuraPulse")
    st.sidebar.caption("Restaurant review analysis: sentiment, recurring issues, draft replies. Zero paid APIs.")

    default_index = _DATA_SOURCES.index(_DEMO_LIVE if ollama_ok else _FROZEN)
    data_source = st.sidebar.radio(
        "Data source",
        _DATA_SOURCES,
        index=default_index,
        help=(
            "🧊 Instant demo works anywhere -- it replays a real run captured earlier. The ⚡📁 "
            "live options classify and draft in real time and need a local Ollama server running "
            "on this machine."
        ),
    )

    disabled_reason = _run_disabled_reason(data_source, ollama_ok)
    if disabled_reason:
        st.sidebar.warning(disabled_reason)
    elif data_source == _REAL_LIVE:
        st.sidebar.caption(f"Reads `{DEFAULT_INPUT_PATH}` + `{DEFAULT_REVIEWS_PATH}`.")

    run_clicked = st.sidebar.button(
        "▶️ Run pipeline",
        type="primary",
        width="stretch",
        disabled=disabled_reason is not None,
        help=disabled_reason,
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Never publishes a reply automatically — every draft is for a human to read, edit, and "
        "send themselves."
    )
    st.sidebar.caption(
        "[Source](https://github.com/serpeigd/AuraPulse) · "
        "[Design notes](https://github.com/serpeigd/AuraPulse/blob/main/docs/DESIGN.md)"
    )
    return data_source, run_clicked


def _render_landing() -> None:
    st.title("📋 AuraPulse")
    st.caption(
        "Turns restaurant review sentiment into signal a business can act on: which locations "
        "have a recurring inconsistency (e.g. great food, consistently bad wait times), which "
        "negative reviews need a human-reviewed reply, and which need urgent attention."
    )
    st.info("👈 Pick a data source in the sidebar and click **▶️ Run pipeline** to see it in action.")


def _render_header(mode: str) -> None:
    st.title("📋 AuraPulse")
    if mode == "frozen":
        st.info(
            "🧊 **Instant demo** — a real run's actual output, captured once locally against a "
            "live model, not invented text. See exactly what it did per review in the walkthrough "
            "below. Read-only here; run this app locally to reject a draft and watch it regenerate."
        )
    else:
        st.success(
            "⚡ **Live run** — classification and drafting just happened in real time against "
            "your local Ollama server."
        )


def _render_kpis(business_reports: list[BusinessReport], escalations: list[EscalationFlag], draft_count: int) -> None:
    total_reviews = sum(r.review_count for r in business_reports)
    cols = st.columns(4)
    cols[0].metric("Businesses", len(business_reports))
    cols[1].metric("Reviews processed", total_reviews)
    cols[2].metric("Draft replies", draft_count)
    cols[3].metric("Escalations", len(escalations))


def _render_business_reports(business_reports: list[BusinessReport]) -> None:
    if not business_reports:
        st.caption("No businesses in this batch.")
        return

    for report in business_reports:
        with st.expander(f"🏪 {report.business_id} — {report.review_count} reviews", expanded=len(business_reports) == 1):
            cols = st.columns(3)
            cols[0].metric("🙂 Positive", report.sentiment_counts.get(Sentiment.POSITIVE, 0))
            cols[1].metric("😐 Neutral", report.sentiment_counts.get(Sentiment.NEUTRAL, 0))
            cols[2].metric("🙁 Negative", report.sentiment_counts.get(Sentiment.NEGATIVE, 0))

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
                    width="stretch",
                )
            else:
                st.caption("No aspects mentioned.")

            for flag in report.inconsistent_aspects:
                st.warning(f"⚠️ **Inconsistency:** {flag}")


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

    with st.container(border=True):
        st.markdown(f"**{review_id}** · {business_id} · draft {attempt}/{max_attempts}")
        st.caption(f"Original review: {review_text}")
        st.info(values["draft_text"])

        if outcome == "approved":
            st.success("✅ Approved — logged for a human to send (never auto-sent).")
            return
        if outcome == "needs_human_rewrite":
            st.error(
                f"❌ Rejected {max_attempts}x in a row — the model isn't converging. This one "
                "needs a human-written reply instead of another regeneration."
            )
            return

        reject_label = "❌ Reject & regenerate" if attempt < max_attempts else "❌ Reject (last automatic attempt)"
        with st.form(key=f"decision::{review_id}::{attempt}"):
            feedback = st.text_input(
                "Optional note (why reject / what to change)", key=f"feedback::{review_id}::{attempt}"
            )
            col_approve, col_reject = st.columns(2)
            approve_clicked = col_approve.form_submit_button("✅ Approve", width="stretch")
            reject_clicked = col_reject.form_submit_button(reject_label, width="stretch")

        if not (approve_clicked or reject_clicked):
            return

        approved = approve_clicked
        record_draft_decision(
            DraftDecision(
                review_id=review_id, business_id=business_id, approved=approved, feedback=(feedback or None)
            )
        )
        spinner_msg = (
            "Regenerating draft with your feedback..." if reject_clicked and attempt < max_attempts else None
        )
        if spinner_msg:
            with st.spinner(spinner_msg):
                graph.invoke(Command(resume={"approved": approved, "feedback": feedback or None}), config=config)
        else:
            graph.invoke(Command(resume={"approved": approved, "feedback": feedback or None}), config=config)
        st.rerun()


def _render_drafts_live(review_ids: list[str], draft_failures: list[str], review_texts: dict[str, str]) -> None:
    st.caption(
        "Each draft is for a human to review — approve it as-is, or reject it with a note to "
        "have it regenerated with that feedback (up to 3 attempts)."
    )
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


_SENTIMENT_BADGE = {"positive": "🟢 Positive", "neutral": "🟡 Neutral", "negative": "🔴 Negative"}
_ASPECT_ICON = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
_ROUTE_INFO = {
    "aggregate": ("📊 Aggregated only", "Positive/neutral — folded into the business report, no reply needed."),
    "draft_response": ("✍️ Drafted a reply", "Negative, but no severity signal — generated a reply for a human to review."),
    "escalate": ("🚩 Escalated to a human", "Flagged as severe — sent straight to a human, no draft generated."),
}


def _render_pipeline_walkthrough(reviews: list[dict]) -> None:
    """The Instant demo's flagship view: review in, real pipeline decision out, for every review.

    Unlike ``_render_drafts_frozen`` (only the 5 reviews that got a draft),
    this covers all 8 -- including the ones that were aggregated-only or
    escalated -- so a visitor sees the actual routing decision at work,
    not just its one most visible outcome.
    """
    st.caption(
        "Every review below is real, and every step is what the pipeline actually produced for "
        "it — captured once against a live model, not invented (see docs/DESIGN.md). This is the "
        "same classify → route → act flow `scripts/run_pipeline.py` runs end to end."
    )
    if not reviews:
        st.caption("No reviews in this snapshot.")
        return

    for r in reviews:
        route_label, route_reason = _ROUTE_INFO[r["route"]]
        with st.container(border=True):
            st.markdown(f"**{r['review_id']}** · {r['business_id']}")
            st.markdown(f"📝 _{r['review_text']}_")

            aspect_bits = [f"{a['aspect']} {_ASPECT_ICON[a['sentiment']]}" for a in r["aspects"]]
            aspects_line = ", ".join(aspect_bits) if aspect_bits else "none mentioned"
            st.markdown(f"**Sentiment:** {_SENTIMENT_BADGE[r['overall_sentiment']]} &nbsp; **Aspects:** {aspects_line}")
            st.markdown(f"**→ Route:** {route_label} — _{route_reason}_")

            if r["draft_text"]:
                st.success(f"**Draft reply:** {r['draft_text']}")
            elif r["escalation_reason"]:
                st.error(f"**Escalation reason:** {r['escalation_reason']}")


def _render_escalations(escalations: list[EscalationFlag]) -> None:
    if not escalations:
        st.caption("No reviews escalated for this batch — nothing needed urgent human attention.")
        return
    st.dataframe(
        [{"review_id": e.review_id, "business_id": e.business_id, "reason": e.reason} for e in escalations],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Escalations are also appended to `data/escalations/escalations.jsonl` by "
        "`scripts/run_pipeline.py` — this view only reflects the current in-app run."
    )


def _render_other_aspect_summary(summary: OtherAspectSummary) -> None:
    st.write(
        f"`other` used in **{summary.other_mentions}/{summary.total_mentions}** aspect mentions "
        f"(**{summary.other_share:.1%}**). A high share here would signal the closed aspect "
        "enum (food/service/price/cleanliness/wait_time/ambience) needs a new category."
    )
    if summary.other_details:
        with st.expander("What fell through to `other`"):
            for detail in summary.other_details[:20]:
                st.write(f"- {detail}")


def main() -> None:
    """Entry point for `streamlit run app/streamlit_app.py`."""
    _init_state()
    ollama_ok = _ollama_reachable()
    data_source, run_clicked = _render_sidebar(ollama_ok)

    # Only ever auto-run the instant demo -- it's free and fast. The live options
    # are real (slow) model calls and always need an explicit click.
    auto_run = not st.session_state["auto_run_done"] and data_source == _FROZEN
    if run_clicked or auto_run:
        st.session_state["auto_run_done"] = True
        try:
            if data_source == _FROZEN:
                _run_pipeline_frozen()
            else:
                _run_pipeline_live(use_demo=data_source == _DEMO_LIVE)
        except FileNotFoundError as exc:
            st.error(str(exc))

    business_reports: list[BusinessReport] | None = st.session_state["business_reports"]
    if business_reports is None:
        _render_landing()
        return

    mode = st.session_state["mode"]
    draft_count = (
        len(st.session_state["frozen_drafts"]) if mode == "frozen" else len(st.session_state["draft_review_ids"])
    )

    _render_header(mode)
    _render_kpis(business_reports, st.session_state["escalations"], draft_count)
    st.divider()

    if mode == "frozen":
        tab_walkthrough, tab_reports, tab_escalations, tab_aspects = st.tabs(
            [
                f"🔬 Pipeline walkthrough ({len(st.session_state['frozen_reviews'])})",
                f"📊 Business reports ({len(business_reports)})",
                f"🚩 Escalations ({len(st.session_state['escalations'])})",
                "🔍 Aspect coverage",
            ]
        )
        with tab_walkthrough:
            _render_pipeline_walkthrough(st.session_state["frozen_reviews"])
        with tab_reports:
            _render_business_reports(business_reports)
        with tab_escalations:
            _render_escalations(st.session_state["escalations"])
        with tab_aspects:
            _render_other_aspect_summary(st.session_state["other_summary"])
        return

    tab_reports, tab_drafts, tab_escalations, tab_aspects = st.tabs(
        [
            f"📊 Business reports ({len(business_reports)})",
            f"✍️ Draft replies ({draft_count})",
            f"🚩 Escalations ({len(st.session_state['escalations'])})",
            "🔍 Aspect coverage",
        ]
    )
    with tab_reports:
        _render_business_reports(business_reports)
    with tab_drafts:
        _render_drafts_live(
            st.session_state["draft_review_ids"],
            st.session_state["draft_failures"],
            st.session_state["review_texts"],
        )
    with tab_escalations:
        _render_escalations(st.session_state["escalations"])
    with tab_aspects:
        _render_other_aspect_summary(st.session_state["other_summary"])


if __name__ == "__main__":
    main()
