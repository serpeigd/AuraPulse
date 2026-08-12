"""The reject -> regenerate loop for draft replies -- this project's first
and only LangGraph use, and the concrete justification for introducing it.

Every earlier entry in docs/DESIGN.md described LangGraph's trigger
condition abstractly: "a reject/regenerate loop with persisted, pausable
state." ``routing.decide_route`` stays a plain ``if/elif`` -- 3 known
outcomes is still legible without a framework, and this module doesn't
change that. This is a *different* problem: a human needs to reject a
draft, have it regenerated with their feedback, and possibly reject
again, with the whole thing pausable indefinitely between clicks (a
Streamlit rerun, or the app being closed and reopened). A plain Python
loop can't pause and resume across process boundaries; LangGraph's
``interrupt()``/``Command(resume=...)`` mechanism, backed by a
checkpointer, is built for exactly this.

Design, approved before implementation (see docs/DESIGN.md):
    - Two nodes: ``generate_draft`` (calls ``response_draft.
      generate_draft_response``, optionally with feedback from the last
      rejection) and ``human_review`` (pauses via ``interrupt()`` until
      a human's decision is supplied via ``Command(resume=...)``).
    - Feedback handling is the "simple" option: each regeneration call
      only sees the most recent rejection note, not an accumulated
      history of every prior draft. Revisit if an eval shows the model
      repeating the same mistake across attempts.
    - Bounded by ``MAX_ATTEMPTS`` (3): after that many drafts, a
      rejection ends the loop in ``needs_human_rewrite`` instead of
      regenerating forever against a model that isn't converging.
    - ``MemorySaver`` (in-process, zero-cost) as the checkpointer --
      no persistence across a process restart, which is an accepted
      limitation for a local single-user demo (see docs/DESIGN.md).
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from aurapulse.response_draft import generate_draft_response
from aurapulse.schemas import ReviewAnalysis

MAX_ATTEMPTS = 3

DraftOutcome = Literal["pending", "approved", "needs_human_rewrite"]


class DraftGraphState(TypedDict):
    """Everything the graph threads between nodes for one review's draft loop."""

    review_id: str
    business_id: str
    review_text: str
    analysis: ReviewAnalysis
    draft_text: str | None
    feedback: str | None
    approved: bool
    attempt: int
    max_attempts: int


class HumanDecision(TypedDict):
    """The payload a caller passes to ``Command(resume=...)`` after a human decides."""

    approved: bool
    feedback: str | None


def initial_state(
    review_id: str,
    business_id: str,
    review_text: str,
    analysis: ReviewAnalysis,
    *,
    max_attempts: int = MAX_ATTEMPTS,
) -> DraftGraphState:
    """Build the state to pass to the first ``graph.invoke(...)`` call for a review."""
    return {
        "review_id": review_id,
        "business_id": business_id,
        "review_text": review_text,
        "analysis": analysis,
        "draft_text": None,
        "feedback": None,
        "approved": False,
        "attempt": 0,
        "max_attempts": max_attempts,
    }


def draft_outcome(state_values: dict) -> DraftOutcome:
    """Classify a graph state snapshot's values into what a UI should show.

    Reads plain dict values (e.g. from ``graph.get_state(config).values``
    or a node's return dict) rather than requiring a full
    ``DraftGraphState`` -- callers often have a partial view.
    """
    if state_values.get("approved"):
        return "approved"
    if state_values.get("attempt", 0) >= state_values.get("max_attempts", MAX_ATTEMPTS):
        return "needs_human_rewrite"
    return "pending"


def _generate_draft_node(state: DraftGraphState) -> dict:
    """Call the LLM for a new draft, using ``state['feedback']`` if this is a regeneration.

    Raises:
        ClassificationError: propagated from ``generate_draft_response``
            (e.g. Ollama unreachable) -- callers of ``graph.invoke``
            must handle this themselves; the graph does not swallow it.
    """
    draft = generate_draft_response(
        state["review_id"],
        state["business_id"],
        state["review_text"],
        state["analysis"],
        feedback=state.get("feedback"),
    )
    return {"draft_text": draft.draft_text, "attempt": state.get("attempt", 0) + 1, "feedback": None}


def _human_review_node(state: DraftGraphState) -> dict:
    """Pause the graph until a human supplies a ``HumanDecision`` via ``Command(resume=...)``."""
    decision: HumanDecision = interrupt(
        {"review_id": state["review_id"], "draft_text": state["draft_text"], "attempt": state["attempt"]}
    )
    return {"approved": bool(decision.get("approved")), "feedback": decision.get("feedback")}


def _route_after_review(state: DraftGraphState) -> str:
    if state.get("approved") or state["attempt"] >= state.get("max_attempts", MAX_ATTEMPTS):
        return END
    return "generate_draft"


def build_draft_graph() -> CompiledStateGraph:
    """Compile the reject/regenerate graph with an in-process (zero-cost) checkpointer.

    One compiled graph is reusable across many reviews -- each review
    gets its own ``thread_id`` in the config passed to ``invoke``/
    ``get_state``, so their states never collide.
    """
    graph = StateGraph(DraftGraphState)
    graph.add_node("generate_draft", _generate_draft_node)
    graph.add_node("human_review", _human_review_node)
    graph.set_entry_point("generate_draft")
    graph.add_edge("generate_draft", "human_review")
    graph.add_conditional_edges("human_review", _route_after_review, {"generate_draft": "generate_draft", END: END})
    return graph.compile(checkpointer=MemorySaver())
