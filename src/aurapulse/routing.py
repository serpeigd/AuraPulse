"""Hito 1 routing: decide what happens next to an already-classified review.

Plain ``if/elif`` on purpose — see docs/DESIGN.md, "Hito 1 routing structure"
for why a graph orchestrator (LangGraph) isn't justified for 3 known routes,
and why the decision is split from the actions that follow it (Option B).

``decide_route`` is pure: no LLM call, no I/O, no side effects. By the time
a review reaches it, an LLM has already done the only judgment call that
matters (``classify_review``) — routing is just reading the result.
"""

from __future__ import annotations

from aurapulse.schemas import ReviewAnalysis, Route, Sentiment


def decide_route(analysis: ReviewAnalysis) -> Route:
    """Decide a classified review's route: aggregate, draft a reply, or escalate.

    Rules (per CLAUDE.md's routing spec):
        - Not negative (positive or neutral) -> AGGREGATE. A neutral
          "nothing special" review isn't a complaint that needs a reply.
        - Negative with ``severity_flag`` set -> ESCALATE.
        - Negative without ``severity_flag`` -> DRAFT_RESPONSE.

    Every review is aggregated regardless of route (see ``aggregation.py``)
    — this only decides whether an *additional* action also fires.

    Args:
        analysis: An already-classified review.

    Returns:
        The route this review should take next.
    """
    if analysis.overall_sentiment != Sentiment.NEGATIVE:
        return Route.AGGREGATE
    if analysis.severity_flag:
        return Route.ESCALATE
    return Route.DRAFT_RESPONSE
