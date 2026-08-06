"""Hito 1 glue: route every classified review and collect what fires.

This is the "executor" half of Option B (see docs/DESIGN.md) — the other
half, ``routing.decide_route``, only decides. Nothing here makes a
routing decision itself; it just dispatches to the right handler and
gathers results.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from aurapulse.aggregation import BusinessReport, aggregate_reviews
from aurapulse.classifier import ClassificationError
from aurapulse.response_draft import flag_for_escalation, generate_draft_response
from aurapulse.routing import decide_route
from aurapulse.schemas import DraftResponse, EscalationFlag, ReviewAnalysis, Route

logger = logging.getLogger(__name__)


class Hito1Batch(BaseModel):
    """Everything that came out of routing a batch of classified reviews."""

    business_reports: list[BusinessReport] = Field(
        description="Aggregated per-business reports, covering every review regardless of route."
    )
    drafts: list[DraftResponse] = Field(default_factory=list, description="Reviews routed to DRAFT_RESPONSE.")
    escalations: list[EscalationFlag] = Field(default_factory=list, description="Reviews routed to ESCALATE.")
    draft_failures: list[str] = Field(
        default_factory=list,
        description=(
            "review_ids routed to DRAFT_RESPONSE where generation itself failed "
            "(e.g. Ollama unreachable) -- surfaced explicitly rather than silently dropped."
        ),
    )


def process_reviews(analyses: list[ReviewAnalysis], review_texts: dict[str, str]) -> Hito1Batch:
    """Route every review, dispatch to its handler, and aggregate all of them.

    Args:
        analyses: Already-classified reviews (``classify_review`` output).
        review_texts: Raw review text keyed by ``review_id`` -- needed
            only for reviews routed to DRAFT_RESPONSE, since drafting a
            reply requires the original text, not just the structured
            classification.

    Returns:
        Business reports (unconditional, all reviews), plus drafts and
        escalations for the reviews routed to those actions.

    Raises:
        KeyError: if a review routed to DRAFT_RESPONSE is missing from
            ``review_texts`` -- fails loud rather than silently skipping
            a review that needed a reply.
    """
    drafts: list[DraftResponse] = []
    escalations: list[EscalationFlag] = []
    draft_failures: list[str] = []

    for analysis in analyses:
        route = decide_route(analysis)
        if route == Route.AGGREGATE:
            continue
        if route == Route.ESCALATE:
            escalations.append(flag_for_escalation(analysis))
            continue
        # route == Route.DRAFT_RESPONSE
        text = review_texts[analysis.review_id]
        try:
            drafts.append(generate_draft_response(analysis.review_id, analysis.business_id, text, analysis))
        except ClassificationError as exc:
            draft_failures.append(analysis.review_id)
            logger.warning("review_id=%s: draft generation failed, skipping: %s", analysis.review_id, exc)

    return Hito1Batch(
        business_reports=aggregate_reviews(analyses),
        drafts=drafts,
        escalations=escalations,
        draft_failures=draft_failures,
    )
