"""Pydantic schemas for structured LLM output in AuraPulse.

Defines the contract the classification step must satisfy for every
review: sentiment, one or more aspects (closed enum + escape hatch),
and a severity flag now consumed by Hito 1 routing (see ``Route``).
Also defines Hito 1's two possible per-review outcomes beyond plain
aggregation: a human-reviewable draft reply, or an escalation flag.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Sentiment(str, Enum):
    """Overall sentiment of a review, validated against the Yelp star rating."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Aspect(str, Enum):
    """Closed set of recurring business aspects tracked for aggregation.

    See docs/DESIGN.md for the enum-vs-free-text trade-off. ``OTHER`` is
    the escape hatch for content that doesn't fit; see
    ``AspectMention.other_detail``.
    """

    FOOD = "food"
    SERVICE = "service"
    PRICE = "price"
    CLEANLINESS = "cleanliness"
    WAIT_TIME = "wait_time"
    AMBIENCE = "ambience"
    OTHER = "other"


class AspectMention(BaseModel):
    """A single aspect surfaced in a review, with its own local sentiment.

    A review can mention several aspects with different sentiment each
    (e.g. food positive, wait_time negative) — this is what makes the
    inconsistency-detection framing possible at aggregation time.
    """

    aspect: Aspect
    sentiment: Sentiment
    other_detail: str | None = Field(
        default=None,
        description=(
            "Free-text detail, populated only when aspect == OTHER. "
            "Tracked in aggregation to spot when the enum needs a new category."
        ),
    )

    @model_validator(mode="after")
    def _other_detail_matches_aspect(self) -> AspectMention:
        """Enforce the OTHER <-> other_detail pairing the docstring promises.

        Schema-valid output from an LLM can still be logically
        inconsistent (e.g. aspect=food with other_detail filled in, or
        aspect=other with nothing to say) — catching that here means
        callers never have to re-check it downstream.

        Ollama's structured-output mode can't omit an optional string
        field cleanly, so models routinely fill an unset
        ``other_detail`` with ``""`` instead of JSON ``null`` — treat
        blank as unset rather than rejecting otherwise-valid output.
        """
        if self.other_detail is not None and not self.other_detail.strip():
            self.other_detail = None
        if self.aspect == Aspect.OTHER:
            if not self.other_detail:
                raise ValueError("other_detail is required when aspect is OTHER")
        elif self.other_detail is not None:
            raise ValueError(f"other_detail must be unset when aspect is {self.aspect.value!r}, not OTHER")
        return self


class ClassifiedAnalysis(BaseModel):
    """LLM output for a single review — everything except identifiers.

    ``review_id``/``business_id`` are known to the caller before
    classification even runs, so they're deliberately excluded from
    what we ask the model to produce (asking an LLM to echo back an
    identifier it didn't generate just invites it to hallucinate one).
    The classifier injects them after validating this against the
    model's structured output.
    """

    overall_sentiment: Sentiment
    aspects: list[AspectMention] = Field(default_factory=list)
    severity_flag: bool = Field(
        default=False,
        description="Drives Hito 1 escalation routing — see Route.ESCALATE in routing.decide_route.",
    )


class ReviewAnalysis(ClassifiedAnalysis):
    """Structured analysis output for a single review, with identifiers."""

    review_id: str
    business_id: str


class Route(str, Enum):
    """Where a classified review goes next, decided by ``routing.decide_route``.

    Deliberately NOT itself an LLM output: routing is a pure Python
    if/elif over already-classified fields (see docs/DESIGN.md for why a
    graph orchestrator isn't justified for 3 known routes). Every review
    is aggregated regardless of route — this only decides whether an
    *additional* action (draft, escalation) also fires.
    """

    AGGREGATE = "aggregate"
    DRAFT_RESPONSE = "draft_response"
    ESCALATE = "escalate"


class DraftText(BaseModel):
    """LLM output for a draft reply — everything except identifiers.

    Split from ``DraftResponse`` for the same reason ``ClassifiedAnalysis``
    is split from ``ReviewAnalysis``: identifiers are known to the caller
    before generation runs, so the model is never asked to echo one back.
    """

    draft_text: str = Field(description="Proposed reply text, for human review only — never auto-sent.")


class DraftResponse(DraftText):
    """An LLM-generated draft reply to a negative, non-severe review, with identifiers.

    A draft only. Per CLAUDE.md's non-negotiable rule, AuraPulse never
    publishes a response without explicit human approval — nothing in
    this codebase has a "publish" capability at all; that's enforced by
    the capability simply not existing, not by a flag on this model.
    """

    review_id: str
    business_id: str


class EscalationFlag(BaseModel):
    """A review flagged for a human to review urgently.

    Built deterministically from already-classified fields (no LLM call
    — see ``response_draft.flag_for_escalation``).
    """

    review_id: str
    business_id: str
    reason: str = Field(description="Human-readable explanation of why this review was escalated.")
