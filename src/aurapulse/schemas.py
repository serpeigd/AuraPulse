"""Pydantic schemas for structured LLM output in AuraPulse.

Defines the contract the classification step must satisfy for every
review: sentiment, one or more aspects (closed enum + escape hatch),
and an optional severity flag reserved for future escalation routing
(Hito 1+, not used in Hito 0 aggregation).
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
        """
        if self.aspect == Aspect.OTHER:
            if not self.other_detail or not self.other_detail.strip():
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
        description="Reserved for future escalation routing (Hito 1+). Unused in Hito 0.",
    )


class ReviewAnalysis(ClassifiedAnalysis):
    """Structured analysis output for a single review, with identifiers.

    ``severity_flag`` is carried in the schema now (per CLAUDE.md — leave
    the door open for future escalation) but is NOT consumed by any
    Hito 0 logic; escalation routing is explicitly out of scope until
    Hito 1/2.
    """

    review_id: str
    business_id: str
