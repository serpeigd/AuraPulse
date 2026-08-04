"""Per-business aggregation: sentiment distribution, recurring aspects,
and the inconsistency signal that is AuraPulse's core value proposition
(see docs/DESIGN.md, "Product framing").

Takes already-classified ``ReviewAnalysis`` records — this module never
calls the classifier itself — groups them by ``business_id``, and
computes summaries that always carry their own denominator (never a
bare score; see CLAUDE.md's "never report a metric without visible
case count" rule).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from aurapulse.schemas import Aspect, ReviewAnalysis, Sentiment

# An aspect needs at least this many mentions before it's eligible to be
# flagged as an inconsistency — avoids flagging off a single noisy review.
MIN_MENTIONS_FOR_FLAG = 3

# How far above the average negative_share of a business's OTHER aspects
# one aspect's negative_share must be before it's flagged. Comparing
# against the business's own baseline (not a fixed cutoff) means a
# business that's uniformly bad everywhere doesn't get every aspect
# flagged — only ones that stand out as worse than the rest.
INCONSISTENCY_THRESHOLD = 0.30


class AspectSummary(BaseModel):
    """How one aspect reads across every mention of it for a business."""

    aspect: Aspect
    total_mentions: int
    positive_count: int
    neutral_count: int
    negative_count: int
    negative_share: float = Field(description="negative_count / total_mentions; 0.0 if never mentioned.")


class BusinessReport(BaseModel):
    """Aggregated reputation report for a single business."""

    business_id: str
    review_count: int
    sentiment_counts: dict[Sentiment, int]
    aspect_summaries: list[AspectSummary] = Field(
        description="Sorted by total_mentions, most-discussed aspect first."
    )
    inconsistent_aspects: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable flags for aspects whose negative_share is "
            "disproportionately worse than the business's other aspects. "
            "Empty if nothing stands out."
        ),
    )


def _summarize_aspects(analyses: list[ReviewAnalysis]) -> list[AspectSummary]:
    counts: dict[Aspect, Counter[Sentiment]] = defaultdict(Counter)
    for analysis in analyses:
        for mention in analysis.aspects:
            counts[mention.aspect][mention.sentiment] += 1

    summaries = []
    for aspect, sentiment_counts in counts.items():
        total = sum(sentiment_counts.values())
        negative = sentiment_counts[Sentiment.NEGATIVE]
        summaries.append(
            AspectSummary(
                aspect=aspect,
                total_mentions=total,
                positive_count=sentiment_counts[Sentiment.POSITIVE],
                neutral_count=sentiment_counts[Sentiment.NEUTRAL],
                negative_count=negative,
                negative_share=negative / total if total else 0.0,
            )
        )
    summaries.sort(key=lambda s: s.total_mentions, reverse=True)
    return summaries


def _flag_inconsistencies(summaries: list[AspectSummary]) -> list[str]:
    eligible = [s for s in summaries if s.total_mentions >= MIN_MENTIONS_FOR_FLAG]
    if len(eligible) < 2:
        return []

    flags = []
    for target in eligible:
        others = [s for s in eligible if s is not target]
        baseline = sum(s.negative_share for s in others) / len(others)
        if target.negative_share - baseline >= INCONSISTENCY_THRESHOLD:
            flags.append(
                f"{target.aspect.value}: {target.negative_share:.0%} negative "
                f"({target.negative_count}/{target.total_mentions}) vs. "
                f"{baseline:.0%} average across this business's other aspects"
            )
    return flags


def build_business_report(business_id: str, analyses: list[ReviewAnalysis]) -> BusinessReport:
    """Aggregate one business's classified reviews into a report.

    Args:
        business_id: The business these analyses belong to (not
            re-derived from ``analyses``, so an empty list is still valid).
        analyses: Every classified review available for this business.

    Returns:
        Sentiment distribution, per-aspect summaries, and any flagged
        inconsistencies.
    """
    sentiment_counts = dict.fromkeys(Sentiment, 0)
    for analysis in analyses:
        sentiment_counts[analysis.overall_sentiment] += 1

    aspect_summaries = _summarize_aspects(analyses)
    return BusinessReport(
        business_id=business_id,
        review_count=len(analyses),
        sentiment_counts=sentiment_counts,
        aspect_summaries=aspect_summaries,
        inconsistent_aspects=_flag_inconsistencies(aspect_summaries),
    )


def aggregate_reviews(analyses: list[ReviewAnalysis]) -> list[BusinessReport]:
    """Group classified reviews by business and build a report for each.

    Returns:
        One ``BusinessReport`` per distinct ``business_id`` present in
        ``analyses``, sorted by ``business_id`` for deterministic output.
    """
    by_business: dict[str, list[ReviewAnalysis]] = defaultdict(list)
    for analysis in analyses:
        by_business[analysis.business_id].append(analysis)

    return [build_business_report(business_id, group) for business_id, group in sorted(by_business.items())]
