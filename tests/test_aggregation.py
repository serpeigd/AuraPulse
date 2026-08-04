"""Tests for per-business aggregation and inconsistency flagging.

Uses hand-built ReviewAnalysis records with known aspect/sentiment
mixes -- no LLM, no dataset -- so the flagging threshold logic is
exercised precisely rather than relying on fake_reviews.py's small
fixed dataset (too few mentions per aspect there to cross
MIN_MENTIONS_FOR_FLAG).
"""

from __future__ import annotations

from aurapulse.aggregation import aggregate_reviews, build_business_report
from aurapulse.fake_reviews import generate_fixed_dataset
from aurapulse.schemas import Aspect, AspectMention, ReviewAnalysis, Sentiment


def _review(review_id: str, business_id: str, aspect_sentiments: dict[Aspect, Sentiment]) -> ReviewAnalysis:
    aspects = [AspectMention(aspect=a, sentiment=s) for a, s in aspect_sentiments.items()]
    overall = Sentiment.NEGATIVE if Sentiment.NEGATIVE in aspect_sentiments.values() else Sentiment.POSITIVE
    return ReviewAnalysis(review_id=review_id, business_id=business_id, overall_sentiment=overall, aspects=aspects)


def test_build_business_report_counts_sentiment_and_aspects() -> None:
    analyses = [
        _review("r1", "biz-1", {Aspect.FOOD: Sentiment.POSITIVE}),
        _review("r2", "biz-1", {Aspect.FOOD: Sentiment.POSITIVE, Aspect.SERVICE: Sentiment.NEGATIVE}),
    ]

    report = build_business_report("biz-1", analyses)

    assert report.business_id == "biz-1"
    assert report.review_count == 2
    assert report.sentiment_counts[Sentiment.POSITIVE] == 1
    assert report.sentiment_counts[Sentiment.NEGATIVE] == 1
    assert report.sentiment_counts[Sentiment.NEUTRAL] == 0

    by_aspect = {s.aspect: s for s in report.aspect_summaries}
    assert by_aspect[Aspect.FOOD].total_mentions == 2
    assert by_aspect[Aspect.FOOD].positive_count == 2
    assert by_aspect[Aspect.FOOD].negative_share == 0.0
    assert by_aspect[Aspect.SERVICE].total_mentions == 1
    assert by_aspect[Aspect.SERVICE].negative_share == 1.0


def test_build_business_report_handles_no_reviews() -> None:
    report = build_business_report("biz-empty", [])

    assert report.review_count == 0
    assert report.aspect_summaries == []
    assert report.inconsistent_aspects == []
    assert all(count == 0 for count in report.sentiment_counts.values())


def test_aspect_summaries_sorted_by_mention_count_descending() -> None:
    analyses = [
        _review("r1", "biz-1", {Aspect.FOOD: Sentiment.POSITIVE, Aspect.PRICE: Sentiment.NEUTRAL}),
        _review("r2", "biz-1", {Aspect.FOOD: Sentiment.POSITIVE}),
        _review("r3", "biz-1", {Aspect.FOOD: Sentiment.NEGATIVE}),
    ]

    report = build_business_report("biz-1", analyses)

    assert [s.aspect for s in report.aspect_summaries] == [Aspect.FOOD, Aspect.PRICE]


def test_flags_the_core_inconsistency_case() -> None:
    """Food consistently good, wait_time consistently bad -> flagged.

    This is the exact "AuraPulse" scenario from CLAUDE.md: food praised
    while another aspect is consistently criticized.
    """
    analyses = []
    for i in range(10):
        wait_sentiment = Sentiment.NEGATIVE if i < 7 else Sentiment.POSITIVE
        analyses.append(
            _review(f"r{i}", "biz-inconsistent", {Aspect.FOOD: Sentiment.POSITIVE, Aspect.WAIT_TIME: wait_sentiment})
        )

    report = build_business_report("biz-inconsistent", analyses)

    assert len(report.inconsistent_aspects) == 1
    assert "wait_time" in report.inconsistent_aspects[0]


def test_does_not_flag_when_uniformly_bad() -> None:
    """Every aspect equally bad -> nothing stands out, nothing flagged."""
    analyses = [
        _review(f"r{i}", "biz-uniform", {Aspect.FOOD: Sentiment.NEGATIVE, Aspect.SERVICE: Sentiment.NEGATIVE})
        for i in range(5)
    ]

    report = build_business_report("biz-uniform", analyses)

    assert report.inconsistent_aspects == []


def test_does_not_flag_below_minimum_mentions() -> None:
    """A single bad mention of an aspect isn't enough signal to flag."""
    analyses = [
        _review("r1", "biz-thin", {Aspect.FOOD: Sentiment.POSITIVE}),
        _review("r2", "biz-thin", {Aspect.FOOD: Sentiment.POSITIVE}),
        _review("r3", "biz-thin", {Aspect.WAIT_TIME: Sentiment.NEGATIVE}),
    ]

    report = build_business_report("biz-thin", analyses)

    assert report.inconsistent_aspects == []


def test_aggregate_reviews_groups_by_business_and_sorts_output() -> None:
    analyses = [
        _review("r1", "biz-b", {Aspect.FOOD: Sentiment.POSITIVE}),
        _review("r2", "biz-a", {Aspect.SERVICE: Sentiment.NEGATIVE}),
        _review("r3", "biz-b", {Aspect.FOOD: Sentiment.NEGATIVE}),
    ]

    reports = aggregate_reviews(analyses)

    assert [r.business_id for r in reports] == ["biz-a", "biz-b"]
    assert {r.business_id: r.review_count for r in reports} == {"biz-a": 1, "biz-b": 2}


def test_aggregate_reviews_runs_end_to_end_on_fake_dataset() -> None:
    """Sanity check against the project's own deterministic fixtures.

    Not a flagging test (too few mentions per aspect in this small
    dataset to cross MIN_MENTIONS_FOR_FLAG) -- just confirms grouping
    and counting don't crash or silently drop reviews on realistic
    multi-aspect data.
    """
    analyses = [review.expected for review in generate_fixed_dataset()]

    reports = aggregate_reviews(analyses)

    assert {r.business_id for r in reports} == {"biz-alpha", "biz-beta", "biz-gamma"}
    assert sum(r.review_count for r in reports) == len(analyses)
