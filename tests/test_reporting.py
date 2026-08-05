"""Tests for the human-readable report formatting."""

from __future__ import annotations

from aurapulse.aggregation import (
    aggregate_reviews,
    build_business_report,
    summarize_other_aspect_usage,
)
from aurapulse.reporting import (
    format_business_report,
    format_full_report,
    format_other_aspect_summary,
)
from aurapulse.schemas import Aspect, AspectMention, ReviewAnalysis, Sentiment


def _review(review_id: str, business_id: str, aspect_sentiments: dict[Aspect, Sentiment]) -> ReviewAnalysis:
    aspects = [AspectMention(aspect=a, sentiment=s) for a, s in aspect_sentiments.items()]
    overall = Sentiment.NEGATIVE if Sentiment.NEGATIVE in aspect_sentiments.values() else Sentiment.POSITIVE
    return ReviewAnalysis(review_id=review_id, business_id=business_id, overall_sentiment=overall, aspects=aspects)


def test_format_business_report_includes_key_numbers() -> None:
    analyses = [
        _review("r1", "biz-1", {Aspect.FOOD: Sentiment.POSITIVE}),
        _review("r2", "biz-1", {Aspect.FOOD: Sentiment.NEGATIVE}),
    ]
    report = build_business_report("biz-1", analyses)

    text = format_business_report(report)

    assert "biz-1" in text
    assert "2 reviews" in text
    assert "food" in text
    assert "2 mentions" in text


def test_format_business_report_shows_flags_when_present() -> None:
    analyses = []
    for i in range(10):
        wait_sentiment = Sentiment.NEGATIVE if i < 7 else Sentiment.POSITIVE
        analyses.append(
            _review(
                f"r{i}", "biz-inconsistent", {Aspect.FOOD: Sentiment.POSITIVE, Aspect.WAIT_TIME: wait_sentiment}
            )
        )
    report = build_business_report("biz-inconsistent", analyses)

    text = format_business_report(report)

    assert "INCONSISTENCIES FLAGGED" in text
    assert "wait_time" in text


def test_format_business_report_no_flags_section_when_none() -> None:
    report = build_business_report("biz-quiet", [_review("r1", "biz-quiet", {Aspect.FOOD: Sentiment.POSITIVE})])

    text = format_business_report(report)

    assert "INCONSISTENCIES FLAGGED" not in text


def test_format_business_report_handles_no_aspects() -> None:
    report = build_business_report("biz-empty", [])

    text = format_business_report(report)

    assert "biz-empty" in text
    assert "none mentioned" in text


def test_format_other_aspect_summary_includes_share_and_details() -> None:
    review_with_other = ReviewAnalysis(
        review_id="r1",
        business_id="biz-1",
        overall_sentiment=Sentiment.NEGATIVE,
        aspects=[AspectMention(aspect=Aspect.OTHER, sentiment=Sentiment.NEGATIVE, other_detail="parking")],
    )
    summary = summarize_other_aspect_usage([review_with_other])

    text = format_other_aspect_summary(summary)

    assert "1/1" in text
    assert "parking" in text


def test_format_full_report_includes_every_business_and_the_enum_summary() -> None:
    analyses = [
        _review("r1", "biz-a", {Aspect.FOOD: Sentiment.POSITIVE}),
        _review("r2", "biz-b", {Aspect.SERVICE: Sentiment.NEGATIVE}),
    ]
    business_reports = aggregate_reviews(analyses)
    other_summary = summarize_other_aspect_usage(analyses)

    text = format_full_report(business_reports, other_summary)

    assert "biz-a" in text
    assert "biz-b" in text
    assert "Aspect enum coverage" in text
