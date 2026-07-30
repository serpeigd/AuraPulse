"""Tests for the deterministic fake review generator.

These tests validate the generator itself (determinism, schema
coverage) — not an LLM. That comparison happens later, once real
classification exists, by running the same fixed dataset's review
text through the classifier and diffing against ``expected``.
"""

from aurapulse.fake_reviews import generate_fixed_dataset
from aurapulse.schemas import Aspect, ReviewAnalysis, Sentiment


def test_dataset_is_non_empty() -> None:
    dataset = generate_fixed_dataset()
    assert len(dataset) > 0


def test_every_review_has_valid_schema_and_nonempty_text() -> None:
    for review in generate_fixed_dataset():
        assert isinstance(review.expected, ReviewAnalysis)
        assert review.text.strip() != ""


def test_generation_is_deterministic() -> None:
    first = generate_fixed_dataset()
    second = generate_fixed_dataset()
    assert [r.text for r in first] == [r.text for r in second]
    assert [r.expected.model_dump() for r in first] == [r.expected.model_dump() for r in second]


def test_every_aspect_is_covered() -> None:
    seen_aspects = {
        mention.aspect for review in generate_fixed_dataset() for mention in review.expected.aspects
    }
    assert seen_aspects == set(Aspect)


def test_every_sentiment_is_covered() -> None:
    seen_sentiments = {review.expected.overall_sentiment for review in generate_fixed_dataset()}
    assert seen_sentiments == set(Sentiment)


def test_other_aspect_always_has_detail() -> None:
    for review in generate_fixed_dataset():
        for mention in review.expected.aspects:
            if mention.aspect == Aspect.OTHER:
                assert mention.other_detail is not None and mention.other_detail.strip() != ""


def test_inconsistency_scenario_exists() -> None:
    """At least one review must mix a positive and a negative aspect.

    This is the core AuraPulse case (e.g. food good, wait_time bad) —
    if no fixture covers it, aggregation logic for inconsistency
    detection can't be tested against known ground truth.
    """
    for review in generate_fixed_dataset():
        sentiments = {mention.sentiment for mention in review.expected.aspects}
        if Sentiment.POSITIVE in sentiments and Sentiment.NEGATIVE in sentiments:
            return
    raise AssertionError("no fixture review mixes a positive and a negative aspect")


def test_severity_flag_is_exercised() -> None:
    flags = {review.expected.severity_flag for review in generate_fixed_dataset()}
    assert flags == {True, False}
