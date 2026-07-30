"""Deterministic fake review generator.

Produces synthetic review text paired with its known-correct
``ReviewAnalysis`` (ground truth). This exists so the classification
pipeline can be tested end-to-end before a single real LLM call is
made, and before the Yelp dataset is even loaded — per CLAUDE.md,
ground truth must never come from an LLM.

Everything here is template-based and fully deterministic: no
``random`` module, no network access, no LLM. Calling the same
function twice always yields byte-identical output.
"""

from __future__ import annotations

from typing import NamedTuple

from aurapulse.schemas import Aspect, AspectMention, ReviewAnalysis, Sentiment

# One canonical sentence per (aspect, sentiment) pair. Kept to a single
# template each (rather than a random pick) so output stays deterministic
# without needing to thread a seed through the API.
_SENTENCE_TEMPLATES: dict[tuple[Aspect, Sentiment], str] = {
    (Aspect.FOOD, Sentiment.POSITIVE): "The food was delicious and beautifully presented.",
    (Aspect.FOOD, Sentiment.NEUTRAL): "The food was fine, nothing special.",
    (Aspect.FOOD, Sentiment.NEGATIVE): "The food was bland and arrived overcooked.",
    (Aspect.SERVICE, Sentiment.POSITIVE): "Our waiter was attentive and friendly all night.",
    (Aspect.SERVICE, Sentiment.NEUTRAL): "Service was okay, nothing memorable either way.",
    (Aspect.SERVICE, Sentiment.NEGATIVE): "The staff was rude and ignored us for most of the meal.",
    (Aspect.PRICE, Sentiment.POSITIVE): "Great value for the amount of food we got.",
    (Aspect.PRICE, Sentiment.NEUTRAL): "Prices were about what you'd expect for this kind of place.",
    (Aspect.PRICE, Sentiment.NEGATIVE): "Way overpriced for what you actually get on the plate.",
    (Aspect.CLEANLINESS, Sentiment.POSITIVE): "The whole restaurant was spotless.",
    (Aspect.CLEANLINESS, Sentiment.NEUTRAL): "Cleanliness was acceptable, no complaints.",
    (Aspect.CLEANLINESS, Sentiment.NEGATIVE): "The tables were sticky and the bathroom was dirty.",
    (Aspect.WAIT_TIME, Sentiment.POSITIVE): "We were seated immediately and the food came out fast.",
    (Aspect.WAIT_TIME, Sentiment.NEUTRAL): "The wait was about average for a Friday night.",
    (Aspect.WAIT_TIME, Sentiment.NEGATIVE): "We waited over 45 minutes just to get our order taken.",
    (Aspect.AMBIENCE, Sentiment.POSITIVE): "The atmosphere was cozy and the music was just right.",
    (Aspect.AMBIENCE, Sentiment.NEUTRAL): "The ambience was unremarkable, a fairly generic dining room.",
    (Aspect.AMBIENCE, Sentiment.NEGATIVE): "It was way too loud and cramped to hold a conversation.",
}

_OTHER_SENTENCE = "The parking lot was a nightmare and we almost gave up before even sitting down."


class FakeReview(NamedTuple):
    """A synthetic review paired with its known-correct analysis."""

    text: str
    expected: ReviewAnalysis


def make_fake_review(
    review_id: str,
    business_id: str,
    aspect_sentiments: dict[Aspect, Sentiment],
    *,
    overall_sentiment: Sentiment,
    include_other: bool = False,
    severity_flag: bool = False,
) -> FakeReview:
    """Build one synthetic review from explicit (aspect -> sentiment) choices.

    Args:
        review_id: Identifier to embed in the expected analysis.
        business_id: Identifier to embed in the expected analysis.
        aspect_sentiments: Which aspects to mention and their sentiment.
            Order is preserved in both the generated text and the
            expected ``aspects`` list.
        overall_sentiment: Expected overall sentiment. Passed explicitly
            (not inferred from ``aspect_sentiments``) so the ground truth
            stays under the caller's control, e.g. to model a review
            whose overall tone doesn't match a single dominant aspect.
        include_other: If True, appends a fixed OTHER-aspect sentence
            with its ``other_detail`` populated.
        severity_flag: Expected value for the reserved severity field.

    Returns:
        The generated review text and its matching ``ReviewAnalysis``.
    """
    sentences = [_SENTENCE_TEMPLATES[(aspect, sentiment)] for aspect, sentiment in aspect_sentiments.items()]
    mentions = [
        AspectMention(aspect=aspect, sentiment=sentiment) for aspect, sentiment in aspect_sentiments.items()
    ]

    if include_other:
        sentences.append(_OTHER_SENTENCE)
        mentions.append(
            AspectMention(aspect=Aspect.OTHER, sentiment=Sentiment.NEGATIVE, other_detail="parking availability")
        )

    text = " ".join(sentences)
    expected = ReviewAnalysis(
        review_id=review_id,
        business_id=business_id,
        overall_sentiment=overall_sentiment,
        aspects=mentions,
        severity_flag=severity_flag,
    )
    return FakeReview(text=text, expected=expected)


def generate_fixed_dataset() -> list[FakeReview]:
    """Return a fixed, deterministic set of fake reviews covering the schema.

    Coverage by design:
        - Every ``Aspect`` value appears at least once.
        - Every ``Sentiment`` value appears at least once per aspect that
          isn't OTHER.
        - At least one multi-aspect review with conflicting sentiment
          per aspect (the "food good, wait_time bad" inconsistency case
          that is AuraPulse's core value proposition).
        - At least one review using the OTHER escape hatch.
        - At least one review with ``severity_flag=True``.

    This is the full set of scenarios the aggregation/reporting logic
    will be evaluated against before any real LLM classification runs.
    """
    return [
        make_fake_review(
            "fake-001",
            "biz-alpha",
            {
                Aspect.FOOD: Sentiment.POSITIVE,
                Aspect.SERVICE: Sentiment.POSITIVE,
                Aspect.AMBIENCE: Sentiment.POSITIVE,
            },
            overall_sentiment=Sentiment.POSITIVE,
        ),
        make_fake_review(
            "fake-002",
            "biz-alpha",
            {
                Aspect.FOOD: Sentiment.NEGATIVE,
                Aspect.SERVICE: Sentiment.NEGATIVE,
                Aspect.CLEANLINESS: Sentiment.NEGATIVE,
            },
            overall_sentiment=Sentiment.NEGATIVE,
        ),
        make_fake_review(
            "fake-003",
            "biz-alpha",
            {Aspect.FOOD: Sentiment.NEUTRAL, Aspect.PRICE: Sentiment.NEUTRAL},
            overall_sentiment=Sentiment.NEUTRAL,
        ),
        # The core AuraPulse inconsistency case: great food, bad wait time.
        make_fake_review(
            "fake-004",
            "biz-beta",
            {Aspect.FOOD: Sentiment.POSITIVE, Aspect.WAIT_TIME: Sentiment.NEGATIVE},
            overall_sentiment=Sentiment.NEGATIVE,
        ),
        make_fake_review(
            "fake-005",
            "biz-beta",
            {Aspect.PRICE: Sentiment.POSITIVE, Aspect.CLEANLINESS: Sentiment.NEGATIVE},
            overall_sentiment=Sentiment.NEUTRAL,
        ),
        make_fake_review(
            "fake-006",
            "biz-gamma",
            {Aspect.WAIT_TIME: Sentiment.POSITIVE, Aspect.AMBIENCE: Sentiment.NEGATIVE},
            overall_sentiment=Sentiment.NEUTRAL,
        ),
        make_fake_review(
            "fake-007",
            "biz-gamma",
            {Aspect.SERVICE: Sentiment.NEUTRAL},
            overall_sentiment=Sentiment.NEUTRAL,
            include_other=True,
        ),
        make_fake_review(
            "fake-008",
            "biz-gamma",
            {Aspect.CLEANLINESS: Sentiment.NEGATIVE, Aspect.SERVICE: Sentiment.NEGATIVE},
            overall_sentiment=Sentiment.NEGATIVE,
            severity_flag=True,
        ),
    ]
