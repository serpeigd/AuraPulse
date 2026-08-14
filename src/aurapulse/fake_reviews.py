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
    (Aspect.DELIVERY, Sentiment.POSITIVE): "Delivery arrived quickly and everything was still hot when it got here.",
    (Aspect.DELIVERY, Sentiment.NEUTRAL): "Delivery took about as long as we expected, nothing worth mentioning either way.",
    (Aspect.DELIVERY, Sentiment.NEGATIVE): "Our delivery order showed up missing half the items we paid for.",
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
        # Mixed positive+negative aspects -> overall NEGATIVE: a real
        # complaint drags the overall tone down even when something else
        # was fine. Matches both typical Yelp star-rating behavior and
        # what the local model does consistently on its own (see
        # docs/DESIGN.md). NEUTRAL is reserved for genuinely lukewarm
        # language (see fake-003), not for "one good, one bad".
        make_fake_review(
            "fake-005",
            "biz-beta",
            {Aspect.PRICE: Sentiment.POSITIVE, Aspect.CLEANLINESS: Sentiment.NEGATIVE},
            overall_sentiment=Sentiment.NEGATIVE,
        ),
        make_fake_review(
            "fake-006",
            "biz-gamma",
            {Aspect.WAIT_TIME: Sentiment.POSITIVE, Aspect.AMBIENCE: Sentiment.NEGATIVE},
            overall_sentiment=Sentiment.NEGATIVE,
        ),
        make_fake_review(
            "fake-007",
            "biz-gamma",
            {Aspect.SERVICE: Sentiment.NEUTRAL},
            overall_sentiment=Sentiment.NEGATIVE,
            include_other=True,
        ),
        make_fake_review(
            "fake-008",
            "biz-gamma",
            {Aspect.CLEANLINESS: Sentiment.NEGATIVE, Aspect.SERVICE: Sentiment.NEGATIVE},
            overall_sentiment=Sentiment.NEGATIVE,
            severity_flag=True,
        ),
        # Aspect.DELIVERY coverage (added 2026-08-13 -- see docs/DESIGN.md): on
        # biz-beta specifically, not a new business, so this doesn't touch the
        # {"biz-alpha", "biz-beta", "biz-gamma"} set any existing test asserts.
        make_fake_review(
            "fake-009",
            "biz-beta",
            {Aspect.DELIVERY: Sentiment.NEGATIVE},
            overall_sentiment=Sentiment.NEGATIVE,
        ),
    ]


class SeverityCase(NamedTuple):
    """One review text paired with its expected ``severity_flag`` value only.

    Deliberately lighter than ``FakeReview``/``ReviewAnalysis`` -- these
    cases exist purely to test the severity judgment call, not aspect or
    sentiment extraction, so there's no aspect/sentiment ground truth to
    carry around.
    """

    review_id: str
    text: str
    severity_flag: bool


def generate_severity_dataset() -> list[SeverityCase]:
    """Return a fixed, deterministic set of reviews for testing ``severity_flag`` specifically.

    ``generate_fixed_dataset()`` has only one severity=True case out of
    eight reviews -- nowhere near enough signal to tell a real reliability
    problem from noise (see docs/DESIGN.md's "Known gap" entry, based on
    exactly that one-case denominator). This exists to fix that before any
    prompt change is attempted.

    Two groups, by design:
        - TRUE cases: genuine safety/health/legal issues, spread across
          different categories (foodborne illness, physical injury,
          health-code violation, harassment, fraud, allergic reaction,
          fire/evacuation, physical altercation) so a fix can't just
          overfit to one phrasing.
        - FALSE "near miss" cases: ordinary negative reviews using
          emotionally intense language ("disgusting", "nightmare",
          "worst ... of my life") about food/service/price/wait/ambience
          — the working hypothesis this dataset is built to test is that
          the model over-triggers on emotional intensity rather than
          genuine severity criteria (safety/health/legal). If false
          cases score much worse than true cases, that hypothesis holds.
    """
    return [
        # --- TRUE: genuine safety/health/legal issues ---
        SeverityCase(
            "sev-true-001",
            "I got severe food poisoning after eating here and had to go to the "
            "emergency room that same night.",
            True,
        ),
        SeverityCase(
            "sev-true-002",
            "I bit into my sandwich and there was a piece of glass in it that cut "
            "the inside of my mouth.",
            True,
        ),
        SeverityCase(
            "sev-true-003",
            "We saw a live rat run across the kitchen floor, in plain view of the "
            "dining room, while we were eating.",
            True,
        ),
        SeverityCase(
            "sev-true-004",
            "The manager screamed racial slurs at my family and told us to get out.",
            True,
        ),
        SeverityCase(
            "sev-true-005",
            "They charged my card twice without authorization and refused to reverse "
            "it when I called to dispute it.",
            True,
        ),
        SeverityCase(
            "sev-true-006",
            "My son has a severe peanut allergy. Despite telling the waiter twice, "
            "they served him food with peanuts in it and he had to use his EpiPen.",
            True,
        ),
        SeverityCase(
            "sev-true-007",
            "There was a small kitchen fire while we were seated, and staff never "
            "evacuated us or explained what was happening.",
            True,
        ),
        SeverityCase(
            "sev-true-008",
            "A cook came out of the kitchen and shoved my husband during an argument "
            "over the bill.",
            True,
        ),
        # --- FALSE: ordinary complaints in emotionally intense language ---
        SeverityCase(
            "sev-false-001",
            "Absolutely the worst meal of my life. I will never come back here again.",
            False,
        ),
        SeverityCase(
            "sev-false-002",
            "This is disgusting, the food tasted like it came out of a can.",
            False,
        ),
        SeverityCase(
            "sev-false-003",
            "We waited over an hour for a table and it was infuriating.",
            False,
        ),
        SeverityCase(
            "sev-false-004",
            "The service was a nightmare, our waiter was rude to us the entire time.",
            False,
        ),
        SeverityCase(
            "sev-false-005",
            "Way too expensive for what you get, total ripoff.",
            False,
        ),
        SeverityCase(
            "sev-false-006",
            "The place was filthy, I could barely enjoy my meal.",
            False,
        ),
        SeverityCase(
            "sev-false-007",
            "Terrible experience all around, would give zero stars if I could.",
            False,
        ),
        SeverityCase(
            "sev-false-008",
            "The music was way too loud and the tables were so cramped we could "
            "barely move.",
            False,
        ),
    ]
