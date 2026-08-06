"""Unit tests for decide_route -- pure function, no mocking needed."""

from __future__ import annotations

import pytest

from aurapulse.routing import decide_route
from aurapulse.schemas import ReviewAnalysis, Route, Sentiment


def _analysis(sentiment: Sentiment, severity_flag: bool = False) -> ReviewAnalysis:
    return ReviewAnalysis(
        review_id="r1",
        business_id="b1",
        overall_sentiment=sentiment,
        aspects=[],
        severity_flag=severity_flag,
    )


@pytest.mark.parametrize(
    ("sentiment", "severity_flag", "expected"),
    [
        (Sentiment.POSITIVE, False, Route.AGGREGATE),
        (Sentiment.POSITIVE, True, Route.AGGREGATE),  # severity_flag ignored unless negative
        (Sentiment.NEUTRAL, False, Route.AGGREGATE),
        (Sentiment.NEUTRAL, True, Route.AGGREGATE),  # same -- neutral never drafts or escalates
        (Sentiment.NEGATIVE, False, Route.DRAFT_RESPONSE),
        (Sentiment.NEGATIVE, True, Route.ESCALATE),
    ],
)
def test_decide_route(sentiment: Sentiment, severity_flag: bool, expected: Route) -> None:
    assert decide_route(_analysis(sentiment, severity_flag)) == expected
