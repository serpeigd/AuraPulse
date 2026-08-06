"""Integration-style tests for process_reviews using the deterministic fake dataset.

Mocks only the LLM boundary (draft generation) -- routing, escalation,
and aggregation all run for real, since none of them touch a model.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aurapulse.fake_reviews import generate_fixed_dataset
from aurapulse.orchestrator import process_reviews
from aurapulse.schemas import Aspect, AspectMention, ReviewAnalysis, Sentiment


def _dataset() -> tuple[list[ReviewAnalysis], dict[str, str]]:
    fake_reviews = generate_fixed_dataset()
    analyses = [fr.expected for fr in fake_reviews]
    texts = {fr.expected.review_id: fr.text for fr in fake_reviews}
    return analyses, texts


def _valid_draft_payload() -> str:
    return json.dumps({"draft_text": "Sorry to hear that. — The team"})


def _chat_response(content: str) -> dict:
    return {"message": {"content": content}}


@patch("aurapulse.response_draft.ollama.Client")
def test_process_reviews_routes_the_fixed_dataset_correctly(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_valid_draft_payload())

    analyses, texts = _dataset()
    batch = process_reviews(analyses, texts)

    # fake-008 is the only severity_flag=True review in the fixed dataset -> ESCALATE.
    assert {e.review_id for e in batch.escalations} == {"fake-008"}
    # Every other NEGATIVE review (002, 004, 005, 006, 007) -> DRAFT_RESPONSE.
    assert {d.review_id for d in batch.drafts} == {"fake-002", "fake-004", "fake-005", "fake-006", "fake-007"}
    # Aggregation covers ALL reviews, including escalated/drafted ones.
    assert sum(r.review_count for r in batch.business_reports) == len(analyses)
    assert batch.draft_failures == []


@patch("aurapulse.response_draft.ollama.Client")
def test_process_reviews_records_draft_failures_without_dropping_the_batch(mock_client_cls: MagicMock) -> None:
    from aurapulse.classifier import ClassificationError

    mock_client = mock_client_cls.return_value
    mock_client.chat.side_effect = ClassificationError("boom")

    analyses, texts = _dataset()
    batch = process_reviews(analyses, texts)

    assert set(batch.draft_failures) == {"fake-002", "fake-004", "fake-005", "fake-006", "fake-007"}
    assert batch.drafts == []
    # Aggregation and escalation are unaffected by draft failures.
    assert {e.review_id for e in batch.escalations} == {"fake-008"}
    assert sum(r.review_count for r in batch.business_reports) == len(analyses)


def test_process_reviews_raises_if_a_drafted_review_text_is_missing() -> None:
    analyses, texts = _dataset()
    del texts["fake-002"]  # fake-002 is NEGATIVE/non-severe -> would be routed to DRAFT_RESPONSE

    with pytest.raises(KeyError):
        process_reviews(analyses, texts)


def test_process_reviews_handles_an_all_positive_batch_with_no_llm_calls() -> None:
    positive = ReviewAnalysis(
        review_id="p1",
        business_id="b1",
        overall_sentiment=Sentiment.POSITIVE,
        aspects=[AspectMention(aspect=Aspect.FOOD, sentiment=Sentiment.POSITIVE)],
        severity_flag=False,
    )

    batch = process_reviews([positive], {})  # no review_texts needed -- nothing gets drafted

    assert batch.drafts == []
    assert batch.escalations == []
    assert batch.draft_failures == []
    assert batch.business_reports[0].review_count == 1
