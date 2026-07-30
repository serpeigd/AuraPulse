"""Unit tests for the Ollama-backed classifier's control flow.

These mock the Ollama client and test retry/error-handling logic only
— they do NOT require a running Ollama server and say nothing about
model output quality. Quality is measured separately, against a live
model, by scripts/eval_fake_reviews.py.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aurapulse.classifier import ClassificationError, classify_review
from aurapulse.schemas import Aspect, Sentiment


def _valid_payload() -> str:
    return json.dumps(
        {
            "overall_sentiment": "positive",
            "aspects": [{"aspect": "food", "sentiment": "positive", "other_detail": None}],
            "severity_flag": False,
        }
    )


def _chat_response(content: str) -> dict:
    return {"message": {"content": content}}


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_success(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_valid_payload())

    result = classify_review("r1", "b1", "Great food!")

    assert result.review_id == "r1"
    assert result.business_id == "b1"
    assert result.overall_sentiment == Sentiment.POSITIVE
    assert result.aspects[0].aspect == Aspect.FOOD
    mock_client.chat.assert_called_once()


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_retries_on_invalid_json_then_succeeds(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.side_effect = [
        _chat_response("not valid json"),
        _chat_response(_valid_payload()),
    ]

    result = classify_review("r1", "b1", "Great food!", max_retries=2)

    assert result.overall_sentiment == Sentiment.POSITIVE
    assert mock_client.chat.call_count == 2


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_raises_after_exhausting_retries(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response("not valid json")

    with pytest.raises(ClassificationError):
        classify_review("r1", "b1", "Great food!", max_retries=1)

    assert mock_client.chat.call_count == 2  # initial attempt + 1 retry


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_fails_fast_on_connection_error(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.side_effect = ConnectionError("connection refused")

    with pytest.raises(ClassificationError):
        classify_review("r1", "b1", "Great food!", max_retries=2)

    mock_client.chat.assert_called_once()
