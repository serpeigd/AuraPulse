"""Unit tests for draft-reply generation and escalation flagging.

Same posture as test_classifier.py: mock the Ollama client, test control
flow only, say nothing about draft *quality* -- see
scripts/eval_draft_responses.py for that.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aurapulse.classifier import ClassificationError
from aurapulse.response_draft import (
    _FEW_SHOT_EXAMPLES,
    flag_for_escalation,
    generate_draft_response,
)
from aurapulse.schemas import (
    Aspect,
    AspectMention,
    DraftText,
    ReviewAnalysis,
    Sentiment,
)

_BANNED_BOILERPLATE_PHRASES = (
    "we take this seriously",
    "we take all feedback",
    "we take all complaints",
    "thank you for bringing this to our attention",
    "thank you for your feedback",
)


def _analysis(**aspect_sentiments: Sentiment) -> ReviewAnalysis:
    aspects = [AspectMention(aspect=Aspect(name), sentiment=sentiment) for name, sentiment in aspect_sentiments.items()]
    return ReviewAnalysis(
        review_id="r1",
        business_id="b1",
        overall_sentiment=Sentiment.NEGATIVE,
        aspects=aspects,
        severity_flag=False,
    )


def _valid_draft_payload(text: str = "Thanks for the feedback, we're sorry to hear this. — The team") -> str:
    return json.dumps({"draft_text": text})


def _chat_response(content: str) -> dict:
    return {"message": {"content": content}}


# --- generate_draft_response ---


@patch("aurapulse.response_draft.ollama.Client")
def test_generate_draft_response_success(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_valid_draft_payload())

    result = generate_draft_response("r1", "b1", "The wait was way too long.", _analysis(wait_time=Sentiment.NEGATIVE))

    assert result.review_id == "r1"
    assert result.business_id == "b1"
    assert "sorry" in result.draft_text.lower()
    mock_client.chat.assert_called_once()


@patch("aurapulse.response_draft.ollama.Client")
def test_generate_draft_response_includes_only_negative_aspects_in_prompt(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_valid_draft_payload())

    generate_draft_response(
        "r1", "b1", "Food was great, service was slow.", _analysis(food=Sentiment.POSITIVE, service=Sentiment.NEGATIVE)
    )

    user_content = mock_client.chat.call_args.kwargs["messages"][-1]["content"]
    assert "service" in user_content
    assert "food" not in user_content.split("criticized:")[1]  # positive aspect excluded from the steering hint


@patch("aurapulse.response_draft.ollama.Client")
def test_generate_draft_response_retries_on_invalid_json_then_succeeds(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.side_effect = [
        _chat_response("not valid json"),
        _chat_response(_valid_draft_payload()),
    ]

    result = generate_draft_response("r1", "b1", "Too slow.", _analysis(wait_time=Sentiment.NEGATIVE), max_retries=2)

    assert result.draft_text
    assert mock_client.chat.call_count == 2


@patch("aurapulse.response_draft.ollama.Client")
def test_generate_draft_response_raises_after_exhausting_retries(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response("not valid json")

    with pytest.raises(ClassificationError):
        generate_draft_response("r1", "b1", "Too slow.", _analysis(wait_time=Sentiment.NEGATIVE), max_retries=1)

    assert mock_client.chat.call_count == 2


@patch("aurapulse.response_draft.ollama.Client")
def test_generate_draft_response_fails_fast_on_connection_error(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.side_effect = ConnectionError("connection refused")

    with pytest.raises(ClassificationError):
        generate_draft_response("r1", "b1", "Too slow.", _analysis(wait_time=Sentiment.NEGATIVE), max_retries=2)

    mock_client.chat.assert_called_once()


@patch("aurapulse.response_draft.ollama.Client")
def test_generate_draft_response_folds_feedback_into_the_prompt_when_given(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_valid_draft_payload())

    generate_draft_response(
        "r1", "b1", "Too slow.", _analysis(wait_time=Sentiment.NEGATIVE), feedback="too generic, be specific"
    )

    user_content = mock_client.chat.call_args.kwargs["messages"][-1]["content"]
    assert "too generic, be specific" in user_content


@patch("aurapulse.response_draft.ollama.Client")
def test_generate_draft_response_omits_feedback_note_when_none_given(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_valid_draft_payload())

    generate_draft_response("r1", "b1", "Too slow.", _analysis(wait_time=Sentiment.NEGATIVE))

    user_content = mock_client.chat.call_args.kwargs["messages"][-1]["content"]
    assert "rejected your previous draft" not in user_content


@patch("aurapulse.response_draft.ollama.Client")
def test_generate_draft_response_emits_trace(mock_client_cls: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_valid_draft_payload())

    with caplog.at_level("INFO", logger="aurapulse.response_draft"):
        generate_draft_response("r1", "b1", "Too slow.", _analysis(wait_time=Sentiment.NEGATIVE))

    traces = [json.loads(r.message) for r in caplog.records if r.message.startswith('{"event": "draft_generation"')]
    assert len(traces) == 1
    assert traces[0]["outcome"] == "success"


# --- few-shot examples (2026-08-10 genericness fix, see docs/DESIGN.md) ---


def test_few_shot_examples_are_schema_valid() -> None:
    for _, example_draft in _FEW_SHOT_EXAMPLES:
        DraftText.model_validate_json(example_draft.model_dump_json())


def test_few_shot_examples_avoid_the_banned_boilerplate_phrases() -> None:
    """Regression guard: the examples must model what they're teaching, not the failure mode.

    Every one of the 14 drafts in the 2026-08-09 human-validated quality eval
    read as generic, wrapped in phrases like "we take this seriously" -- the
    few-shot answers must not reproduce that pattern themselves.
    """
    for _, example_draft in _FEW_SHOT_EXAMPLES:
        lowered = example_draft.draft_text.lower()
        for phrase in _BANNED_BOILERPLATE_PHRASES:
            assert phrase not in lowered, f"{phrase!r} found in few-shot draft: {example_draft.draft_text!r}"


@patch("aurapulse.response_draft.ollama.Client")
def test_generate_draft_response_sends_few_shot_examples(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_valid_draft_payload())

    generate_draft_response("r1", "b1", "Too slow.", _analysis(wait_time=Sentiment.NEGATIVE))

    sent_messages = mock_client.chat.call_args.kwargs["messages"]
    assert len(sent_messages) == 2 + 2 * len(_FEW_SHOT_EXAMPLES)
    assert "Too slow." in sent_messages[-1]["content"]


# --- flag_for_escalation ---


def test_flag_for_escalation_lists_negative_aspects() -> None:
    analysis = _analysis(cleanliness=Sentiment.NEGATIVE, service=Sentiment.NEGATIVE)
    analysis.severity_flag = True

    flag = flag_for_escalation(analysis)

    assert flag.review_id == "r1"
    assert flag.business_id == "b1"
    assert "cleanliness" in flag.reason
    assert "service" in flag.reason


def test_flag_for_escalation_without_negative_aspects_has_fallback_reason() -> None:
    analysis = ReviewAnalysis(
        review_id="r2", business_id="b1", overall_sentiment=Sentiment.NEGATIVE, aspects=[], severity_flag=True
    )

    flag = flag_for_escalation(analysis)

    assert "no specific aspect" in flag.reason.lower()
