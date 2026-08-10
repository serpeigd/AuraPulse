"""Unit tests for the LLM-as-judge draft quality assessment.

Same posture as test_classifier.py/test_response_draft.py: mock the
Ollama client, test control flow only. Says nothing about whether the
judge's verdicts agree with a human -- that's the point of the
human-validation pass documented in docs/DESIGN.md, not something a unit
test with a mocked, self-consistent payload could ever demonstrate.

``addresses_specific_complaint`` is included in every payload here even
though it's not a trusted field (see DraftQualityVerdict's docstring) --
it's still part of the schema and must round-trip correctly.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aurapulse.classifier import ClassificationError
from aurapulse.draft_judge import _JUDGE_SYSTEM_PROMPT, judge_draft


def _verdict_payload(**overrides: bool | str) -> str:
    payload = {
        "addresses_specific_complaint": True,
        "appropriate_tone": True,
        "usable_with_minor_edits": True,
        "not_generic": True,
        "reasoning": "Directly acknowledges the wait-time complaint in a professional tone.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _chat_response(content: str) -> dict:
    return {"message": {"content": content}}


@patch("aurapulse.draft_judge.ollama.Client")
def test_judge_draft_success(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_verdict_payload())

    verdict = judge_draft("We waited 45 minutes.", "Sorry about the wait — the team.")

    assert verdict.appropriate_tone is True
    assert verdict.usable_with_minor_edits is True
    assert verdict.reasoning
    mock_client.chat.assert_called_once()


@patch("aurapulse.draft_judge.ollama.Client")
def test_judge_draft_can_return_mixed_verdict(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_verdict_payload(not_generic=False, usable_with_minor_edits=False))

    verdict = judge_draft("We waited 45 minutes.", "We're sorry you had a bad experience with us.")

    assert verdict.not_generic is False
    assert verdict.usable_with_minor_edits is False


@patch("aurapulse.draft_judge.ollama.Client")
def test_judge_draft_retries_on_invalid_json_then_succeeds(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.side_effect = [
        _chat_response("not valid json"),
        _chat_response(_verdict_payload()),
    ]

    verdict = judge_draft("Too slow.", "Sorry — the team.", max_retries=2)

    assert verdict.reasoning
    assert mock_client.chat.call_count == 2


@patch("aurapulse.draft_judge.ollama.Client")
def test_judge_draft_raises_after_exhausting_retries(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response("not valid json")

    with pytest.raises(ClassificationError):
        judge_draft("Too slow.", "Sorry — the team.", max_retries=1)

    assert mock_client.chat.call_count == 2


@patch("aurapulse.draft_judge.ollama.Client")
def test_judge_draft_fails_fast_on_connection_error(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.side_effect = ConnectionError("connection refused")

    with pytest.raises(ClassificationError):
        judge_draft("Too slow.", "Sorry — the team.", max_retries=2)

    mock_client.chat.assert_called_once()


@patch("aurapulse.draft_judge.ollama.Client")
def test_judge_draft_sends_both_review_and_draft_text(mock_client_cls: MagicMock) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_verdict_payload())

    judge_draft("The wait was way too long.", "Sorry about the delay — the team.")

    user_content = mock_client.chat.call_args.kwargs["messages"][1]["content"]
    assert "The wait was way too long." in user_content
    assert "Sorry about the delay — the team." in user_content


def test_prompt_still_asks_all_four_questions() -> None:
    """Regression guard: do not remove addresses_specific_complaint from the prompt.

    Dropping that question was tried and measurably destabilized
    appropriate_tone on the same drafts (14/14 True -> 0/14 True) even
    though its own wording never changed -- see DraftQualityVerdict's
    docstring and docs/DESIGN.md. The validated prompt shape must be
    kept intact even though this field's value isn't trusted.
    """
    for keyword in ["addresses_specific_complaint", "appropriate_tone", "usable_with_minor_edits", "not_generic"]:
        assert keyword in _JUDGE_SYSTEM_PROMPT
