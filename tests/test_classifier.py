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

from aurapulse.classifier import (
    _FEW_SHOT_EXAMPLES,
    _SYSTEM_PROMPT,
    ClassificationError,
    _build_messages,
    _emit_trace,
    classify_review,
)
from aurapulse.schemas import Aspect, ClassifiedAnalysis, Sentiment


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


# --- Few-shot prompt construction (2026-08-05 aspect-precision fix, see docs/DESIGN.md) ---


def test_few_shot_examples_are_schema_valid() -> None:
    """Each few-shot answer must itself validate against ClassifiedAnalysis.

    Guards against the few-shot examples silently drifting out of sync
    with the schema they're meant to be teaching the model to follow.
    """
    for _, example_output in _FEW_SHOT_EXAMPLES:
        # Round-trip through JSON, same path the model's real output takes.
        ClassifiedAnalysis.model_validate_json(example_output.model_dump_json())


def test_few_shot_examples_never_use_neutral_for_omitted_aspects() -> None:
    """Regression guard for the exact failure mode the few-shot pairs target.

    39/62 false positives in the 2026-08-05 aspect-proxy eval were
    aspects tagged "neutral" that the review never discussed -- the
    few-shot answers must not model that pattern themselves.
    """
    for _, example_output in _FEW_SHOT_EXAMPLES:
        assert all(mention.sentiment != Sentiment.NEUTRAL for mention in example_output.aspects)


def test_build_messages_structure() -> None:
    """The real review must come last, after system + every few-shot pair, unmodified."""
    messages = _build_messages("The tacos were fine, nothing special.")

    assert messages[0] == {"role": "system", "content": _SYSTEM_PROMPT}

    body = messages[1:-1]
    assert len(body) == 2 * len(_FEW_SHOT_EXAMPLES)
    for i, (example_text, example_output) in enumerate(_FEW_SHOT_EXAMPLES):
        assert body[2 * i] == {"role": "user", "content": example_text}
        assert body[2 * i + 1] == {"role": "assistant", "content": example_output.model_dump_json()}

    assert messages[-1] == {"role": "user", "content": "The tacos were fine, nothing special."}


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_sends_few_shot_examples(mock_client_cls: MagicMock) -> None:
    """End-to-end: classify_review must route through _build_messages, not a bare 2-turn prompt."""
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_valid_payload())

    classify_review("r1", "b1", "Great food!")

    sent_messages = mock_client.chat.call_args.kwargs["messages"]
    assert len(sent_messages) == 2 + 2 * len(_FEW_SHOT_EXAMPLES)
    assert sent_messages[-1]["content"] == "Great food!"


# --- Observability: one structured trace line per call (2026-08-05, see docs/DESIGN.md) ---


def _trace_payloads(caplog: pytest.LogCaptureFixture) -> list[dict]:
    """Parse every classification-trace JSON line emitted during a test."""
    return [json.loads(r.message) for r in caplog.records if r.message.startswith('{"event": "classification"')]


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_emits_success_trace(mock_client_cls: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_valid_payload())

    with caplog.at_level("INFO", logger="aurapulse.classifier"):
        classify_review("r1", "b1", "Great food!")

    traces = _trace_payloads(caplog)
    assert len(traces) == 1
    assert traces[0]["review_id"] == "r1"
    assert traces[0]["outcome"] == "success"
    assert traces[0]["attempts"] == 1
    assert traces[0]["elapsed_ms"] >= 0
    assert "error" not in traces[0]


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_emits_schema_invalid_trace_after_exhausting_retries(
    mock_client_cls: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response("not valid json")

    with caplog.at_level("INFO", logger="aurapulse.classifier"), pytest.raises(ClassificationError):
        classify_review("r1", "b1", "Great food!", max_retries=1)

    traces = _trace_payloads(caplog)
    assert len(traces) == 1  # exactly one trace line for the whole call, not one per attempt
    assert traces[0]["outcome"] == "schema_invalid"
    assert traces[0]["attempts"] == 2
    assert "error" in traces[0]


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_emits_connection_error_trace(
    mock_client_cls: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.side_effect = ConnectionError("connection refused")

    with caplog.at_level("INFO", logger="aurapulse.classifier"), pytest.raises(ClassificationError):
        classify_review("r1", "b1", "Great food!", max_retries=2)

    traces = _trace_payloads(caplog)
    assert len(traces) == 1
    assert traces[0]["outcome"] == "connection_error"
    assert traces[0]["attempts"] == 1  # fails fast, no retries burned


def test_trace_never_includes_review_text(caplog: pytest.LogCaptureFixture) -> None:
    """The trace payload must stay safe to keep around -- no review/business text, just metadata."""
    with caplog.at_level("INFO", logger="aurapulse.classifier"):
        _emit_trace("r1", "llama3.1:8b", attempts=1, elapsed_ms=12.3, outcome="success")

    payload = json.loads(caplog.records[0].message)
    assert set(payload) <= {"event", "review_id", "model", "attempts", "elapsed_ms", "outcome", "error"}


# --- other_detail normalization (2026-08-06, see docs/DESIGN.md) ---
#
# A prompt-only fix for the model attaching other_detail to a named aspect
# was tried first and made the failure rate worse (36%/24% baseline ->
# 43%/33% few-shot-only -> 40%/28% with the extra prompt sentence). These
# tests cover the code-level fix that replaced it: strip the stray field
# instead of rejecting the whole review.


def _payload_with_stray_other_detail() -> str:
    return json.dumps(
        {
            "overall_sentiment": "neutral",
            "aspects": [{"aspect": "price", "sentiment": "neutral", "other_detail": "menu transparency"}],
            "severity_flag": False,
        }
    )


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_strips_other_detail_on_named_aspect(mock_client_cls: MagicMock) -> None:
    """The exact shape that used to raise ClassificationError after 3 attempts now succeeds on the first."""
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_payload_with_stray_other_detail())

    result = classify_review("r1", "b1", "The menu doesn't list prices up front.")

    assert result.aspects[0].aspect == Aspect.PRICE
    assert result.aspects[0].other_detail is None
    mock_client.chat.assert_called_once()  # no retry burned


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_traces_other_detail_stripped_count(
    mock_client_cls: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(_payload_with_stray_other_detail())

    with caplog.at_level("INFO", logger="aurapulse.classifier"):
        classify_review("r1", "b1", "The menu doesn't list prices up front.")

    traces = _trace_payloads(caplog)
    assert traces[0]["other_detail_stripped"] == 1


@patch("aurapulse.classifier.ollama.Client")
def test_classify_review_preserves_other_detail_on_real_other_aspect(mock_client_cls: MagicMock) -> None:
    """Regression guard: normalization must not strip a legitimately-used OTHER aspect's detail."""
    mock_client = mock_client_cls.return_value
    mock_client.chat.return_value = _chat_response(
        json.dumps(
            {
                "overall_sentiment": "negative",
                "aspects": [{"aspect": "other", "sentiment": "negative", "other_detail": "renaming of locations"}],
                "severity_flag": False,
            }
        )
    )

    result = classify_review("r1", "b1", "They keep renaming this place, hard to find on the app.")

    assert result.aspects[0].aspect == Aspect.OTHER
    assert result.aspects[0].other_detail == "renaming of locations"
