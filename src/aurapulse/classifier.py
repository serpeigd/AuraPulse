"""Review classification via a local Ollama model.

No paid API is used — see docs/DESIGN.md for the zero-cost decision.
Structured output is enforced by passing ``ClassifiedAnalysis``'s JSON
schema to Ollama's ``format`` parameter, so the model's response is
constrained to that shape on the wire rather than merely prompted to
follow it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx
import ollama
from pydantic import ValidationError

from aurapulse.schemas import (
    Aspect,
    AspectMention,
    ClassifiedAnalysis,
    ReviewAnalysis,
    Sentiment,
)

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

# Fail fast if the server itself is unreachable (connect), but allow
# generous time for local generation on modest hardware (read).
_REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=10.0, pool=5.0)

_SYSTEM_PROMPT = """You are a review analysis engine for a restaurant reputation tool.

Given a single customer review, extract:
- overall_sentiment: positive, neutral, or negative — the reviewer's
  overall tone. Use neutral for lukewarm, so-so, or "nothing special"
  language — don't default to positive just because there's no
  explicit complaint.
- aspects: ONLY the restaurant aspects the review text actually
  discusses (food, service, price, cleanliness, wait_time, ambience,
  or other), each with its own local sentiment. Do NOT include an
  aspect the review doesn't mention, even to be thorough or complete —
  if the review only talks about two things, return exactly two
  aspect entries, not more. A single review can mention several
  aspects with different sentiment each (e.g. food positive,
  wait_time negative) — capture all of them, don't collapse to one
  aspect, but never invent ones that aren't there.
  "neutral" is a sentiment for an aspect the review actually discusses
  in a lukewarm or mixed way — it is NEVER a placeholder for an aspect
  the review doesn't discuss. If you're not sure whether an aspect is
  really being evaluated, leave it out entirely rather than adding it
  as neutral. A restaurant review not mentioning price, wait time, or
  ambience at all should produce zero entries for those, even though
  every restaurant technically has a price, a wait, and an ambience.
  Only use "other" when the content genuinely doesn't fit the other
  categories, and in that case fill other_detail with a short
  free-text label for what it actually is.
- severity_flag: true only if the review describes something a
  business owner would need to act on urgently (e.g. a safety,
  health, or legal issue). False for ordinary complaints about food,
  service, price, cleanliness, wait time, or ambience.

Respond with nothing but the structured JSON described by the schema.
"""

# Synthetic (not real Yelp text -- classifier.py ships in the public repo,
# and data/processed/*.csv is gitignored precisely to keep dataset content
# out of it, see docs/DESIGN.md) few-shot pairs demonstrating restraint.
#
# Both target the failure mode found in the 2026-08-05 aspect-proxy eval
# (100 hand-labeled reviews, 36% aspect-set match / 24% aspect+sentiment
# match): the model was adding aspects the review never discusses, almost
# always tagged "neutral" -- 39 of 62 false-positive aspect predictions in
# price/wait_time/ambience/other were "neutral" filler. See docs/DESIGN.md
# for the full before/after writeup.
_FEW_SHOT_EXAMPLES: list[tuple[str, ClassifiedAnalysis]] = [
    (
        (
            "Best tacos in town, hands down. My friend and I split three plates and "
            "could not stop talking about the salsa verde."
        ),
        ClassifiedAnalysis(
            overall_sentiment=Sentiment.POSITIVE,
            aspects=[AspectMention(aspect=Aspect.FOOD, sentiment=Sentiment.POSITIVE)],
        ),
    ),
    (
        (
            "The pasta was outstanding, but our waiter forgot to bring water twice and "
            "never checked back on us. Everything else about the visit was unremarkable."
        ),
        ClassifiedAnalysis(
            # Mixed positive+negative aspects -> overall NEGATIVE, per the
            # ground-truth convention in docs/DESIGN.md -- kept consistent
            # here so the few-shot answer doesn't contradict eval labels.
            overall_sentiment=Sentiment.NEGATIVE,
            aspects=[
                AspectMention(aspect=Aspect.FOOD, sentiment=Sentiment.POSITIVE),
                AspectMention(aspect=Aspect.SERVICE, sentiment=Sentiment.NEGATIVE),
            ],
        ),
    ),
]


def _build_messages(text: str) -> list[dict[str, str]]:
    """Assemble the chat turns sent to Ollama: system prompt, few-shot pairs, then the review.

    Kept as its own function (rather than inlined in
    ``_request_completion``) so tests can assert on the few-shot content
    without mocking a live chat call.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for example_text, example_output in _FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": example_output.model_dump_json()})
    messages.append({"role": "user", "content": text})
    return messages


class ClassificationError(RuntimeError):
    """Raised when a review can't be classified: server unreachable,
    or the model never returns schema-valid output within the retry
    budget."""


def _emit_trace(
    review_id: str,
    model: str,
    *,
    attempts: int,
    elapsed_ms: float,
    outcome: str,
    error: str | None = None,
    other_detail_stripped: int = 0,
) -> None:
    """Emit one structured JSON trace line for a completed ``classify_review`` call.

    Zero-cost, zero-new-dependency observability (see CLAUDE.md's no-paid-
    services rule): every call emits exactly one line, machine-parseable
    with ``grep '"event": "classification"' | jq``. No dashboard or
    external service -- this is meant to make retry/latency/failure-rate
    questions answerable from a log file, not to be a production APM.
    Deliberately excludes review/business text and full error tracebacks
    (only a short error string) to keep trace lines safe to keep around
    without becoming a second copy of dataset content.
    """
    payload: dict[str, Any] = {
        "event": "classification",
        "review_id": review_id,
        "model": model,
        "attempts": attempts,
        "elapsed_ms": round(elapsed_ms),
        "outcome": outcome,
    }
    if error is not None:
        payload["error"] = error
    if other_detail_stripped:
        payload["other_detail_stripped"] = other_detail_stripped
    logger.info(json.dumps(payload))


def _normalize_other_detail(raw: dict[str, Any], review_id: str) -> tuple[dict[str, Any], int]:
    """Drop ``other_detail`` from any aspect mention that isn't OTHER.

    The model occasionally attaches a clarifying note to a *named* aspect
    (e.g. ``{"aspect": "price", "other_detail": "menu transparency"}``)
    instead of leaving the field unset. ``AspectMention``'s validator
    correctly rejects that shape (see ``src/aurapulse/schemas.py``), but
    discarding the whole review over one extraneous string throws away an
    aspect+sentiment call that's very likely still correct. A system-prompt
    instruction explicitly forbidding this was tried and made the failure
    rate *worse*, not better (see docs/DESIGN.md) — this is a deterministic,
    zero-inference-cost fix instead: strip the stray field before
    validating, keep everything else the model produced as-is.

    Mutates and returns ``raw`` for convenience; also returns how many
    mentions were stripped, so the caller can log/trace it.
    """
    stripped = 0
    for mention in raw.get("aspects", []):
        if mention.get("aspect") != Aspect.OTHER.value and mention.get("other_detail"):
            mention["other_detail"] = None
            stripped += 1
    if stripped:
        logger.warning(
            "review_id=%s: stripped other_detail from %d aspect mention(s) that weren't OTHER",
            review_id,
            stripped,
        )
    return raw, stripped


def _request_completion(
    client: ollama.Client, model: str, host: str, text: str, response_schema: dict[str, Any]
) -> ollama.ChatResponse:
    """Send one chat request to Ollama, wrapping connection failures.

    Raises:
        ClassificationError: if the server is unreachable, the model
            isn't pulled, or any other transport-level failure occurs.
    """
    try:
        return client.chat(
            model=model,
            messages=_build_messages(text),
            format=response_schema,
            options={"temperature": 0},
        )
    except Exception as exc:  # Ollama unreachable, model not pulled, etc.
        raise ClassificationError(
            f"could not get a response from Ollama at {host!r} with model {model!r}: {exc}"
        ) from exc


def classify_review(
    review_id: str,
    business_id: str,
    text: str,
    *,
    model: str | None = None,
    host: str | None = None,
    max_retries: int = 2,
) -> ReviewAnalysis:
    """Classify a single review's sentiment and aspects via a local Ollama model.

    Emits exactly one structured JSON trace line (via ``_emit_trace``) per
    call at INFO level, regardless of outcome — see that function's
    docstring for the log-based observability rationale.

    Args:
        review_id: Identifier to embed in the returned analysis (not
            sent to the model — see ``ClassifiedAnalysis``).
        business_id: Identifier to embed in the returned analysis.
        text: Raw review text to classify.
        model: Ollama model tag. Defaults to $OLLAMA_MODEL, then
            "llama3.1:8b".
        host: Ollama server URL. Defaults to $OLLAMA_HOST, then
            "http://localhost:11434".
        max_retries: Extra attempts if the model's response doesn't
            validate against the schema (e.g. malformed JSON, or an
            other_detail/aspect mismatch — see
            ``AspectMention``). Does NOT retry on connection failures
            — the server being unreachable isn't something a retry
            with the same call will fix, so that fails fast as a
            single ClassificationError.

    Returns:
        A validated ``ReviewAnalysis`` with ``review_id``/``business_id``
        set to the values passed in.

    Raises:
        ClassificationError: if Ollama is unreachable, or the model
            doesn't return schema-valid output within ``max_retries``.
    """
    resolved_model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    resolved_host = host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    client = ollama.Client(host=resolved_host, timeout=_REQUEST_TIMEOUT)
    response_schema = ClassifiedAnalysis.model_json_schema()
    start = time.monotonic()

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = _request_completion(client, resolved_model, resolved_host, text, response_schema)
        except ClassificationError as exc:
            _emit_trace(
                review_id,
                resolved_model,
                attempts=attempt + 1,
                elapsed_ms=(time.monotonic() - start) * 1000,
                outcome="connection_error",
                error=str(exc),
            )
            raise
        other_detail_stripped = 0
        try:
            raw = json.loads(response["message"]["content"])
            raw, other_detail_stripped = _normalize_other_detail(raw, review_id)
            parsed = ClassifiedAnalysis.model_validate(raw)
        except (ValidationError, KeyError, TypeError, ValueError, AttributeError) as exc:  # malformed/invalid JSON, schema mismatch
            last_error = exc
            logger.warning(
                "review_id=%s: attempt %d/%d returned schema-invalid output: %s",
                review_id,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            continue

        logger.debug("review_id=%s: classified successfully on attempt %d", review_id, attempt + 1)
        _emit_trace(
            review_id,
            resolved_model,
            attempts=attempt + 1,
            elapsed_ms=(time.monotonic() - start) * 1000,
            outcome="success",
            other_detail_stripped=other_detail_stripped,
        )
        return ReviewAnalysis(review_id=review_id, business_id=business_id, **parsed.model_dump())

    logger.error(
        "review_id=%s: model %r gave no schema-valid output after %d attempt(s)",
        review_id,
        resolved_model,
        max_retries + 1,
    )
    _emit_trace(
        review_id,
        resolved_model,
        attempts=max_retries + 1,
        elapsed_ms=(time.monotonic() - start) * 1000,
        outcome="schema_invalid",
        error=str(last_error),
    )
    raise ClassificationError(
        f"model {resolved_model!r} did not return schema-valid output after "
        f"{max_retries + 1} attempt(s): {last_error}"
    )
