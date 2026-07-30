"""Review classification via a local Ollama model.

No paid API is used — see docs/DESIGN.md for the zero-cost decision.
Structured output is enforced by passing ``ClassifiedAnalysis``'s JSON
schema to Ollama's ``format`` parameter, so the model's response is
constrained to that shape on the wire rather than merely prompted to
follow it.
"""

from __future__ import annotations

import os

import httpx
import ollama

from aurapulse.schemas import ClassifiedAnalysis, ReviewAnalysis

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
  Only use "other" when the content genuinely doesn't fit the other
  categories, and in that case fill other_detail with a short
  free-text label for what it actually is.
- severity_flag: true only if the review describes something a
  business owner would need to act on urgently (e.g. a safety,
  health, or legal issue). False for ordinary complaints about food,
  service, price, cleanliness, wait time, or ambience.

Respond with nothing but the structured JSON described by the schema.
"""


class ClassificationError(RuntimeError):
    """Raised when a review can't be classified: server unreachable,
    or the model never returns schema-valid output within the retry
    budget."""


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
            validate against the schema (e.g. malformed JSON). Does
            NOT retry on connection failures — the server being
            unreachable isn't something a retry with the same call
            will fix, so that fails fast as a single ClassificationError.

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

    try:
        response = client.chat(
            model=resolved_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            format=response_schema,
            options={"temperature": 0},
        )
    except Exception as exc:  # Ollama unreachable, model not pulled, etc.
        raise ClassificationError(
            f"could not get a response from Ollama at {resolved_host!r} "
            f"with model {resolved_model!r}: {exc}"
        ) from exc

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            parsed = ClassifiedAnalysis.model_validate_json(response["message"]["content"])
            return ReviewAnalysis(
                review_id=review_id,
                business_id=business_id,
                **parsed.model_dump(),
            )
        except Exception as exc:  # invalid JSON, schema mismatch, etc.
            last_error = exc
            if attempt == max_retries:
                break
            try:
                response = client.chat(
                    model=resolved_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    format=response_schema,
                    options={"temperature": 0},
                )
            except Exception as retry_exc:
                raise ClassificationError(
                    f"could not get a response from Ollama at {resolved_host!r} "
                    f"with model {resolved_model!r}: {retry_exc}"
                ) from retry_exc

    raise ClassificationError(
        f"model {resolved_model!r} did not return schema-valid output after "
        f"{max_retries + 1} attempt(s): {last_error}"
    )
