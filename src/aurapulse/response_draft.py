"""Draft-reply generation and escalation flagging for Hito 1 routing.

Two very different kinds of "action" live here on purpose, both reachable
from ``routing.decide_route``'s output:

- ``generate_draft_response`` calls a local Ollama model (zero-cost, same
  posture as ``classifier.py``) to write a reply for a human to review.
  Per CLAUDE.md's non-negotiable rule, this is a draft only — nothing in
  this module (or anywhere else in the codebase) can send or publish a
  review reply. That's enforced by the capability not existing, not by a
  flag anyone could flip.
- ``flag_for_escalation`` needs no LLM at all: severity is already known
  from classification, so escalating is just formatting a reason string.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import ollama
from pydantic import ValidationError

from aurapulse.classifier import (
    _REQUEST_TIMEOUT,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    ClassificationError,
)
from aurapulse.schemas import (
    DraftResponse,
    DraftText,
    EscalationFlag,
    ReviewAnalysis,
    Sentiment,
)

logger = logging.getLogger(__name__)

_DRAFT_SYSTEM_PROMPT = """You are drafting a reply to a negative customer review, for a \
restaurant owner to read, edit, and send themselves. You never send anything yourself — \
you only produce a draft for a human to approve.

Write a short, professional, empathetic reply that:
- Acknowledges the specific complaint(s) in the review, using details from the review
  text itself. Do not write something generic that could apply to any negative review.
- Does NOT promise a specific remedy (refund, discount, free item, replacement) — that
  decision belongs to the owner, not you.
- Does NOT claim the business has already fixed the issue — you have no way of knowing
  that, and a false claim would be worse than no claim.
- Stays under 100 words.
- Signs off generically (e.g. "— The team") — never invent an owner's or manager's name.

Respond with nothing but the structured JSON described by the schema.
"""


def _emit_trace(
    event: str,
    review_id: str,
    model: str,
    *,
    attempts: int,
    elapsed_ms: float,
    outcome: str,
    error: str | None = None,
) -> None:
    """Same structured-JSON tracing approach as ``classifier._emit_trace``.

    Deliberately duplicated rather than shared (see docs/DESIGN.md) —
    this is the second LLM-call site, not yet the third; extract a common
    helper if a third one shows up.
    """
    payload: dict[str, Any] = {
        "event": event,
        "review_id": review_id,
        "model": model,
        "attempts": attempts,
        "elapsed_ms": round(elapsed_ms),
        "outcome": outcome,
    }
    if error is not None:
        payload["error"] = error
    logger.info(json.dumps(payload))


def generate_draft_response(
    review_id: str,
    business_id: str,
    text: str,
    analysis: ReviewAnalysis,
    *,
    model: str | None = None,
    host: str | None = None,
    max_retries: int = 2,
) -> DraftResponse:
    """Generate a draft reply to a negative, non-severe review via a local Ollama model.

    Args:
        review_id: Identifier to embed in the returned draft (not sent to
            the model — see ``DraftText``).
        business_id: Identifier to embed in the returned draft.
        text: Raw review text the reply should respond to.
        analysis: The review's classification. Only the negative aspects
            are passed to the model, as a steering hint grounded in
            already-extracted structure — the model still reads ``text``
            itself rather than relying solely on this summary.
        model: Ollama model tag. Defaults to $OLLAMA_MODEL, then the
            same default as ``classifier.classify_review``.
        host: Ollama server URL. Same defaulting as ``classify_review``.
        max_retries: Extra attempts if the model's response doesn't
            validate against ``DraftText``.

    Returns:
        A validated ``DraftResponse`` with ``review_id``/``business_id``
        set to the values passed in. Never sent or published by this
        function or anything it calls.

    Raises:
        ClassificationError: reused from ``classifier`` — same failure
            modes (server unreachable, or no schema-valid output within
            ``max_retries``), same exception type since callers already
            handle it for classification.
    """
    resolved_model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    resolved_host = host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    client = ollama.Client(host=resolved_host, timeout=_REQUEST_TIMEOUT)
    response_schema = DraftText.model_json_schema()

    negative_aspects = [a.aspect.value for a in analysis.aspects if a.sentiment == Sentiment.NEGATIVE]
    user_content = (
        f"Review:\n{text}\n\nAspects the review criticized: {', '.join(negative_aspects) or 'unspecified'}"
    )
    messages = [
        {"role": "system", "content": _DRAFT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    start = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat(
                model=resolved_model,
                messages=messages,
                format=response_schema,
                options={"temperature": 0},
            )
        except Exception as exc:  # Ollama unreachable, model not pulled, etc.
            wrapped = ClassificationError(
                f"could not get a response from Ollama at {resolved_host!r} with model {resolved_model!r}: {exc}"
            )
            _emit_trace(
                "draft_generation",
                review_id,
                resolved_model,
                attempts=attempt + 1,
                elapsed_ms=(time.monotonic() - start) * 1000,
                outcome="connection_error",
                error=str(wrapped),
            )
            raise wrapped from exc

        try:
            parsed = DraftText.model_validate_json(response["message"]["content"])
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "review_id=%s: draft attempt %d/%d returned schema-invalid output: %s",
                review_id,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            continue

        _emit_trace(
            "draft_generation",
            review_id,
            resolved_model,
            attempts=attempt + 1,
            elapsed_ms=(time.monotonic() - start) * 1000,
            outcome="success",
        )
        return DraftResponse(review_id=review_id, business_id=business_id, **parsed.model_dump())

    _emit_trace(
        "draft_generation",
        review_id,
        resolved_model,
        attempts=max_retries + 1,
        elapsed_ms=(time.monotonic() - start) * 1000,
        outcome="schema_invalid",
        error=str(last_error),
    )
    raise ClassificationError(
        f"model {resolved_model!r} did not return a schema-valid draft after "
        f"{max_retries + 1} attempt(s): {last_error}"
    )


def flag_for_escalation(analysis: ReviewAnalysis) -> EscalationFlag:
    """Build an escalation record for a review routed to ``Route.ESCALATE``.

    Deterministic, no LLM call: severity is already known from
    classification (``analysis.severity_flag``), so this only needs to
    format a human-readable reason from fields that already exist.

    Args:
        analysis: A classified review with ``severity_flag`` set. Not
            re-checked here — call this only for reviews
            ``routing.decide_route`` actually routed to ESCALATE.

    Returns:
        An ``EscalationFlag`` summarizing why this review needs a human.
    """
    negative_aspects = [a.aspect.value for a in analysis.aspects if a.sentiment == Sentiment.NEGATIVE]
    if negative_aspects:
        reason = f"Flagged as severe; negative aspects: {', '.join(negative_aspects)}"
    else:
        reason = "Flagged as severe; no specific aspect identified"
    return EscalationFlag(review_id=analysis.review_id, business_id=analysis.business_id, reason=reason)
