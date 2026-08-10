"""LLM-as-judge quality assessment for generated draft replies.

``scripts/eval_draft_responses.py`` already checks structural/policy
constraints (word limit, no promised remedy, no false fix claim) but says
nothing about whether a draft is actually *good* -- relevant, well-toned,
usable with minimal editing. This module is that missing quality signal.

**Self-evaluation bias, stated plainly.** The only model available under
this project's zero-cost constraint is the same local `llama3.1:8b` that
wrote the drafts in the first place -- there's no free external judge.
Using a model to grade its own output is a known-biased setup in the LLM
eval literature (a model tends to be lenient toward its own phrasing and
blind to its own systematic weaknesses). Two things are done about this,
neither of which eliminates the bias, both of which make it visible:
    1. The rubric asks specific, checkable yes/no questions instead of
       one holistic "how good is this" score -- vague holistic scores
       are more exploitable by this exact bias.
    2. The judge's verdicts are validated against a human's independent
       verdicts on the same drafts before being trusted for ongoing
       regression use -- see docs/DESIGN.md for the agreement-rate
       result. If a future prompt change drops that agreement rate, the
       judge itself needs re-validating, not just the drafts.

**Four questions are asked, only three are trusted.**
``addresses_specific_complaint`` failed human validation (0/14 agreement,
4/14 after a revision informed by the human's specific disagreements) and
is NOT used for regression decisions -- see ``DraftQualityVerdict``'s
docstring. It stays in the prompt anyway: dropping the question was tried
first, and measurably broke ``appropriate_tone`` on the same drafts (14/14
True -> 0/14 True, confirmed by isolating the change) even though that
question's wording never changed. This model doesn't judge these criteria
independently, so the validated prompt shape is kept exactly as it was
when validated -- only the untrusted field's value is ignored downstream,
never the prompt structure that produced it. See docs/DESIGN.md.
"""

from __future__ import annotations

import logging
import os

import ollama
from pydantic import ValidationError

from aurapulse.classifier import (
    _REQUEST_TIMEOUT,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    ClassificationError,
)
from aurapulse.schemas import DraftQualityVerdict

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """You are reviewing a draft reply a restaurant is considering sending to a \
customer's negative review. You did NOT write the draft -- you are checking it before a human \
decides whether to send it.

Given the original customer review and the draft reply, answer four yes/no questions honestly and \
critically. Do not default to "yes" -- a mediocre draft should score "no" on the criteria it \
actually fails:

- addresses_specific_complaint: does the reply mention or clearly respond to what THIS review
  specifically complained about, not just a generic acknowledgment?
- appropriate_tone: is the tone empathetic and professional, without being defensive, dismissive,
  or so over-apologetic that it sounds insincere?
- usable_with_minor_edits: could a business owner send this with only small edits (a name, a
  detail), or would they need to substantially rewrite it?
- not_generic: does this read like it was written for this specific complaint, rather than a
  boilerplate reply that could apply to almost any negative review?

Then give a short (1-2 sentence) reasoning explaining your answers.

Respond with nothing but the structured JSON described by the schema.
"""


def judge_draft(
    review_text: str,
    draft_text: str,
    *,
    model: str | None = None,
    host: str | None = None,
    max_retries: int = 2,
) -> DraftQualityVerdict:
    """Judge one draft reply against the original review it responds to.

    Args:
        review_text: The original customer review.
        draft_text: The generated draft reply to assess.
        model: Ollama model tag. Defaults to $OLLAMA_MODEL, then the
            same default as ``classifier.classify_review``.
        host: Ollama server URL. Same defaulting as ``classify_review``.
        max_retries: Extra attempts if the model's response doesn't
            validate against ``DraftQualityVerdict``.

    Returns:
        A validated ``DraftQualityVerdict``.

    Raises:
        ClassificationError: reused from ``classifier`` -- same failure
            modes (server unreachable, or no schema-valid output within
            ``max_retries``).
    """
    resolved_model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    resolved_host = host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    client = ollama.Client(host=resolved_host, timeout=_REQUEST_TIMEOUT)
    response_schema = DraftQualityVerdict.model_json_schema()

    user_content = f"Original review:\n{review_text}\n\nDraft reply:\n{draft_text}"
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

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
            raise ClassificationError(
                f"could not get a response from Ollama at {resolved_host!r} with model {resolved_model!r}: {exc}"
            ) from exc

        try:
            return DraftQualityVerdict.model_validate_json(response["message"]["content"])
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "judge attempt %d/%d returned schema-invalid output: %s", attempt + 1, max_retries + 1, exc
            )

    raise ClassificationError(
        f"model {resolved_model!r} did not return a schema-valid verdict after "
        f"{max_retries + 1} attempt(s): {last_error}"
    )
