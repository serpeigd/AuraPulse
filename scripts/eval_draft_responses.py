"""Offline eval: structural/policy checks on generated draft replies.

Not a quality eval — there's no free or cheap ground truth for "is this a
good reply" (unlike sentiment, which has the star-rating proxy). This
checks the things that are cheap and objective to verify instead: does
the draft honor the non-negotiable constraints from the system prompt
(no promised remedies, no false "already fixed" claims, reasonable
length, mentions the actual complaint)? A judged-quality eval (e.g. an
LLM-as-judge rubric) is a real gap this doesn't close — see
docs/DESIGN.md.

Runs against the deterministic fake-review dataset's NEGATIVE reviews
(same ground truth generator used by eval_fake_reviews.py) rather than
real Yelp text, so this needs no hand-labeling and stays free to re-run.

Usage:
    python scripts/eval_draft_responses.py
"""

from __future__ import annotations

import sys

# Windows consoles default to a non-UTF-8 codepage; draft text can contain
# characters like em dashes that would otherwise print as "?" / mojibake
# and make failures look like data corruption when it's only a display
# issue. Reconfigure rather than silently losing/replacing characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aurapulse.classifier import ClassificationError
from aurapulse.fake_reviews import generate_fixed_dataset
from aurapulse.response_draft import generate_draft_response
from aurapulse.schemas import Sentiment

_MAX_WORDS = 130  # some slack over the prompt's "under 100 words" instruction
_FORBIDDEN_PROMISE_PHRASES = ("refund", "discount", "free ", "compensat", "replace", "voucher", "coupon")
_FALSE_FIX_CLAIM_PHRASES = ("we've fixed", "we have fixed", "we already fixed", "we've resolved", "we have resolved")


def _check_draft(draft_text: str, negative_aspects: list[str]) -> dict[str, bool]:
    """Run every structural check against one draft. Returns a name -> passed dict."""
    lowered = draft_text.lower()
    return {
        "under_word_limit": len(draft_text.split()) <= _MAX_WORDS,
        "no_promised_remedy": not any(phrase in lowered for phrase in _FORBIDDEN_PROMISE_PHRASES),
        "no_false_fix_claim": not any(phrase in lowered for phrase in _FALSE_FIX_CLAIM_PHRASES),
        "mentions_a_complained_aspect": (
            any(aspect.replace("_", " ") in lowered for aspect in negative_aspects) if negative_aspects else True
        ),
    }


def main() -> int:
    """Run the eval and print a report. Returns a process exit code."""
    negative_reviews = [r for r in generate_fixed_dataset() if r.expected.overall_sentiment == Sentiment.NEGATIVE]
    total = len(negative_reviews)

    check_pass_counts: dict[str, int] = {}
    failures: list[str] = []
    evaluated = 0

    for review in negative_reviews:
        negative_aspects = [
            m.aspect.value for m in review.expected.aspects if m.sentiment == Sentiment.NEGATIVE
        ]
        try:
            draft = generate_draft_response(
                review.expected.review_id, review.expected.business_id, review.text, review.expected
            )
        except ClassificationError as exc:
            failures.append(f"{review.expected.review_id}: {exc}")
            continue

        evaluated += 1
        results = _check_draft(draft.draft_text, negative_aspects)
        print(f"[{review.expected.review_id}] {results} -- {draft.draft_text!r}")
        for check_name, passed in results.items():
            check_pass_counts[check_name] = check_pass_counts.get(check_name, 0) + int(passed)

    print(f"\nDrafts evaluated: {evaluated}/{total} ({len(failures)} generation failures)")
    for failure in failures:
        print(f"  FAILED: {failure}")

    if evaluated == 0:
        print("No drafts were generated - nothing to score.")
        return 1

    print("\nStructural checks (not a quality judgment -- see docstring):")
    for check_name, passed_count in check_pass_counts.items():
        print(f"  {check_name}: {passed_count}/{evaluated} ({passed_count / evaluated:.1%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
