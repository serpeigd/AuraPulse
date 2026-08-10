"""Generate draft replies for a diverse set of negative reviews and judge each
with the local LLM against the 3 criteria validated against human judgment
(see docs/DESIGN.md) -- this is the trusted, repeatable draft-quality
regression check for future changes to ``response_draft.py``'s prompt.

Candidate reviews (14 total, deliberately mixing two existing fixture sources
rather than writing new ones):
    - The 6 NEGATIVE reviews in ``fake_reviews.generate_fixed_dataset()``
      (mixed-aspect complaints, the "core" draft-response candidates).
    - The 8 "near miss" cases in ``fake_reviews.generate_severity_dataset()``
      (severity_flag=False) -- ordinary complaints in emotionally intense
      language, covering a different style/tone range than the fixed dataset.

Reports aggregate pass rates with denominators for ``appropriate_tone``,
``usable_with_minor_edits``, and ``not_generic`` only. The judge is also
asked ``addresses_specific_complaint`` (and it's written to the CSV for
visibility) but that field is NOT included in the aggregate report --
it failed human validation (0/14, then 4/14 after a revision) and,
critically, removing the question from the prompt entirely was tried and
found to destabilize the other three criteria on the same drafts. See
``DraftQualityVerdict``'s docstring and docs/DESIGN.md for the full
story. The CSV is for spot-checking; the aggregate numbers are what to
trust, and this is not a re-run of the human-validation pass.

Usage:
    python scripts/eval_draft_quality.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Windows consoles default to a non-UTF-8 codepage; draft text can contain
# characters like em dashes that would otherwise print as "?" / mojibake
# and make failures look like data corruption when it's only a display
# issue. Reconfigure rather than silently losing/replacing characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aurapulse.classifier import ClassificationError
from aurapulse.draft_judge import judge_draft
from aurapulse.fake_reviews import generate_fixed_dataset, generate_severity_dataset
from aurapulse.response_draft import generate_draft_response
from aurapulse.schemas import ReviewAnalysis, Sentiment

OUTPUT_PATH = Path("data/processed/draft_quality_eval.csv")

FIELDNAMES = [
    "review_id",
    "review_text",
    "draft_text",
    "judge_addresses_specific_complaint",  # NOT validated -- see module docstring, excluded below
    "judge_appropriate_tone",
    "judge_usable_with_minor_edits",
    "judge_not_generic",
    "judge_reasoning",
]

# Validated against an independent human's verdicts on the same 14 drafts --
# see docs/DESIGN.md. Keep this list in sync with DraftQualityVerdict's
# fields if criteria are ever added or removed.
VALIDATED_CRITERIA = ["judge_appropriate_tone", "judge_usable_with_minor_edits", "judge_not_generic"]


def _candidate_reviews() -> list[tuple[str, str, ReviewAnalysis]]:
    """Return (review_id, review_text, analysis) triples to draft and judge."""
    candidates = []
    for fr in generate_fixed_dataset():
        if fr.expected.overall_sentiment == Sentiment.NEGATIVE:
            candidates.append((fr.expected.review_id, fr.text, fr.expected))
    for case in generate_severity_dataset():
        if not case.severity_flag:
            analysis = ReviewAnalysis(
                review_id=case.review_id,
                business_id="eval-business",
                overall_sentiment=Sentiment.NEGATIVE,
                aspects=[],
                severity_flag=False,
            )
            candidates.append((case.review_id, case.text, analysis))
    return candidates


def main() -> int:
    """Generate drafts, judge them, print an aggregate report, and write the CSV."""
    candidates = _candidate_reviews()
    total = len(candidates)

    rows = []
    failures: list[str] = []
    for i, (review_id, text, analysis) in enumerate(candidates, start=1):
        try:
            draft = generate_draft_response(review_id, analysis.business_id, text, analysis)
        except ClassificationError as exc:
            failures.append(f"{review_id} (draft generation): {exc}")
            print(f"[{i}/{total}] {review_id} FAILED to generate draft: {exc}")
            continue

        try:
            verdict = judge_draft(text, draft.draft_text)
        except ClassificationError as exc:
            failures.append(f"{review_id} (judging): {exc}")
            print(f"[{i}/{total}] {review_id} FAILED to judge draft: {exc}")
            continue

        print(f"[{i}/{total}] {review_id}: judge verdict = {verdict.model_dump(exclude={'reasoning'})}")
        rows.append(
            {
                "review_id": review_id,
                "review_text": text,
                "draft_text": draft.draft_text,
                "judge_addresses_specific_complaint": verdict.addresses_specific_complaint,
                "judge_appropriate_tone": verdict.appropriate_tone,
                "judge_usable_with_minor_edits": verdict.usable_with_minor_edits,
                "judge_not_generic": verdict.not_generic,
                "judge_reasoning": verdict.reasoning,
            }
        )

    evaluated = len(rows)
    print(f"\nGenerated + judged: {evaluated}/{total} ({len(failures)} failures)")
    for failure in failures:
        print(f"  FAILED: {failure}")

    if not rows:
        print("Nothing was generated - nothing to score.")
        return 1

    print("\nValidated criteria (see docs/DESIGN.md for the human-agreement check):")
    for criterion in VALIDATED_CRITERIA:
        passed = sum(1 for r in rows if r[criterion])
        print(f"  {criterion}: {passed}/{evaluated} ({passed / evaluated:.1%})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {evaluated} rows to {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
