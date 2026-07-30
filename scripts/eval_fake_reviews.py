"""Offline eval: run the Ollama-backed classifier against the deterministic
fake-review ground truth (see ``aurapulse.fake_reviews``) and report
per-field accuracy with denominators.

This is a script, not a pytest test, on purpose: it requires a running
local Ollama server, and Hito 0 shouldn't let "the model isn't running"
hide behind a green test suite. Run it explicitly, after ``ollama serve``
and ``ollama pull <model>``.

Usage:
    python scripts/eval_fake_reviews.py
"""

from __future__ import annotations

import sys

from aurapulse.classifier import ClassificationError, classify_review
from aurapulse.fake_reviews import generate_fixed_dataset


def main() -> int:
    """Run the eval and print a report. Returns a process exit code."""
    dataset = generate_fixed_dataset()
    total = len(dataset)

    sentiment_correct = 0
    severity_correct = 0
    total_expected_mentions = 0
    total_predicted_mentions = 0
    total_correct_mentions = 0
    failures: list[str] = []

    for i, review in enumerate(dataset):
        try:
            result = classify_review(review.expected.review_id, review.expected.business_id, review.text)
        except ClassificationError as exc:
            if i == 0:
                print(f"Could not reach the local model at all: {exc}")
                print("Is Ollama running? Try: ollama serve   (and: ollama pull llama3.1:8b)")
                return 1
            failures.append(f"{review.expected.review_id}: {exc}")
            continue

        if result.overall_sentiment == review.expected.overall_sentiment:
            sentiment_correct += 1
        if result.severity_flag == review.expected.severity_flag:
            severity_correct += 1

        expected_pairs = {(m.aspect, m.sentiment) for m in review.expected.aspects}
        actual_pairs = {(m.aspect, m.sentiment) for m in result.aspects}
        total_expected_mentions += len(expected_pairs)
        total_predicted_mentions += len(actual_pairs)
        total_correct_mentions += len(expected_pairs & actual_pairs)

    evaluated = total - len(failures)
    print(f"Reviews evaluated: {evaluated}/{total} ({len(failures)} classification failures)")
    for failure in failures:
        print(f"  FAILED: {failure}")

    if evaluated == 0:
        print("No reviews were successfully classified — nothing to score.")
        return 1

    print(f"Overall sentiment accuracy: {sentiment_correct}/{evaluated} ({sentiment_correct / evaluated:.1%})")
    print(f"Severity flag accuracy: {severity_correct}/{evaluated} ({severity_correct / evaluated:.1%})")

    if total_expected_mentions:
        recall = total_correct_mentions / total_expected_mentions
        print(f"Aspect mention recall: {total_correct_mentions}/{total_expected_mentions} ({recall:.1%})")
    if total_predicted_mentions:
        precision = total_correct_mentions / total_predicted_mentions
        print(f"Aspect mention precision: {total_correct_mentions}/{total_predicted_mentions} ({precision:.1%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
