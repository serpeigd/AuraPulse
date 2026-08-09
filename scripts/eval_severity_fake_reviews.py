"""Offline eval: severity_flag accuracy against a balanced deterministic dataset.

Split out from eval_fake_reviews.py on purpose: generate_fixed_dataset()
has only one severity=True case among eight reviews -- nowhere near
enough signal to tell a real reliability problem from noise (see
docs/DESIGN.md's original "Known gap" entry, based on exactly that one-
case denominator). generate_severity_dataset() has a real denominator on
both classes, and separates TRUE cases (genuine safety/health/legal
issues) from FALSE "near miss" cases (ordinary complaints in emotionally
intense language) so a precision/recall split can show *which* direction
the model is wrong in, not just an aggregate accuracy number.

Usage:
    python scripts/eval_severity_fake_reviews.py
"""

from __future__ import annotations

import sys

from aurapulse.classifier import ClassificationError, classify_review
from aurapulse.fake_reviews import generate_severity_dataset


def main() -> int:
    """Run the eval and print a report. Returns a process exit code."""
    dataset = generate_severity_dataset()
    total = len(dataset)

    tp = fp = tn = fn = 0
    failures: list[str] = []
    false_negative_ids: list[str] = []
    false_positive_ids: list[str] = []

    for case in dataset:
        try:
            result = classify_review(case.review_id, "eval-business", case.text)
        except ClassificationError as exc:
            failures.append(f"{case.review_id}: {exc}")
            continue

        predicted = result.severity_flag
        if predicted and case.severity_flag:
            tp += 1
        elif predicted and not case.severity_flag:
            fp += 1
            false_positive_ids.append(case.review_id)
        elif not predicted and case.severity_flag:
            fn += 1
            false_negative_ids.append(case.review_id)
        else:
            tn += 1

    evaluated = total - len(failures)
    print(f"Cases evaluated: {evaluated}/{total} ({len(failures)} classification failures)")
    for failure in failures:
        print(f"  FAILED: {failure}")

    if evaluated == 0:
        print("No cases were successfully classified - nothing to score.")
        return 1

    accuracy = (tp + tn) / evaluated
    print(f"\nOverall accuracy: {tp + tn}/{evaluated} ({accuracy:.1%})")

    true_support = tp + fn
    false_support = tn + fp
    print(f"\nOn the {true_support} genuinely severe cases (recall):")
    print(f"  Correctly flagged: {tp}/{true_support} ({tp / true_support:.1%})" if true_support else "  n/a")
    if false_negative_ids:
        print(f"  Missed (false negative): {false_negative_ids}")

    print(f"\nOn the {false_support} ordinary-but-emotionally-intense cases (specificity):")
    print(f"  Correctly NOT flagged: {tn}/{false_support} ({tn / false_support:.1%})" if false_support else "  n/a")
    if false_positive_ids:
        print(f"  Over-triggered (false positive): {false_positive_ids}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
