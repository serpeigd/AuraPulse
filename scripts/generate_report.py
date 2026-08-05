"""Render the aggregated reputation/inconsistency report (Hito 0 step 9).

Reads already-classified reviews from a JSONL file of `ReviewAnalysis`
records (one per line) and prints the aggregated report. This script
never calls the classifier itself -- classification and reporting are
separate concerns.

`--demo` runs against the project's own deterministic fake-review
dataset instead, so the report format is demonstrable today even
though real classified data isn't ready yet (still blocked on hand-
labeling the aspect sample -- see docs/DESIGN.md).

Usage:
    python scripts/generate_report.py --demo
    python scripts/generate_report.py --input data/processed/classified_reviews.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aurapulse.aggregation import aggregate_reviews, summarize_other_aspect_usage
from aurapulse.fake_reviews import generate_fixed_dataset
from aurapulse.reporting import format_full_report
from aurapulse.schemas import ReviewAnalysis

DEFAULT_INPUT_PATH = Path("data/processed/classified_reviews.jsonl")


def load_classified_reviews(path: Path) -> list[ReviewAnalysis]:
    """Load one ReviewAnalysis per line from a JSONL file.

    Raises:
        FileNotFoundError: if `path` doesn't exist, with a message
            pointing at what step produces it.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This file is produced by classifying the review subset "
            "(Hito 0 step 7) -- not built yet. Try `python scripts/generate_report.py --demo` "
            "to see the report format against the deterministic fake-review dataset instead."
        )
    with path.open(encoding="utf-8") as f:
        return [ReviewAnalysis.model_validate_json(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run against the deterministic fake-review dataset instead of real classified data.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"JSONL file of classified ReviewAnalysis records (default: {DEFAULT_INPUT_PATH})",
    )
    args = parser.parse_args()

    if args.demo:
        analyses = [review.expected for review in generate_fixed_dataset()]
        print(f"[DEMO MODE] Using {len(analyses)} deterministic fake reviews, not real data.\n")
    else:
        try:
            analyses = load_classified_reviews(args.input)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1

    if not analyses:
        print("No classified reviews to report on.")
        return 1

    business_reports = aggregate_reviews(analyses)
    other_summary = summarize_other_aspect_usage(analyses)
    print(format_full_report(business_reports, other_summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
