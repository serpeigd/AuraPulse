"""End-to-end Hito 1 pipeline: route already-classified reviews to aggregation,
a draft reply, or escalation, then print the business report.

This is the first script that actually exercises
``orchestrator.process_reviews()`` outside of tests. ``generate_report.py
--demo`` only covers Hito 0 (classify + aggregate) -- it never routes, drafts,
or escalates, so there was previously no way to see the full Hito 1 flow run
against anything without writing a throwaway script.

``--demo`` reuses the deterministic fake-review dataset's known-correct
``ReviewAnalysis`` (same convention as ``generate_report.py --demo``), so no
Yelp dataset or classification step is needed -- but draft generation still
calls the local Ollama model, so a running server is required even in demo
mode. Escalation needs no LLM at all.

Usage:
    python scripts/run_pipeline.py --demo
    python scripts/run_pipeline.py --input data/processed/classified_reviews.jsonl \\
        --reviews data/processed/review_subset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aurapulse.aggregation import summarize_other_aspect_usage
from aurapulse.escalation_delivery import DEFAULT_ESCALATIONS_PATH, write_escalations
from aurapulse.orchestrator import Hito1Batch, process_reviews
from aurapulse.pipeline_io import (
    DEFAULT_INPUT_PATH,
    DEFAULT_REVIEWS_PATH,
    load_demo_data,
    load_real_data,
)
from aurapulse.reporting import format_full_report

# Windows consoles default to a non-UTF-8 codepage; draft/reasoning text can
# contain characters like em dashes that would otherwise print as mojibake.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _print_batch_summary(batch: Hito1Batch) -> None:
    print("\n=== Routing summary ===")
    print(f"Drafted: {len(batch.drafts)}  |  Escalated: {len(batch.escalations)}")
    if batch.draft_failures:
        print(f"Draft generation FAILED for {len(batch.draft_failures)} review(s): {batch.draft_failures}")
        print("(Ollama may be unreachable -- try: ollama serve)")

    if batch.drafts:
        print("\n--- Draft replies (human review required before sending, per CLAUDE.md) ---")
        for draft in batch.drafts:
            print(f"\n[{draft.review_id}] {draft.draft_text}")

    if batch.escalations:
        print("\n--- Escalations ---")
        for escalation in batch.escalations:
            print(f"[{escalation.review_id}] {escalation.reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run against the deterministic fake-review dataset instead of real classified data.",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_PATH, help=f"Classified reviews JSONL (default: {DEFAULT_INPUT_PATH})"
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=DEFAULT_REVIEWS_PATH,
        help=f"Raw review text CSV, for drafting (default: {DEFAULT_REVIEWS_PATH})",
    )
    parser.add_argument(
        "--escalations-path",
        type=Path,
        default=DEFAULT_ESCALATIONS_PATH,
        help=f"Where to append escalation records (default: {DEFAULT_ESCALATIONS_PATH})",
    )
    args = parser.parse_args()

    if args.demo:
        analyses, review_texts = load_demo_data()
        print(f"[DEMO MODE] Using {len(analyses)} deterministic fake reviews, not real data.")
    else:
        try:
            analyses, review_texts = load_real_data(args.input, args.reviews)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1

    if not analyses:
        print("No classified reviews to process.")
        return 1

    # No try/except ClassificationError here: process_reviews() already catches
    # draft-generation failures per-review and accumulates them in
    # batch.draft_failures rather than raising -- see _print_batch_summary,
    # which surfaces them (e.g. if Ollama is unreachable, every DRAFT_RESPONSE
    # review ends up there instead of the whole run failing).
    batch = process_reviews(analyses, review_texts)

    other_summary = summarize_other_aspect_usage(analyses)
    print(format_full_report(batch.business_reports, other_summary))
    _print_batch_summary(batch)

    if batch.escalations:
        write_escalations(batch.escalations, path=args.escalations_path)
        print(f"\nWrote {len(batch.escalations)} escalation(s) to {args.escalations_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
