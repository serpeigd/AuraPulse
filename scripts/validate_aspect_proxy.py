"""Validate classifier aspect extraction against hand-labeled ground truth.

`aspect` has no free proxy (unlike sentiment, which has the star rating
-- see scripts/validate_sentiment_proxy.py). This compares the
classifier's aspect + per-aspect sentiment output against a 100-review
sample hand-labeled via scripts/build_labeling_sheet.py, on the exact
same deterministic stratified sample so review_id sets line up by
construction.

At ~20-45s/review on local CPU inference, a 100-review run takes
roughly 35-75 minutes.

Usage:
    python scripts/validate_aspect_proxy.py [--labels PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from aurapulse.classifier import ClassificationError, classify_review
from aurapulse.schemas import Aspect, Sentiment

REVIEW_SUBSET_PATH = Path("data/processed/review_subset.csv")
# Prefer a "_corrected" export (produced when a labeler fixes malformed
# rows from a prior save) over the plain filename -- see docs/DESIGN.md.
DEFAULT_LABELS_CANDIDATES = [
    Path("data/processed/aspect_labeling_completed_corrected.csv"),
    Path("data/processed/aspect_labeling_completed.csv"),
]
OUTPUT_PATH = Path("data/processed/aspect_proxy_sample.csv")

ASPECT_COLUMNS = [a.value for a in Aspect]


def _resolve_labels_path(explicit: Path | None) -> Path:
    """Pick the ground-truth CSV to validate against.

    An explicit ``--labels`` path always wins. Otherwise tries each of
    ``DEFAULT_LABELS_CANDIDATES`` in order and prints which one was
    picked -- the choice is never silent.

    Raises:
        FileNotFoundError: if no candidate exists.
    """
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"{explicit} not found")
        return explicit
    for candidate in DEFAULT_LABELS_CANDIDATES:
        if candidate.exists():
            print(f"Using ground truth: {candidate}")
            return candidate
    raise FileNotFoundError(
        f"No ground-truth labels found (looked for {[str(c) for c in DEFAULT_LABELS_CANDIDATES]}). "
        "Run scripts/build_labeling_sheet.py and hand-label it first."
    )


def load_ground_truth(path: Path) -> dict[str, set[tuple[str, str]]]:
    """Parse the wide-format hand-labeled CSV into per-review (aspect, sentiment) sets.

    The sheet has one column per ``Aspect`` enum value holding
    'positive'/'negative'/'neutral' when that aspect is mentioned, or
    blank when it isn't. This reshapes it to the same (aspect,
    sentiment) pair shape the classifier itself produces, so the two
    are directly comparable.

    Raises:
        ValueError: if an aspect column is missing entirely, or a cell
            holds anything other than a known ``Sentiment`` value or
            blank -- a labeling typo should fail loudly rather than
            silently read as "aspect not mentioned".
    """
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    missing_cols = set(ASPECT_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing expected aspect columns: {sorted(missing_cols)}")

    valid_sentiments = {s.value for s in Sentiment}
    ground_truth: dict[str, set[tuple[str, str]]] = {}
    # iterrows() over Series (rather than itertuples()._asdict()) --
    # pandas-stubs types itertuples() rows too loosely to call
    # ._asdict() on cleanly; see validate_sentiment_proxy.py for the
    # analogous itertuples cast workaround used elsewhere.
    for _, series_row in df.iterrows():
        review_id = str(series_row["review_id"])
        pairs: set[tuple[str, str]] = set()
        for aspect in ASPECT_COLUMNS:
            value = str(series_row[aspect]).strip()
            if not value:
                continue
            if value not in valid_sentiments:
                raise ValueError(
                    f"{path}: review {review_id!r}, column {aspect!r} has invalid value {value!r} "
                    f"(expected one of {sorted(valid_sentiments)} or blank)"
                )
            pairs.add((aspect, value))
        ground_truth[review_id] = pairs
    return ground_truth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=None, help="Path to hand-labeled ground-truth CSV")
    args = parser.parse_args()

    if not REVIEW_SUBSET_PATH.exists():
        print(f"ERROR: {REVIEW_SUBSET_PATH} not found. Run scripts/build_subset.py first.")
        return 1

    try:
        labels_path = _resolve_labels_path(args.labels)
        ground_truth = load_ground_truth(labels_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    reviews = pd.read_csv(REVIEW_SUBSET_PATH)
    labeled_reviews = reviews[reviews["review_id"].isin(ground_truth)]
    missing_text = set(ground_truth) - set(labeled_reviews["review_id"])
    if missing_text:
        preview = sorted(missing_text)[:5]
        print(
            f"WARNING: {len(missing_text)} labeled review_id(s) not found in "
            f"{REVIEW_SUBSET_PATH}, skipping: {preview}{'...' if len(missing_text) > 5 else ''}"
        )

    total = len(labeled_reviews)
    print(f"Evaluating aspect extraction on {total} hand-labeled reviews (this can take 35-75 min)")

    per_review_rows = []
    per_aspect_counts = {a.value: {"tp": 0, "fp": 0, "fn": 0} for a in Aspect}
    failures: list[str] = []
    start = time.time()
    for i, row in enumerate(labeled_reviews.itertuples(index=False), start=1):
        true_pairs = ground_truth[str(row.review_id)]
        true_aspects = {a for a, _ in true_pairs}
        try:
            analysis = classify_review(str(row.review_id), str(row.business_id), str(row.text))
        except ClassificationError as exc:
            failures.append(f"{row.review_id}: {exc}")
            print(f"[{i}/{total}] FAILED {row.review_id}: {exc}")
            continue

        pred_pairs = {(m.aspect.value, m.sentiment.value) for m in analysis.aspects}
        pred_aspects = {a for a, _ in pred_pairs}

        # Per-aspect TP/FP/FN, ignoring sentiment: did the model say this
        # aspect was mentioned at all, regardless of which way.
        for aspect_value, counts in per_aspect_counts.items():
            in_true = aspect_value in true_aspects
            in_pred = aspect_value in pred_aspects
            if in_true and in_pred:
                counts["tp"] += 1
            elif in_pred and not in_true:
                counts["fp"] += 1
            elif in_true and not in_pred:
                counts["fn"] += 1

        aspect_match = true_aspects == pred_aspects
        pair_match = true_pairs == pred_pairs
        per_review_rows.append(
            {
                "review_id": row.review_id,
                "true_aspects": ",".join(sorted(true_aspects)) or "-",
                "pred_aspects": ",".join(sorted(pred_aspects)) or "-",
                "aspect_set_match": aspect_match,
                "true_pairs": ",".join(sorted(f"{a}:{s}" for a, s in true_pairs)) or "-",
                "pred_pairs": ",".join(sorted(f"{a}:{s}" for a, s in pred_pairs)) or "-",
                "aspect_sentiment_match": pair_match,
            }
        )
        elapsed = time.time() - start
        avg = elapsed / i
        print(
            f"[{i}/{total}] {row.review_id} "
            f"aspects={'OK' if aspect_match else 'DIFF'} pairs={'OK' if pair_match else 'DIFF'} "
            f"(avg {avg:.1f}s/review, ~{avg * (total - i) / 60:.1f} min left)"
        )

    if not per_review_rows:
        print("No reviews were successfully classified - nothing to score.")
        return 1

    results_df = pd.DataFrame(per_review_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    n = len(results_df)
    n_aspect_match = int(results_df["aspect_set_match"].sum())
    n_pair_match = int(results_df["aspect_sentiment_match"].sum())
    print(f"\nEvaluated {n}/{total} ({len(failures)} classification failures)")
    for f in failures:
        print(f"  FAILED: {f}")
    print(f"Aspect-set exact match (ignoring sentiment): {n_aspect_match}/{n} ({n_aspect_match / n:.1%})")
    print(f"Aspect+sentiment exact match:                {n_pair_match}/{n} ({n_pair_match / n:.1%})")

    print("\nPer-aspect precision/recall (presence only, sentiment ignored):")
    for aspect_value, counts in per_aspect_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        support = tp + fn  # how many labeled reviews actually mention this aspect
        precision_str = f"{tp / (tp + fp):.1%}" if (tp + fp) else "n/a (0 predicted)"
        recall_str = f"{tp / (tp + fn):.1%}" if (tp + fn) else "n/a (0 in ground truth)"
        print(
            f"  {aspect_value:12s} support={support:3d}  "
            f"precision={precision_str:>18s} (tp={tp}, fp={fp})  "
            f"recall={recall_str:>18s} (tp={tp}, fn={fn})"
        )

    other_rate = sum(1 for pairs in ground_truth.values() if any(a == "other" for a, _ in pairs)) / len(
        ground_truth
    )
    print(
        f"\n'other' usage in hand-labeled ground truth: {other_rate:.1%} of {len(ground_truth)} reviews -- "
        "high usage would signal the aspect enum needs revisiting (see docs/DESIGN.md)."
    )

    print(f"\nPer-review detail written to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
