"""Validate classifier sentiment against the free star-rating proxy.

Yelp's star rating is a free, ready-made proxy for ground-truth
sentiment (1-2 stars -> negative, 3 -> neutral, 4-5 -> positive) --
see CLAUDE.md. This is a first quality signal on real Yelp text
before any hand-labeling, deliberately run on a small stratified
sample rather than the full subset: at ~20-45s/review on local CPU
inference, the full 723-review subset would take hours, and that
full run belongs to the final classification pass (Hito 0 step 7),
not this sanity check.

Usage:
    python scripts/validate_sentiment_proxy.py [--per-star N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import cast

import pandas as pd

from aurapulse.classifier import ClassificationError, classify_review
from aurapulse.schemas import Sentiment

REVIEW_SUBSET_PATH = Path("data/processed/review_subset.csv")
OUTPUT_PATH = Path("data/processed/sentiment_proxy_sample.csv")


def stars_to_proxy_sentiment(stars: float) -> Sentiment:
    """Map a Yelp star rating to the proxy sentiment bucket from CLAUDE.md."""
    if stars <= 2:
        return Sentiment.NEGATIVE
    if stars == 3:
        return Sentiment.NEUTRAL
    return Sentiment.POSITIVE


def select_stratified_sample(reviews: pd.DataFrame, per_star: int) -> pd.DataFrame:
    """Deterministically pick up to ``per_star`` reviews per star value.

    Sorted by review_id within each star bucket (no randomness), so
    the same subset always yields the same sample. If a bucket has
    fewer than ``per_star`` reviews, every review in it is kept.
    """
    parts = [
        group.sort_values("review_id").head(per_star)
        for _, group in reviews.groupby("stars", sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-star", type=int, default=12, help="Max reviews to sample per star value (default: 12)"
    )
    args = parser.parse_args()

    if not REVIEW_SUBSET_PATH.exists():
        print(f"ERROR: {REVIEW_SUBSET_PATH} not found. Run scripts/build_subset.py first.")
        return 1

    reviews = pd.read_csv(REVIEW_SUBSET_PATH)
    sample = select_stratified_sample(reviews, args.per_star)
    print(f"Sampled {len(sample)} reviews ({args.per_star} per star value, stars 1-5)")

    results = []
    failures: list[str] = []
    start = time.time()
    for i, row in enumerate(sample.itertuples(index=False), start=1):
        # itertuples() namedtuple fields aren't precisely typed by pandas-stubs
        # (values are dynamically read from the DataFrame), so this cast just
        # documents what build_subset.py's CSV actually guarantees: a float.
        proxy_sentiment = stars_to_proxy_sentiment(cast(float, row.stars))
        try:
            analysis = classify_review(str(row.review_id), str(row.business_id), str(row.text))
        except ClassificationError as exc:
            failures.append(f"{row.review_id}: {exc}")
            print(f"[{i}/{len(sample)}] FAILED {row.review_id}: {exc}")
            continue

        match = analysis.overall_sentiment == proxy_sentiment
        results.append(
            {
                "review_id": row.review_id,
                "business_id": row.business_id,
                "stars": row.stars,
                "proxy_sentiment": proxy_sentiment.value,
                "model_sentiment": analysis.overall_sentiment.value,
                "match": match,
            }
        )
        elapsed = time.time() - start
        avg = elapsed / i
        print(
            f"[{i}/{len(sample)}] stars={row.stars} proxy={proxy_sentiment.value:8s} "
            f"model={analysis.overall_sentiment.value:8s} {'OK' if match else 'MISMATCH'} "
            f"(avg {avg:.1f}s/review, ~{avg * (len(sample) - i) / 60:.1f} min left)"
        )

    if not results:
        print("No reviews were successfully classified - nothing to score.")
        return 1

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    n = len(results_df)
    n_match = int(results_df["match"].sum())
    print(f"\nEvaluated {n}/{len(sample)} ({len(failures)} classification failures)")
    for f in failures:
        print(f"  FAILED: {f}")
    print(f"Overall agreement with star-rating proxy: {n_match}/{n} ({n_match / n:.1%})")

    print("\nBy proxy sentiment bucket:")
    for sentiment_value, group in results_df.groupby("proxy_sentiment"):
        bucket_match = int(group["match"].sum())
        print(f"  {sentiment_value:8s}: {bucket_match}/{len(group)} ({bucket_match / len(group):.1%})")

    print("\nBy star rating:")
    for stars, group in results_df.groupby("stars"):
        bucket_match = int(group["match"].sum())
        print(f"  {stars} stars: {bucket_match}/{len(group)} ({bucket_match / len(group):.1%})")

    print(f"\nPer-review detail written to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
