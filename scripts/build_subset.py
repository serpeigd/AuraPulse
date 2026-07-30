"""Build the Hito 0 restaurant review subset from the raw Yelp dataset.

Filters business.json for restaurants, deterministically selects
~15-20 of them within a bounded review-count range, pulls every
review for those businesses from review.json (one streaming pass —
the file is several GB), and writes both subsets to data/processed/
as CSV for reuse by later steps.

Usage:
    python scripts/build_subset.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from aurapulse.data_loader import (
    DatasetNotFoundError,
    load_restaurant_businesses,
    load_reviews_for_businesses,
    select_business_subset,
)

PROCESSED_DIR = Path("data/processed")


def main() -> int:
    try:
        start = time.time()
        print("Scanning business.json for restaurants...")
        restaurants = load_restaurant_businesses()
        print(f"  found {len(restaurants)} restaurant businesses in {time.time() - start:.1f}s")

        selected = select_business_subset(restaurants)
        expected_reviews = sum(b["review_count"] for b in selected)
        print(
            f"Selected {len(selected)} businesses, {expected_reviews} reviews expected "
            "(per business.json's review_count field)"
        )

        business_ids = {b["business_id"] for b in selected}
        print("Scanning review.json for matching reviews (several GB, this can take a few minutes)...")
        scan_start = time.time()
        reviews = load_reviews_for_businesses(business_ids)
        print(f"  found {len(reviews)} matching reviews in {time.time() - scan_start:.1f}s")
    except DatasetNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    business_df = pd.DataFrame(selected)
    review_df = pd.DataFrame(reviews)
    business_df.to_csv(PROCESSED_DIR / "business_subset.csv", index=False)
    review_df.to_csv(PROCESSED_DIR / "review_subset.csv", index=False)

    print(f"\nWrote {len(business_df)} businesses -> {PROCESSED_DIR / 'business_subset.csv'}")
    print(f"Wrote {len(review_df)} reviews -> {PROCESSED_DIR / 'review_subset.csv'}")

    if len(review_df) and "business_id" in review_df:
        per_business = review_df.groupby("business_id").size()
        print(
            f"Reviews per business: min={per_business.min()}, max={per_business.max()}, "
            f"median={per_business.median():.0f}"
        )

    if not (500 <= len(review_df) <= 1000):
        print(
            f"\nNOTE: total reviews ({len(review_df)}) is outside the 500-1000 target range "
            "from CLAUDE.md - selection bounds in select_business_subset() may need tuning."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
