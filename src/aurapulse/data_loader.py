"""Loads and subsets the Yelp Open Dataset for restaurant reviews.

Both business and review files are JSON-Lines (one JSON object per
line, not a single JSON array), and the raw files are large
(review.json is several GB) — everything here streams line by line
rather than loading a whole file into memory.

The raw dataset itself is never committed (see .gitignore) and must
be placed somewhere under `data/raw/` manually, per Yelp's terms of
use — see CLAUDE.md.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")
BUSINESS_FILENAME = "yelp_academic_dataset_business.json"
REVIEW_FILENAME = "yelp_academic_dataset_review.json"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a required Yelp dataset file can't be located."""


def find_dataset_file(root: Path, filename: str) -> Path:
    """Recursively locate ``filename`` under ``root``.

    The Yelp dataset is downloaded manually and can end up nested
    under an arbitrary subfolder structure depending on how it was
    unzipped, so this searches rather than assuming a fixed path.

    Raises:
        DatasetNotFoundError: if no file, or more than one file, named
            ``filename`` is found under ``root``.
    """
    matches = list(root.rglob(filename))
    if not matches:
        raise DatasetNotFoundError(
            f"could not find {filename!r} anywhere under {root!r}. Download the Yelp Open "
            f"Dataset from https://www.yelp.com/dataset and place it under {root!r}."
        )
    if len(matches) > 1:
        raise DatasetNotFoundError(
            f"found {len(matches)} files named {filename!r} under {root!r}, expected exactly "
            f"one: {[str(m) for m in matches]}"
        )
    return matches[0]


def iter_json_lines(path: Path) -> Iterator[dict]:
    """Yield each line of a JSON-Lines file as a parsed dict.

    Malformed lines are skipped (and logged) rather than aborting the
    whole read — a handful of corrupt lines in a multi-GB file
    shouldn't take down the whole load.
    """
    with path.open(encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping malformed JSON at %s:%d (%s)", path, line_number, exc)


def load_restaurant_businesses(raw_dir: Path = DEFAULT_RAW_DIR) -> list[dict]:
    """Load every business whose ``categories`` field mentions "Restaurants".

    Returns:
        Raw business dicts (all original fields kept), one per
        restaurant found in the dataset.
    """
    path = find_dataset_file(raw_dir, BUSINESS_FILENAME)
    return [b for b in iter_json_lines(path) if "restaurants" in (b.get("categories") or "").lower()]


def select_business_subset(
    businesses: list[dict],
    *,
    min_reviews_per_business: int = 20,
    max_reviews_per_business: int = 120,
    target_business_count: int = 18,
    target_total_reviews: tuple[int, int] = (500, 1000),
) -> list[dict]:
    """Deterministically pick a subset of businesses within Hito 0's target size.

    Candidates are bounded by ``review_count`` (avoids one huge
    business dominating the subset, and avoids near-empty ones) and
    sorted by ``business_id`` for a stable, reproducible pick — no
    randomness, so the same raw dataset always yields the same subset.

    Args:
        businesses: Candidate restaurant records, e.g. from
            ``load_restaurant_businesses``.
        min_reviews_per_business: Skip businesses with fewer reviews
            than this.
        max_reviews_per_business: Skip businesses with more reviews
            than this.
        target_business_count: Stop once this many businesses are
            selected (CLAUDE.md's 15-20 business target).
        target_total_reviews: Stop once cumulative ``review_count``
            falls in this ``(min, max)`` range, even if
            ``target_business_count`` hasn't been reached yet.

    Returns:
        Selected business records, in selection order.
    """
    candidates = sorted(
        (
            b
            for b in businesses
            if min_reviews_per_business <= (b.get("review_count") or 0) <= max_reviews_per_business
        ),
        key=lambda b: b["business_id"],
    )

    selected: list[dict] = []
    total_reviews = 0
    min_target, max_target = target_total_reviews
    for business in candidates:
        if len(selected) >= target_business_count:
            break
        projected = total_reviews + business["review_count"]
        if projected > max_target and total_reviews >= min_target:
            break
        selected.append(business)
        total_reviews = projected
        if total_reviews >= min_target and len(selected) >= 15:
            break
    return selected


def select_stratified_review_sample(reviews: pd.DataFrame, per_star: int) -> pd.DataFrame:
    """Deterministically pick up to ``per_star`` reviews per star value (1-5).

    Sorted by ``review_id`` within each star bucket (no randomness), so
    the same input always yields the same sample. If a bucket has
    fewer than ``per_star`` reviews, every review in it is kept.

    Used both to build a quality-check sample against the star-rating
    proxy (`scripts/validate_sentiment_proxy.py`) and to build the
    hand-labeling sheet for aspect ground truth
    (`scripts/build_labeling_sheet.py`), which has no free proxy.
    """
    parts = [
        group.sort_values("review_id").head(per_star) for _, group in reviews.groupby("stars", sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


def load_reviews_for_businesses(business_ids: set[str], raw_dir: Path = DEFAULT_RAW_DIR) -> list[dict]:
    """Stream the (large) review file once, keeping only matching businesses.

    Args:
        business_ids: Business IDs to keep reviews for.
        raw_dir: Root directory to search for the review file under.

    Returns:
        All matching review records found in a single pass.
    """
    path = find_dataset_file(raw_dir, REVIEW_FILENAME)
    return [r for r in iter_json_lines(path) if r.get("business_id") in business_ids]
