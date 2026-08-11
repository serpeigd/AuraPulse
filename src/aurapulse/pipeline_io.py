"""Loading already-classified review data for the Hito 1 pipeline.

Extracted out of ``scripts/run_pipeline.py`` so ``app/streamlit_app.py``
can load the same demo/real data without duplicating the logic — both
callers need the exact same (analyses, review_texts) shape to hand to
``orchestrator.process_reviews()``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aurapulse.fake_reviews import generate_fixed_dataset
from aurapulse.schemas import ReviewAnalysis

DEFAULT_INPUT_PATH = Path("data/processed/classified_reviews.jsonl")
DEFAULT_REVIEWS_PATH = Path("data/processed/review_subset.csv")


def load_demo_data() -> tuple[list[ReviewAnalysis], dict[str, str]]:
    """Deterministic fake reviews: known-correct ReviewAnalysis, no classification call needed."""
    fake_reviews = generate_fixed_dataset()
    analyses = [fr.expected for fr in fake_reviews]
    review_texts = {fr.expected.review_id: fr.text for fr in fake_reviews}
    return analyses, review_texts


def load_real_data(
    input_path: Path = DEFAULT_INPUT_PATH, reviews_path: Path = DEFAULT_REVIEWS_PATH
) -> tuple[list[ReviewAnalysis], dict[str, str]]:
    """Already-classified reviews (JSONL) + their raw text (CSV), joined on review_id.

    Raises:
        FileNotFoundError: if either file is missing, with a message
            pointing at what step produces it.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. This file is produced by classifying the review subset "
            "(Hito 0 step 7) -- not built yet. Try the demo dataset instead."
        )
    if not reviews_path.exists():
        raise FileNotFoundError(f"{reviews_path} not found. Run scripts/build_subset.py first.")

    with input_path.open(encoding="utf-8") as f:
        analyses = [ReviewAnalysis.model_validate_json(line) for line in f if line.strip()]

    reviews_df = pd.read_csv(reviews_path, dtype={"review_id": str})
    review_texts = dict(zip(reviews_df["review_id"], reviews_df["text"], strict=True))
    return analyses, review_texts
