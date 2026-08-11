"""Unit tests for pipeline_io -- extracted from scripts/run_pipeline.py so
app/streamlit_app.py can reuse the same loading logic (see docs/DESIGN.md)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aurapulse.fake_reviews import generate_fixed_dataset
from aurapulse.pipeline_io import load_demo_data, load_real_data
from aurapulse.schemas import ReviewAnalysis, Sentiment


def test_load_demo_data_matches_the_fixed_fake_dataset() -> None:
    analyses, review_texts = load_demo_data()
    fixed = generate_fixed_dataset()

    assert [a.review_id for a in analyses] == [fr.expected.review_id for fr in fixed]
    assert review_texts == {fr.expected.review_id: fr.text for fr in fixed}


def test_load_real_data_joins_classified_reviews_with_text(tmp_path: Path) -> None:
    input_path = tmp_path / "classified_reviews.jsonl"
    reviews_path = tmp_path / "review_subset.csv"

    analysis = ReviewAnalysis(review_id="r1", business_id="b1", overall_sentiment=Sentiment.POSITIVE)
    input_path.write_text(analysis.model_dump_json() + "\n", encoding="utf-8")
    pd.DataFrame([{"review_id": "r1", "text": "Great food!"}]).to_csv(reviews_path, index=False)

    analyses, review_texts = load_real_data(input_path, reviews_path)

    assert [a.review_id for a in analyses] == ["r1"]
    assert review_texts == {"r1": "Great food!"}


def test_load_real_data_raises_helpful_error_when_input_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="classified_reviews.jsonl"):
        load_real_data(tmp_path / "classified_reviews.jsonl", tmp_path / "review_subset.csv")


def test_load_real_data_raises_helpful_error_when_reviews_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "classified_reviews.jsonl"
    analysis = ReviewAnalysis(review_id="r1", business_id="b1", overall_sentiment=Sentiment.POSITIVE)
    input_path.write_text(analysis.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="review_subset.csv"):
        load_real_data(input_path, tmp_path / "review_subset.csv")
