"""Tests for the Yelp dataset loader, against small synthetic JSONL
fixtures written to a tmp_path — never against the real (multi-GB,
gitignored) dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurapulse.data_loader import (
    DatasetNotFoundError,
    find_dataset_file,
    iter_json_lines,
    load_restaurant_businesses,
    load_reviews_for_businesses,
    select_business_subset,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_find_dataset_file_locates_nested_file(tmp_path: Path) -> None:
    nested = tmp_path / "some_zip_folder" / "yelp_dataset"
    nested.mkdir(parents=True)
    target = nested / "yelp_academic_dataset_business.json"
    target.write_text("{}", encoding="utf-8")

    found = find_dataset_file(tmp_path, "yelp_academic_dataset_business.json")

    assert found == target


def test_find_dataset_file_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError):
        find_dataset_file(tmp_path, "yelp_academic_dataset_business.json")


def test_find_dataset_file_raises_when_ambiguous(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "yelp_academic_dataset_business.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b" / "yelp_academic_dataset_business.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DatasetNotFoundError):
        find_dataset_file(tmp_path, "yelp_academic_dataset_business.json")


def test_iter_json_lines_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"a": 1}\nnot valid json\n{"a": 2}\n\n', encoding="utf-8")

    records = list(iter_json_lines(path))

    assert records == [{"a": 1}, {"a": 2}]


def test_load_restaurant_businesses_filters_by_category(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_jsonl(
        raw_dir / "yelp_academic_dataset_business.json",
        [
            {"business_id": "r1", "categories": "Restaurants, Italian", "review_count": 10},
            {"business_id": "s1", "categories": "Nail Salons, Beauty", "review_count": 10},
            {"business_id": "r2", "categories": "Fast Food, Restaurants", "review_count": 10},
            {"business_id": "n1", "categories": None, "review_count": 10},
        ],
    )

    restaurants = load_restaurant_businesses(raw_dir)

    assert {b["business_id"] for b in restaurants} == {"r1", "r2"}


def test_select_business_subset_respects_review_count_bounds() -> None:
    businesses = [
        {"business_id": "too_small", "review_count": 5},
        {"business_id": "just_right_a", "review_count": 50},
        {"business_id": "just_right_b", "review_count": 60},
        {"business_id": "too_big", "review_count": 500},
    ]

    selected = select_business_subset(
        businesses,
        min_reviews_per_business=20,
        max_reviews_per_business=120,
        target_business_count=18,
        target_total_reviews=(500, 1000),
    )

    ids = {b["business_id"] for b in selected}
    assert ids == {"just_right_a", "just_right_b"}


def test_select_business_subset_is_deterministic_and_sorted_by_id() -> None:
    businesses = [
        {"business_id": "zzz", "review_count": 30},
        {"business_id": "aaa", "review_count": 30},
        {"business_id": "mmm", "review_count": 30},
    ]

    selected = select_business_subset(businesses, target_total_reviews=(0, 10_000))

    assert [b["business_id"] for b in selected] == ["aaa", "mmm", "zzz"]


def test_select_business_subset_stops_at_target_business_count() -> None:
    businesses = [{"business_id": f"biz{i}", "review_count": 30} for i in range(30)]

    selected = select_business_subset(businesses, target_business_count=5, target_total_reviews=(0, 10_000))

    assert len(selected) == 5


def test_load_reviews_for_businesses_filters_by_id(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_jsonl(
        raw_dir / "yelp_academic_dataset_review.json",
        [
            {"review_id": "1", "business_id": "keep", "text": "a"},
            {"review_id": "2", "business_id": "drop", "text": "b"},
            {"review_id": "3", "business_id": "keep", "text": "c"},
        ],
    )

    reviews = load_reviews_for_businesses({"keep"}, raw_dir)

    assert {r["review_id"] for r in reviews} == {"1", "3"}
