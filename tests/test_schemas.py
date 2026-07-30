"""Tests for schema-level invariants that plain type hints don't catch."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aurapulse.schemas import Aspect, AspectMention, Sentiment


def test_other_aspect_requires_other_detail() -> None:
    with pytest.raises(ValidationError):
        AspectMention(aspect=Aspect.OTHER, sentiment=Sentiment.NEGATIVE, other_detail=None)


def test_other_aspect_rejects_blank_other_detail() -> None:
    with pytest.raises(ValidationError):
        AspectMention(aspect=Aspect.OTHER, sentiment=Sentiment.NEGATIVE, other_detail="   ")


def test_other_aspect_with_detail_is_valid() -> None:
    mention = AspectMention(aspect=Aspect.OTHER, sentiment=Sentiment.NEGATIVE, other_detail="parking")
    assert mention.other_detail == "parking"


def test_non_other_aspect_rejects_other_detail() -> None:
    with pytest.raises(ValidationError):
        AspectMention(aspect=Aspect.FOOD, sentiment=Sentiment.POSITIVE, other_detail="unexpected")


def test_non_other_aspect_without_detail_is_valid() -> None:
    mention = AspectMention(aspect=Aspect.FOOD, sentiment=Sentiment.POSITIVE)
    assert mention.other_detail is None
