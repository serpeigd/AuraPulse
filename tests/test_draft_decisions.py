"""Unit tests for draft_decisions.record_draft_decision. Same shape as
test_escalation_delivery.py -- a real temp file (tmp_path), no client to mock."""

from __future__ import annotations

from pathlib import Path

from aurapulse.draft_decisions import record_draft_decision
from aurapulse.schemas import DraftDecision


def _decision(review_id: str, approved: bool, feedback: str | None = None) -> DraftDecision:
    return DraftDecision(review_id=review_id, business_id="b1", approved=approved, feedback=feedback)


def test_record_draft_decision_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "draft_decisions.jsonl"
    record_draft_decision(_decision("r1", approved=True), path=path)
    assert path.exists()


def test_record_draft_decision_writes_one_json_line(tmp_path: Path) -> None:
    path = tmp_path / "draft_decisions.jsonl"
    record_draft_decision(_decision("r1", approved=True), path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    written = DraftDecision.model_validate_json(lines[0])
    assert written.review_id == "r1"
    assert written.approved is True
    assert written.feedback is None


def test_record_draft_decision_appends_across_calls_instead_of_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "draft_decisions.jsonl"
    record_draft_decision(_decision("r1", approved=True), path=path)
    record_draft_decision(_decision("r2", approved=False, feedback="too generic"), path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    written = [DraftDecision.model_validate_json(line) for line in lines]
    assert [d.review_id for d in written] == ["r1", "r2"]
    assert written[1].approved is False
    assert written[1].feedback == "too generic"
