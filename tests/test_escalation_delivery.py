"""Unit tests for escalation_delivery.write_escalations. All against a real
temp file (tmp_path) -- there's no client to mock, the function's whole job
is the file I/O."""

from __future__ import annotations

from pathlib import Path

from aurapulse.escalation_delivery import write_escalations
from aurapulse.schemas import EscalationFlag


def _flag(review_id: str, reason: str = "Flagged as severe; negative aspects: cleanliness") -> EscalationFlag:
    return EscalationFlag(review_id=review_id, business_id="b1", reason=reason)


def test_write_escalations_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "escalations.jsonl"
    write_escalations([_flag("r1")], path=path)
    assert path.exists()


def test_write_escalations_one_json_line_per_escalation(tmp_path: Path) -> None:
    path = tmp_path / "escalations.jsonl"
    write_escalations([_flag("r1"), _flag("r2")], path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    written = [EscalationFlag.model_validate_json(line) for line in lines]
    assert [f.review_id for f in written] == ["r1", "r2"]


def test_write_escalations_appends_across_calls_instead_of_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "escalations.jsonl"
    write_escalations([_flag("r1")], path=path)
    write_escalations([_flag("r2")], path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_write_escalations_with_empty_list_does_not_create_file(tmp_path: Path) -> None:
    path = tmp_path / "escalations.jsonl"
    write_escalations([], path=path)
    assert not path.exists()
