"""Escalation delivery: put ``EscalationFlag`` records where a human can find them.

The zero-cost constraint (see CLAUDE.md) rules out Slack/email APIs for now --
those need real accounts and, for most providers, a paid tier past trivial
volume. This writes one JSON line per escalation to a local file instead, the
same log-based pattern already used for classification/draft-generation
tracing (see docs/DESIGN.md's "Observability" entry). A human tails or reads
this file directly; nothing here sends anything over the network.

This is deliberately the simplest thing that could work, not a final answer --
see docs/DESIGN.md for why, and what would justify a real integration later.
"""

from __future__ import annotations

from pathlib import Path

from aurapulse.schemas import EscalationFlag

DEFAULT_ESCALATIONS_PATH = Path("data/escalations/escalations.jsonl")


def write_escalations(escalations: list[EscalationFlag], path: Path = DEFAULT_ESCALATIONS_PATH) -> None:
    """Append each escalation as one JSON line to ``path``.

    Args:
        escalations: Reviews routed to ``Route.ESCALATE``, typically from
            ``orchestrator.process_reviews()``'s ``Hito1Batch.escalations``.
        path: Destination file. Parent directories are created if missing.
            Always appended to, never overwritten or truncated -- escalations
            accumulate across runs so a human reviewing the file sees the
            full history, not just the latest run.
    """
    if not escalations:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for escalation in escalations:
            f.write(escalation.model_dump_json())
            f.write("\n")
