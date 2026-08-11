"""Record a human's approve/reject call on a generated draft reply.

Same append-one-JSON-line-per-record pattern as
``escalation_delivery.py`` (see that module's docstring for the
zero-cost reasoning) -- kept as a separate small function rather than
factored into a shared helper, matching this project's own "extract a
helper at the third use, not the second" convention (see
``response_draft.py``'s ``_emit_trace``).

This exists to make docs/DESIGN.md's LangGraph trigger condition (a
reject/regenerate loop) something with real recorded evidence behind
it, instead of a hypothetical -- see ``app/streamlit_app.py``. No
regeneration is wired up to a rejection yet; this only logs the
decision.
"""

from __future__ import annotations

from pathlib import Path

from aurapulse.schemas import DraftDecision

DEFAULT_DECISIONS_PATH = Path("data/decisions/draft_decisions.jsonl")


def record_draft_decision(decision: DraftDecision, path: Path = DEFAULT_DECISIONS_PATH) -> None:
    """Append one human decision as a JSON line to ``path``.

    Args:
        decision: The human's approve/reject call, typically built in
            ``app/streamlit_app.py`` from a button click.
        path: Destination file. Parent directories are created if
            missing. Always appended to, never overwritten -- decisions
            accumulate across sessions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(decision.model_dump_json())
        f.write("\n")
