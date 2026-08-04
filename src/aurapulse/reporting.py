"""Renders aggregation results as human-readable text.

Pure formatting functions, no I/O -- kept separate from
`scripts/generate_report.py` so the output format is directly
testable without needing a file on disk.
"""

from __future__ import annotations

from aurapulse.aggregation import BusinessReport, OtherAspectSummary
from aurapulse.schemas import Sentiment


def format_business_report(report: BusinessReport) -> str:
    """Render one business's report as readable text.

    Every number carries its own denominator (total_mentions,
    review_count) rather than a bare percentage -- see CLAUDE.md's
    "never report a metric without visible case count" rule.
    """
    lines = [f"=== {report.business_id} ({report.review_count} reviews) ==="]

    sentiment_parts = [f"{s.value} {report.sentiment_counts[s]}" for s in Sentiment]
    lines.append("Sentiment: " + " | ".join(sentiment_parts))

    if report.aspect_summaries:
        lines.append("")
        lines.append("Aspects (by mention count):")
        for s in report.aspect_summaries:
            mention_word = "mention" if s.total_mentions == 1 else "mentions"
            lines.append(
                f"  {s.aspect.value:<12}: {s.total_mentions} {mention_word} | "
                f"{s.positive_count} positive, {s.neutral_count} neutral, "
                f"{s.negative_count} negative ({s.negative_share:.0%} negative)"
            )
    else:
        lines.append("")
        lines.append("Aspects: none mentioned.")

    if report.inconsistent_aspects:
        lines.append("")
        lines.append("INCONSISTENCIES FLAGGED:")
        for flag in report.inconsistent_aspects:
            lines.append(f"  - {flag}")

    return "\n".join(lines)


def format_other_aspect_summary(summary: OtherAspectSummary) -> str:
    """Render the dataset-wide 'other' aspect usage summary.

    See docs/DESIGN.md: a high other_share signals the closed aspect
    enum needs a new category — this is meant to be skimmed
    periodically, not acted on automatically.
    """
    lines = [
        "=== Aspect enum coverage ===",
        (
            f"'other' used in {summary.other_mentions}/{summary.total_mentions} aspect mentions "
            f"({summary.other_share:.1%})"
        ),
    ]
    if summary.other_details:
        lines.append("Sample of what fell through to 'other':")
        for detail in summary.other_details[:20]:
            lines.append(f"  - {detail}")
        if len(summary.other_details) > 20:
            lines.append(f"  ... and {len(summary.other_details) - 20} more")
    return "\n".join(lines)


def format_full_report(
    business_reports: list[BusinessReport], other_aspect_summary: OtherAspectSummary
) -> str:
    """Render the complete report: every business, then the enum-coverage summary."""
    sections = [format_business_report(r) for r in business_reports]
    sections.append(format_other_aspect_summary(other_aspect_summary))
    return "\n\n".join(sections)
