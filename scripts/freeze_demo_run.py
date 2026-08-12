"""Freeze one real run of the demo dataset to `app/frozen_demo.json`.

Why this exists: Streamlit Community Cloud runs the app in a remote
container that cannot reach a local Ollama server, so the live
"classify -> route -> draft/escalate" flow simply doesn't work there.
Rather than deploy a broken "Run pipeline" button, the cloud deployment
serves this frozen snapshot instead -- a real run's actual output,
captured once, locally, against the real model. See docs/DESIGN.md's
"Streamlit Cloud replay mode" entry.

This is NOT a demo generator that invents plausible-looking output --
every field here came from an actual `generate_draft_response` /
`flag_for_escalation` / `aggregate_reviews` call against the
deterministic fake-review dataset, the same one `--demo` mode always
uses. Re-run this script (needs a local Ollama server) any time the
prompt or fake dataset changes enough that the frozen snapshot should
be refreshed; the frozen file is committed to the repo, not gitignored,
specifically so the cloud deployment has something to serve.

Usage:
    python scripts/freeze_demo_run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aurapulse.aggregation import aggregate_reviews, summarize_other_aspect_usage
from aurapulse.pipeline_io import load_demo_data
from aurapulse.response_draft import flag_for_escalation, generate_draft_response
from aurapulse.routing import decide_route
from aurapulse.schemas import Route

OUTPUT_PATH = Path("app/frozen_demo.json")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    analyses, review_texts = load_demo_data()

    drafts = []
    escalations = []
    reviews = []  # full per-review pipeline trace, for app/streamlit_app.py's walkthrough view
    for analysis in analyses:
        route = decide_route(analysis)
        text = review_texts[analysis.review_id]
        entry = {
            "review_id": analysis.review_id,
            "business_id": analysis.business_id,
            "review_text": text,
            "overall_sentiment": analysis.overall_sentiment.value,
            "aspects": [{"aspect": a.aspect.value, "sentiment": a.sentiment.value} for a in analysis.aspects],
            "severity_flag": analysis.severity_flag,
            "route": route.value,
            "draft_text": None,
            "escalation_reason": None,
        }
        if route == Route.ESCALATE:
            flag = flag_for_escalation(analysis)
            entry["escalation_reason"] = flag.reason
            escalations.append(flag)
        elif route == Route.DRAFT_RESPONSE:
            print(f"Generating draft for {analysis.review_id}...")
            draft = generate_draft_response(analysis.review_id, analysis.business_id, text, analysis)
            entry["draft_text"] = draft.draft_text
            drafts.append((draft, text))
        reviews.append(entry)

    business_reports = aggregate_reviews(analyses)
    other_summary = summarize_other_aspect_usage(analyses)

    snapshot = {
        "business_reports": [r.model_dump(mode="json") for r in business_reports],
        "drafts": [
            {
                "review_id": d.review_id,
                "business_id": d.business_id,
                "draft_text": d.draft_text,
                "review_text": text,
            }
            for d, text in drafts
        ],
        "escalations": [e.model_dump(mode="json") for e in escalations],
        "other_aspect_summary": other_summary.model_dump(mode="json"),
        "reviews": reviews,
    }

    OUTPUT_PATH.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"\nWrote {len(reviews)} review(s) ({len(drafts)} draft(s), {len(escalations)} escalation(s)), "
        f"{len(business_reports)} business report(s) to {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
