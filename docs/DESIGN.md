# Design Decisions — AuraPulse

This file records architectural decisions and their trade-offs as the project evolves. One entry per decision, most recent first.

## Product framing: inconsistency detection, not just sentiment reporting

Status: resolved (2026-07-30).

AuraPulse is positioned as a tool that surfaces *recurring operational inconsistencies* (e.g. food praised consistently while wait time is consistently criticized, or an aspect that degrades over a specific time window) rather than a plain sentiment dashboard. This is a narrative/positioning decision, not a technical scope change for Hito 0: the underlying computation is the same aggregation already planned (sentiment distribution + recurring aspects per business, with temporal evolution if data supports it). The reframing changes how results are presented (as actionable product/ops signal) and gives a concrete north star for later milestones (e.g. flagging aspects whose negative-sentiment share crosses a threshold, or diverges sharply from the business's other aspects).

## `aspect` field: closed enum vs. free text

Status: resolved (2026-07-30) — **hybrid**.

**Decision:** closed enum (`food | service | price | cleanliness | wait_time | other`) plus an optional `other_detail: str | None` field, populated by the LLM only when `aspect == other`.

**Why:**
- A closed enum keeps aggregate reporting consistent (the whole point of "recurring aspects") and gives the classifier a bounded, evaluable task — precision/recall per class with a clear denominator.
- Free text alone would recreate the near-duplicate-category problem this decision was meant to avoid ("wait time" vs "we waited a long time" vs "slow service") and has no cheap ground-truth proxy to validate against.
- The `other_detail` escape hatch avoids silently discarding nuance that doesn't fit the fixed categories, and the *volume* of `other` responses becomes its own useful metric (a high rate of `other` signals the enum needs revisiting before it signals model error).

**Trade-off accepted:** if the enum turns out to be poorly designed, categories already assigned may need re-labeling later. Mitigated by reviewing `other_detail` content periodically to decide whether new categories are warranted — this is a cheap manual check, not a blocker for Hito 0.

## Orchestration framework (LangGraph vs. plain if/elif)

Status: not yet decided — out of scope for Hito 0. Per `CLAUDE.md`, LangGraph is not assumed by default. The routing logic for Hito 1 (positive → aggregation, negative w/o severity → response draft, negative w/ severity signal → escalation) should first be written as plain `if/elif`. Only introduce LangGraph if that conditional logic stops being legible, and document the reasoning here when the call is made.
