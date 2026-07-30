# Design Decisions — AuraPulse

This file records architectural decisions and their trade-offs as the project evolves. One entry per decision, most recent first.

## Ground truth convention: mixed positive+negative aspects → overall NEGATIVE

Status: resolved (2026-07-30), informed by the first offline eval against `llama3.1:8b`.

**Decision:** when a review mentions both a positive and a negative aspect, the fixture ground truth (`tests/test_fake_reviews.py` / `src/aurapulse/fake_reviews.py`) now expects `overall_sentiment = NEGATIVE`, not `NEUTRAL`. `NEUTRAL` is reserved for genuinely lukewarm/uncommitted language (e.g. "the food was fine, nothing special").

**Why:** the first eval run (`scripts/eval_fake_reviews.py`) scored 62.5% sentiment accuracy (5/8), with all 3 misses on fixtures where I had originally encoded "mixed signals = neutral". A manual diff of expected vs. actual per review showed the model was *consistently* calling these NEGATIVE instead — not random noise. That matches how people actually rate on Yelp (one real complaint drags the star rating down even if something else was good), so the original "neutral" convention was arguably the wrong ground truth, not a model failure. Re-running after the convention change would be expected to raise sentiment accuracy without any further prompt changes — worth confirming when the eval is re-run.

**Trade-off accepted:** this convention is a simplification (doesn't weigh *how* positive/negative each aspect is, just presence of both). Revisit if aggregation results downstream look wrong because of it.

## Known gap: `severity_flag` is unreliable from the local model

Status: acknowledged, not fixed — deferred, since `severity_flag` is explicitly unused by any Hito 0 logic.

Across two eval runs, `severity_flag` accuracy sat at 50% (4/8) both times, but *which* 4 reviews it got right changed between runs — i.e. it's not a stable, learnable signal with the current prompt, it's closer to noise. This is fine to defer because escalation routing (the only consumer of this field) is out of scope until Hito 1/2, but flagging it now so it isn't mistaken for "done" — likely needs few-shot examples or a narrower prompt when severity/escalation work actually starts.

## Classification backend: local LLM via Ollama (no paid API keys)

Status: resolved (2026-07-30).

**Decision:** classification (sentiment + aspect extraction, structured against `ReviewAnalysis`) runs against a **local model served by Ollama**, not a paid cloud API. Default model: `llama3.1:8b` (overridable via `OLLAMA_MODEL` in `.env`), called through the `ollama` Python client using its structured-output mode (`format=<json schema>`) so responses are constrained to the Pydantic schema shape.

**Why:**
- Hard constraint: the project must run at zero cost, no paid API keys anywhere (see `CLAUDE.md`).
- Rejected the pure rule-based/lexicon alternative: it would run for free too, but this project's explicit portfolio purpose is demonstrating LLM-based agent/orchestration work — a classifier with no LLM in the loop undercuts that goal.
- Rejected the hybrid (rules baseline + LLM for ambiguous cases) for now: more engineering than Hito 0 needs; worth revisiting in Hito 1+ if latency or local-hardware constraints become a real problem, at which point it would double as a genuine "when is more model power worth it" routing example.

**Trade-offs accepted:** slower inference than a cloud API, output quality depends on the local model's instruction-following, and it requires the user to install Ollama and pull a multi-GB model locally. Model choice is not final — swap `OLLAMA_MODEL` if `llama3.1:8b` underperforms on the eval set (see fake-review ground truth in `tests/test_fake_reviews.py`) or is too slow/heavy for the available hardware.

## Product framing: inconsistency detection, not just sentiment reporting

Status: resolved (2026-07-30).

AuraPulse is positioned as a tool that surfaces *recurring operational inconsistencies* (e.g. food praised consistently while wait time is consistently criticized, or an aspect that degrades over a specific time window) rather than a plain sentiment dashboard. This is a narrative/positioning decision, not a technical scope change for Hito 0: the underlying computation is the same aggregation already planned (sentiment distribution + recurring aspects per business, with temporal evolution if data supports it). The reframing changes how results are presented (as actionable product/ops signal) and gives a concrete north star for later milestones (e.g. flagging aspects whose negative-sentiment share crosses a threshold, or diverges sharply from the business's other aspects).

## `aspect` field: closed enum vs. free text

Status: resolved (2026-07-30) — **hybrid**.

**Decision:** closed enum (`food | service | price | cleanliness | wait_time | ambience | other`) plus an optional `other_detail: str | None` field, populated by the LLM only when `aspect == other`. `ambience` (comfort, noise, decor, atmosphere) was added after initial review as a distinct recurring driver of restaurant reviews that doesn't cleanly fit under `cleanliness` or `service`.

**Why:**
- A closed enum keeps aggregate reporting consistent (the whole point of "recurring aspects") and gives the classifier a bounded, evaluable task — precision/recall per class with a clear denominator.
- Free text alone would recreate the near-duplicate-category problem this decision was meant to avoid ("wait time" vs "we waited a long time" vs "slow service") and has no cheap ground-truth proxy to validate against.
- The `other_detail` escape hatch avoids silently discarding nuance that doesn't fit the fixed categories, and the *volume* of `other` responses becomes its own useful metric (a high rate of `other` signals the enum needs revisiting before it signals model error).

**Trade-off accepted:** if the enum turns out to be poorly designed, categories already assigned may need re-labeling later. Mitigated by reviewing `other_detail` content periodically to decide whether new categories are warranted — this is a cheap manual check, not a blocker for Hito 0.

## Orchestration framework (LangGraph vs. plain if/elif)

Status: not yet decided — out of scope for Hito 0. Per `CLAUDE.md`, LangGraph is not assumed by default. The routing logic for Hito 1 (positive → aggregation, negative w/o severity → response draft, negative w/ severity signal → escalation) should first be written as plain `if/elif`. Only introduce LangGraph if that conditional logic stops being legible, and document the reasoning here when the call is made.
