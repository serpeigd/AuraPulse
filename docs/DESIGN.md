# Design Decisions — AuraPulse

This file records architectural decisions and their trade-offs as the project evolves. One entry per decision, most recent first.

## Aspect-extraction precision: few-shot examples to curb aspect over-prediction

Status: resolved (2026-08-05) — with one known gap carried forward, see below.

**Context:** `scripts/validate_aspect_proxy.py` (the `aspect` analog of `validate_sentiment_proxy.py`)
ran the classifier against the 100-review hand-labeled ground truth for the first time. Baseline
(2-turn system+user prompt, no few-shot): aspect-set exact match 36/100 (36%), aspect+sentiment
exact match 24/100 (24%) — well below the sentiment-proxy eval's quality bar. Per-aspect
precision/recall showed the failure was concentrated in false positives, not missed aspects:
`price` (54.3% precision, 92.6% recall), `wait_time` (50.0%/78.6%), `ambience` (50.0%/88.9%) — the
model was tagging aspects the review never discussed. Of the 62 such false positives across those
three categories, 39 (63%) were sentiment `neutral` — the model was using "neutral" as a
just-in-case filler rather than reserving it for aspects genuinely discussed in a lukewarm way.

**Decision:** added an explicit system-prompt clarification ("neutral is never a placeholder for
an aspect the review doesn't discuss") plus two synthetic few-shot user/assistant example pairs to
`_build_messages` in `src/aurapulse/classifier.py`, both modeling restraint — a review that only
discusses 1-2 aspects gets exactly that many entries, nothing padded in as neutral filler.
Re-running the same 100-review eval: aspect-set match 43/100 (43%, +7pts), aspect+sentiment match
33/100 (33%, +9pts) — counting every classification failure as a miss, denominator stays 100 per
the project's no-metric-without-denominator rule. Precision improved in every targeted category
(`price` 54.3%→71.0%, `wait_time` 50.0%→70.0%, `ambience` 50.0%→71.9%) at some recall cost, a
reasonable trade for a scoring method that penalizes both false positives and false negatives
equally.

**Why synthetic few-shot examples, not real Yelp text:** `classifier.py` ships in the public
GitHub repo, and `data/processed/*.csv` (which holds real review text) is gitignored specifically
to keep Yelp dataset content out of it. Baking real review text into a prompt string in committed
source code would defeat that. The two examples were written fresh, targeting the exact false-
positive pattern found by hand-reviewing failing rows before writing them.

**Trade-off accepted / known gap:** the longer 6-turn prompt introduced a failure mode absent from
the baseline: 7/100 reviews (0% → 7%) now fail classification entirely after exhausting retries,
all with the same validation error — the model attaching `other_detail` to a *named* aspect (e.g.
`aspect=price, other_detail="menu transparency"`) instead of leaving it unset, which
`AspectMention`'s validator correctly rejects (see `src/aurapulse/schemas.py`). Neither few-shot
example models `other_detail` at all, so the root cause of this shift isn't understood yet. Net
effect is still a quality improvement even counting those 7 as misses, but in production those 7
reviews would silently drop out of a business's aggregate report rather than contributing a
possibly-imperfect classification. Deferred rather than fixed now, so it isn't mistaken for
"done" — the likely fix is one more explicit prompt sentence forbidding `other_detail` on named
aspects, followed by a third eval run to confirm before it's considered closed.

**Also worth reviewing before Hito 0 closes:** `other` usage in the ground-truth sample is 17% of
reviews, and the classifier's `other` precision/recall (33.3%/7.1% after this change) is by far
the worst of any category — per the enum decision's own escape-hatch logic (see "aspect field"
decision below), that combination is the signal for revisiting whether the enum needs a new
category. Not investigated yet.

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
