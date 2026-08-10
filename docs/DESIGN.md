# Design Decisions — AuraPulse

This file records architectural decisions and their trade-offs as the project evolves. One entry per decision, most recent first.

## Fixing the generic-draft finding — and a judge that didn't generalize to the fix

Status: resolved (2026-08-10), with a documented limit on how far the judge's validation reaches.

**Fix.** `response_draft.py`'s prompt now requires three explicit things per draft, in order: (1)
name the concrete fact from the review, (2) acknowledge its specific impact, (3) name one concrete
internal action tied to that complaint — plus an explicit ban on the boilerplate phrases every one
of the 14 originally-judged drafts used ("we take this seriously," "thank you for your feedback,"
etc.). Two new synthetic few-shot pairs demonstrate the pattern, written to avoid every banned
phrase themselves. Re-ran `scripts/eval_draft_quality.py` against the same 14 candidates:
`appropriate_tone` and `usable_with_minor_edits` held at 14/14 (no regression), confirming the
earlier lesson about validating the whole prompt, not just the changed part.

**The judge said `not_generic` was still 0/14 — but manual inspection and an explicit second human
check both disagreed with it.** The new drafts are substantively different: e.g. `fake-004` now
opens with "A 45-minute wait before anyone even took your order is too long" (the literal fact)
and closes with "we're reviewing how tables were staffed and assigned that night" (a concrete
action) — not the old templated "Dear valued customer... we take this seriously" pattern. The
human reviewer confirmed the fix works. Working theory for the disagreement: the original
human-validation pass (see the draft-quality eval entry below) only ever showed the judge
phrase-level boilerplate drafts; it never validated the judge's calibration on this new draft
*style* (an explicit fact → impact → action structure that, being consistently applied across all
14, may itself read to the judge as a structural template even though the content is now
specific). That's a different question than the one originally validated, and this eval's
human-agreement check does not extend to it.

**Decision:** trust the human read over the judge here, since the judge's `not_generic` calibration
was validated against boilerplate-phrase genericness specifically, not against structural-template
genericness — an untested generalization, not a re-confirmed one. Did not re-run a full 14-item
human-validation pass a second time for this narrower question given the same-session evidence
already gathered (spot-checked drafts + a direct human confirmation). Flagging as a real
methodological limit for future draft-quality work: **a validated LLM-judge is only validated for
the distribution of outputs it was validated against** — a large-enough prompt change to the
generator can produce outputs outside that distribution, and the judge needs re-validating against
the new distribution before being trusted on it again, exactly like a classifier eval needs a fresh
sample when the input distribution shifts.

## Draft-quality eval: LLM-as-judge validated against a human, with a methodological surprise

Status: resolved (2026-08-07) — 3/4 rubric criteria validated and trusted; the 4th kept but not trusted.

**Context.** `scripts/eval_draft_responses.py` (Hito 1) only checks structural/policy constraints
(word limit, no promised remedy, no false fix claim) — nothing about whether a draft is actually
*good*. There's no free ground truth for reply quality the way star ratings proxy sentiment, so
three options were on the table: (A) LLM-as-judge with the same local model, (B) pure human
review, (C) hybrid — build the judge, validate its verdicts against an independent human's on the
same drafts before trusting it, matching the project's own established pattern for anything
without a free proxy (aspect needed hand-labeling before the extractor was trusted; this needed
the same for judge-then-generation trust). Chose C.

**Rubric design.** Four specific, checkable yes/no questions instead of one holistic "rate 1-10"
score (`src/aurapulse/draft_judge.py`, `DraftQualityVerdict`) — vague holistic scores are more
exploitable by the self-evaluation bias of using the same model (`llama3.1:8b`) to grade its own
output. Generated + judged drafts for 14 candidate reviews (the 6 NEGATIVE reviews in
`generate_fixed_dataset()` + the 8 near-miss cases in `generate_severity_dataset()`), published as
an interactive artifact for independent human review.

**First human-agreement result:** 3 of 4 criteria matched the human's verdicts on all 14 drafts —
`appropriate_tone` 14/14, `usable_with_minor_edits` 14/14, `not_generic` 0/14 (both judge and
human agreed every draft *is* generic — a real quality finding about `response_draft.py`, not a
judge failure; see "Not done yet" below). The 4th, `addresses_specific_complaint`, scored **0/14
agreement** — the human's read: the judge conflated "wrapped in boilerplate apology language"
with "doesn't address the complaint's substance," marking replies False even when they explicitly
named the complaint (e.g. a reply literally containing "a long wait" was still marked as not
addressing a "45 minute wait" complaint).

**First fix attempt: revise the criterion's wording**, informed directly by the human's specific
disagreements (an explicit paraphrase-counts example, an explicit "don't let boilerplate wrapper
language count against this" instruction, an explicit carve-out for reviews too vague to have a
concrete detail to respond to). Re-judged the *same* 14 drafts (no regeneration needed). Result:
4/14 agreement, up from 0/14 but nowhere near acceptable — and in the case the revision's own
worked example almost exactly matched (paraphrasing a "45 minute wait" as "a long wait"), the
judge still failed it. Conclusion: this is a capability ceiling for an 8B quantized model on this
specific semantic-equivalence judgment, not a prompt-wording problem — more prompt iteration on
the same axis wasn't going to close the gap, echoing the `other_detail` and `severity_flag`
lessons earlier in this project about knowing when to stop iterating blindly.

**Second attempt — and the actual finding worth remembering: dropping the criterion entirely
broke the criteria that were already validated.** The natural next move was to remove
`addresses_specific_complaint` from the schema and prompt (down to 3 questions) and keep the 3
that worked. Doing that and re-running the *same* eval flipped `appropriate_tone` from 14/14 True
to **0/14 True** — on unchanged draft text, with a question whose own wording was never touched.
Isolated the cause with a 3-item spot check re-running the exact original 4-question prompt
verbatim: `appropriate_tone` came back to 3/3 True immediately. Confirmed: **this model does not
judge these rubric criteria independently** — the set of questions present in the prompt measurably
affects verdicts on unrelated ones, at least for this task/model. This is a bigger and more
useful finding than "one criterion doesn't work": it means a multi-criteria LLM-judge prompt has
to be validated and re-validated as a *whole*, not criterion-by-criterion — changing or removing
one question, even one nobody trusts, can silently invalidate calibration already established for
the others.

**Final decision:** kept the exact 4-question prompt structure as originally validated.
`addresses_specific_complaint` is still asked (removing it destabilizes the rest) but its value is
never surfaced in the trusted aggregate report — `scripts/eval_draft_quality.py`'s
`VALIDATED_CRITERIA` covers only the 3 confirmed criteria, and `DraftQualityVerdict`'s docstring
states plainly which field not to trust and why. Re-ran the full 14-drafts eval one more time
against the restored prompt to confirm reproducibility: `appropriate_tone` 14/14, `usable_with_minor_edits`
14/14, `not_generic` 0/14 — an exact match to the original human-validated run.

**Not done yet (a real quality finding, not just an eval-infrastructure one):** both the human
reviewer and the judge agree every one of the 14 drafts reads as generic/templated ("Dear valued
customer... we take this seriously... Thank you... — The team") even when the underlying content
is on-topic. `response_draft.py`'s prompt hasn't been iterated on for this specific failure mode
yet — worth a future pass, informed by the human's own suggested fix pattern: acknowledge the
concrete fact, its impact, and a specific action, not just a templated apology wrapper.

## Fixing `severity_flag` reliability: bigger ground truth first, then the same few-shot playbook

Status: resolved (2026-08-06).

**Context.** `severity_flag` had been flagged as unreliable since Hito 0 (50% accuracy, not the
same cases each run — see the "Known gap" entry further down), but was never fixed because nothing
consumed it until Hito 1 routing shipped. That 50% number came from `generate_fixed_dataset()`,
which has exactly **one** `severity_flag=True` case among eight reviews — nowhere near enough
signal to diagnose anything. Before touching the prompt, added `generate_severity_dataset()`
(`src/aurapulse/fake_reviews.py`, `SeverityCase`) and `scripts/eval_severity_fake_reviews.py`: 8
genuinely severe cases across different categories (foodborne illness, injury, health-code
violation, harassment, fraud, allergic reaction, fire, physical altercation) and 8 "near miss"
cases — ordinary complaints in emotionally intense language ("worst meal of my life", "disgusting",
"nightmare") that aren't actually safety/health/legal issues.

**Baseline on the real denominator:** 100% recall (8/8 genuinely severe cases correctly flagged) but
only 25% specificity (2/8 — six of the eight "near miss" cases were false-flagged). The model was
conflating emotional intensity with genuine severity, not missing real issues. Worth naming
explicitly: 100%-recall-poor-precision is the *safer* of the two possible failure directions for a
field that gates human escalation — better some false alarms than a missed emergency — but 75%
false-positive rate on ordinary complaints would still flood a human reviewer and make the flag
useless in practice (the "cried wolf" problem).

**Fix — same playbook as the aspect-precision fix:** reinforced the `severity_flag` system-prompt
bullet ("this depends on WHAT happened, not how dramatically the reviewer describes it") plus two
new synthetic few-shot pairs, one demonstrating an intensely-worded ordinary complaint staying
`False`, one demonstrating a calmly-worded genuine safety issue staying `True` — deliberately
different text from `eval_severity_fake_reviews.py`'s fixtures, so the fix teaches the general
tone-vs-substance distinction rather than memorizing eval sentences (which would invalidate the
eval as a generalization test). Result, confirmed stable across two separate runs: **100% accuracy**
(8/8 recall, 8/8 specificity) — the false-positive problem fully resolved on this dataset.

**A false alarm worth recording.** The first pass at verifying this fix looked like it broke the
mixed-positive+negative-aspects → NEGATIVE convention from the aspect-precision fix (three
`generate_fixed_dataset()` reviews started coming back NEUTRAL). Before shipping that as a
trade-off, re-ran the *unmodified* `main` prompt under the same conditions — and got the *same*
failure pattern on the *unchanged* code. `generate_fixed_dataset()`'s 8-review sample turns out to
already be too small and noisy to reliably attribute a specific outcome to a specific prompt change
— a lesson to carry forward, not just for this change. Also re-ran the trusted, larger 100-review
real-data aspect eval (`scripts/validate_aspect_proxy.py`) as the actual regression check: 42%/32%
match, 0 classification failures — within the run-to-run noise band already observed for that eval
across prior sessions (36–44% swings on an *unchanged* prompt), not a real regression.

**New known gap, discovered as a side effect, not fixed here:** the mixed-positive+negative →
NEGATIVE convention is flakier on `generate_fixed_dataset()`'s tiny sample than previously
documented — it isn't specific to this change, and predates it. Worth either expanding that fixed
dataset's mixed-sentiment coverage or accepting the noise and relying on the larger real-data evals
for sentiment-convention regression checks going forward.

## Hito 1 kickoff: routing structure, draft generation, escalation

Status: first slice resolved (2026-08-06); several sub-decisions below still open.

**Routing structure — Option B (decide, then act, kept separate).** Three structures were on the
table: (A) one function that both decides a review's route and executes the action inline; (B) a
pure `decide_route(analysis) -> Route` decision function, separate from a handler per route,
wired together by a thin orchestrator; (C) the decision itself persisted as a `RoutingDecision`
record, fully decoupled from execution — the natural stepping stone to a graph orchestrator if one
is ever justified. Chose B. Reasoning: `severity_flag` (the ESCALATE trigger) is already known to
be unreliable (see the "Known gap" entry below in this same file, further down) — the one thing
worth being able to test cheaply and repeatedly, without mocking LLM calls or generating real
drafts, is "does routing decide correctly given a classification". B makes `decide_route` pure and
trivially testable (`tests/test_routing.py` — parametrized over every sentiment/severity
combination, no mocks) without C's extra persistence layer, which isn't earning its cost yet at 3
routes. `decide_route` itself makes **no LLM call** — by the time a review reaches it,
classification already made the only judgment call that matters; routing just reads the result.
This is also the concrete basis for the project's core "when is a graph orchestrator justified"
question: as long as this stays a 4-line `if/elif` (see `src/aurapulse/routing.py`), LangGraph
would be unjustified complexity.

**Neutral reviews route to AGGREGATE, not DRAFT_RESPONSE.** CLAUDE.md's original routing spec
only says "positive → aggregation"; neutral was ambiguous. Decided: neutral behaves like positive
(aggregate only). A "food was fine, nothing special" review isn't a complaint that warrants a
reply — treating every non-positive review as draft-worthy would generate noise for business
owners to wade through.

**Draft generation — new LLM call, enforced draft-only.** `response_draft.generate_draft_response`
calls the same local Ollama model via its own system prompt (not reusing the classifier's), with
explicit constraints: never promise a specific remedy (refund/discount/replacement — that's the
owner's call), never claim the business already fixed the issue (no way to know that), stay under
~100 words, sign off generically. Per CLAUDE.md's non-negotiable rule, nothing in this codebase
has a "publish" or "send" capability at all — `DraftResponse` is a plain data object with nowhere
to go except back to a human. Ran `scripts/eval_draft_responses.py` (structural/policy checks, not
a quality judgment — no cheap ground truth exists for "is this reply good") against the 6 NEGATIVE
reviews in the deterministic fake-review dataset: 6/6 generated successfully, 100% respected every
hard constraint (word limit, no promised remedy, no false fix claim). One soft check,
`mentions_a_complained_aspect` (does the draft text contain the literal aspect name, e.g. "wait
time"), scored 4/6 (66.7%) — but manually reading the 2 "misses" (`fake-004`: "apologize for the
long wait" instead of "wait time"; `fake-006`: "noise level... atmosphere" instead of "ambience")
shows the model addressed the complaint correctly in both cases, just in different words than the
enum's literal term. The check itself is a weak literal-substring proxy, not a real quality gap —
recorded here rather than silently treating 66.7% as if it meant something worse than it does.

**Escalation — deterministic, no LLM.** `response_draft.flag_for_escalation` only formats a
reason string from fields classification already produced (`severity_flag` + which aspects are
negative). No model call, no failure mode beyond "the input wasn't actually flagged as severe."

**Known gaps, not yet closed:**
- `severity_flag` reliability (documented separately in this file, below) directly gates how much
  the ESCALATE route can be trusted — routing structure and escalation formatting are done, but
  shipping this to a real user still means escalations may be noisy until that's addressed.
- No LLM-judge quality eval for drafts exists yet — only the structural/policy eval above. Whether
  a draft is actually *good* (tone, relevance, doesn't sound robotic) is unmeasured.
- No delivery mechanism for `EscalationFlag`s — they're returned as data from `process_reviews`,
  with nowhere to go yet (email/Slack/dashboard are all out of scope for this slice).

## Fixing the `other_detail` failure mode: code-level normalization, not prompt engineering

Status: resolved (2026-08-06).

**Context:** the aspect-precision few-shot fix (see below) left a known gap — 7/100 reviews
failing classification because the model attaches `other_detail` to a *named* aspect instead of
leaving it unset, and `AspectMention`'s validator correctly rejects that shape. First attempt: add
one more explicit system-prompt sentence forbidding it. Re-running the 100-review eval showed this
made things *worse*, not better — 8/100 failures (up from 7) and lower match rates (40%/28% vs.
43%/33% without the sentence). Four of the failing review IDs were identical across both runs,
suggesting a real, content-driven tendency rather than pure inference noise, but the rest churned
between runs — CPU-backed local inference isn't perfectly deterministic even at `temperature=0`.
Conclusion: this model doesn't reliably follow one more prompt constraint layered on an already
6-turn conversation, and spending a third ~40-minute eval run iterating blindly on more prompt text
wasn't a good bet.

**Decision:** stopped trying to prevent the shape via prompt and instead normalize it in code.
`classify_review` now parses the model's raw JSON, strips `other_detail` from any aspect mention
where `aspect != OTHER` (`_normalize_other_detail` in `src/aurapulse/classifier.py`), *then*
validates — instead of validating the raw output and discarding the whole review on failure. The
aspect and sentiment the model assigned (e.g. `price: neutral`) are very likely still correct; only
the stray free-text note was ever the problem. `AspectMention`'s validator in
`src/aurapulse/schemas.py` was deliberately left untouched and still strict — it's protecting a
real invariant for any other caller of the schema, and the fix belongs at the LLM-output boundary,
not in the shared data model. Every strip is logged and counted in the trace (see "Observability"
below), so its frequency stays visible rather than silently absorbed.

**Result:** re-ran the 100-review eval a third time. Failures dropped from 7 → **0/100**, and match
rates improved further to **44%/34%** (up from 43%/33% with the failures still happening) — the
previously-lost reviews turned out to classify correctly once given the chance. Zero added
inference cost or latency, since it's a pure post-processing step on output already received.

**Why this over loosening the schema's own validator:** the alternative — letting
`AspectMention.other_detail` be set on any aspect — would remove a real correctness invariant for
every producer of that model, not just this one LLM's known quirk. Confining the leniency to
`classify_review`'s LLM-output handling keeps the schema's contract meaningful everywhere else
(tests, any future non-LLM producer of `ReviewAnalysis`).

## Observability: structured per-call trace log, not a dashboard

Status: resolved (2026-08-05).

**Decision:** `classify_review` now emits exactly one structured JSON log line per call, at INFO
level, regardless of outcome (`_emit_trace` in `src/aurapulse/classifier.py`). Payload: `review_id`,
`model`, `attempts`, `elapsed_ms`, `outcome` (`success` / `schema_invalid` / `connection_error`),
and a short `error` string when the call failed. No review or business text is included, so trace
lines are safe to keep around without becoming a second copy of dataset content.

**Why now:** before this, the only way to answer "how often does classification fail, and how
long does it take" was to read a live eval run's console output by hand (exactly what happened
after fixing the aspect-precision issue above — the 7% `other_detail` failure rate was only
noticed because I was staring at scrollback). That doesn't scale past a single manual eval run,
and Hito 1's routing logic (positive → aggregation, negative w/o severity → draft, negative w/
severity → escalation) is going to need exactly this kind of per-call signal to debug misrouting.

**Why log lines, not a real APM/tracing backend:** the zero-cost constraint rules out any paid
observability service. A local JSON-lines log is the free alternative that still answers the
questions that matter here — `grep '"event": "classification"' data/logs/*.log | jq` gets
failure rate, latency distribution, and retry counts without new infrastructure. Revisit this if
Hito 1+ needs cross-call aggregation frequently enough that hand-grepping becomes the bottleneck —
at that point a small script that folds the JSONL into a summary table would be the next step, not
a hosted tracing service.

**Trade-off accepted:** this is call-level tracing, not distributed tracing — no span/trace IDs
linking a review's classification to its later routing decision (Hito 1+) or to the aggregation
step that consumes it. Deliberately deferred: Hito 0 has no multi-step pipeline yet for spans to
usefully connect, so building that now would be speculative. Revisit when Hito 1's routing exists.

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

**Trade-off accepted / known gap — resolved 2026-08-06, see "Fixing the `other_detail` failure
mode" below:** the longer 6-turn prompt introduced a failure mode absent from the baseline: 7/100
reviews (0% → 7%) now failed classification entirely after exhausting retries, all with the same
validation error — the model attaching `other_detail` to a *named* aspect (e.g. `aspect=price,
other_detail="menu transparency"`) instead of leaving it unset, which `AspectMention`'s validator
correctly rejects (see `src/aurapulse/schemas.py`). Net effect was still a quality improvement even
counting those 7 as misses, but in production those 7 reviews would have silently dropped out of a
business's aggregate report. Fixed at the code level rather than the prompt — see below.

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

Status: resolved (2026-08-06) — see "Fixing `severity_flag` reliability" above.

Across two eval runs, `severity_flag` accuracy sat at 50% (4/8) both times, but *which* 4 reviews it got right changed between runs — i.e. it's not a stable, learnable signal with the current prompt, it's closer to noise. This was fine to defer while escalation routing was out of scope, but became the top-priority Hito 1 gap once routing shipped (see "Hito 1 kickoff" above) — a routing decision is only as trustworthy as the field it routes on.

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

Status: superseded by "Hito 1 kickoff: routing structure, draft generation, escalation" above —
kept here for the history. Original framing (written before Hito 1 started): not yet decided,
out of scope for Hito 0. Per `CLAUDE.md`, LangGraph is not assumed by default. The routing logic
for Hito 1 (positive → aggregation, negative w/o severity → response draft, negative w/ severity
signal → escalation) should first be written as plain `if/elif`. Only introduce LangGraph if
that conditional logic stops being legible, and document the reasoning here when the call is
made.

That call has now been made once, for the first slice: `decide_route` is a 4-line `if/elif`
(see `src/aurapulse/routing.py`), and LangGraph was judged unjustified at 3 routes — see the
"Hito 1 kickoff" entry above for the full reasoning. This remains an open question going
forward, not a closed one: revisit if routing logic grows past what stays legible as `if/elif`.
