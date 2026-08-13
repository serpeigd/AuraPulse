<p align="center">
  <img src="docs/assets/AuraPulse_logo.png" alt="AuraPulse logo" width="360">
</p>

<h1 align="center">AuraPulse</h1>

<p align="center">
  <a href="https://github.com/serpeigd/AuraPulse/actions/workflows/ci.yml"><img src="https://github.com/serpeigd/AuraPulse/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://mypy-lang.org/"><img src="https://www.mypy-lang.org/static/mypy_badge.svg" alt="Checked with mypy"></a>
</p>

Detects recurring operational inconsistencies in restaurant reviews — e.g. food praised
consistently while wait time is consistently criticized — and turns that into actionable
product-improvement signal for the business. Built entirely on free, local tooling: **no paid
API keys anywhere**, sentiment/aspect classification runs on a local LLM via
[Ollama](https://ollama.com).

## Status: 🚧 Work in progress (Hito 0 complete, Hito 1 in progress)

This is a portfolio project, developed incrementally with documented design decisions (see
[`docs/DESIGN.md`](docs/DESIGN.md)). Hito 0 (classify → aggregate → report) is done. Hito 1
(route each review to a draft reply or a human escalation) has its first slice in place.

**Done — Hito 0:**
- `ReviewAnalysis` Pydantic schema (sentiment + per-aspect sentiment, closed aspect enum with an
  `other` escape hatch) — [`src/aurapulse/schemas.py`](src/aurapulse/schemas.py)
- Deterministic fake-review generator with known ground truth, so the pipeline is testable
  before a single real LLM call — [`src/aurapulse/fake_reviews.py`](src/aurapulse/fake_reviews.py)
- Local-LLM classifier (Ollama, structured output, retries, logging, zero cost) —
  [`src/aurapulse/classifier.py`](src/aurapulse/classifier.py)
- Yelp dataset loader + deterministic restaurant-review subset builder —
  [`src/aurapulse/data_loader.py`](src/aurapulse/data_loader.py)
- Offline evals: 100% sentiment accuracy against the fake-review ground truth, 85% (51/60)
  agreement against the free star-rating proxy on a stratified real-review sample
- Hand-labeling tooling for aspect ground truth (aspect has no free proxy, unlike sentiment) —
  [`scripts/build_labeling_sheet.py`](scripts/build_labeling_sheet.py)
- Aspect-extraction validated against 100 hand-labeled reviews and tuned with few-shot examples +
  output normalization: 44% aspect-set exact match, 34% aspect+sentiment exact match (up from a
  36%/24% baseline), 0 classification failures — see [`docs/DESIGN.md`](docs/DESIGN.md) for the
  full precision/recall breakdown and what didn't work along the way
- Structured per-call classification tracing (latency, retries, outcome) —
  [`src/aurapulse/classifier.py`](src/aurapulse/classifier.py)
- Per-business aggregation with inconsistency flagging (the core "food good, wait time
  consistently bad" signal) — [`src/aurapulse/aggregation.py`](src/aurapulse/aggregation.py)
- Human-readable reporting — [`src/aurapulse/reporting.py`](src/aurapulse/reporting.py)

**Done — Hito 1 (first slice):**
- Routing: a pure, LLM-free `decide_route` function reading already-classified fields (positive/
  neutral → aggregate, negative w/o severity → draft, negative w/ severity → escalate) — plain
  `if/elif`, no orchestration framework — [`src/aurapulse/routing.py`](src/aurapulse/routing.py)
- Draft-reply generation via a local LLM, enforced draft-only (no "publish" capability exists
  anywhere in this codebase) — [`src/aurapulse/response_draft.py`](src/aurapulse/response_draft.py)
- Escalation flagging, fully deterministic, no LLM call — same module
- End-to-end orchestrator wiring routing → handlers → aggregation —
  [`src/aurapulse/orchestrator.py`](src/aurapulse/orchestrator.py)
- `severity_flag` reliability (the ESCALATE trigger): balanced ground truth + few-shot fix took it
  from 100% recall / 25% specificity to 100%/100%, confirmed stable across repeat runs — see
  [`docs/DESIGN.md`](docs/DESIGN.md) for the fix, a false-alarm-turned-side-finding along the way,
  and a newly-documented (separate, pre-existing) flakiness in the small fake-review dataset
- Draft-quality eval: an LLM-as-judge, validated against an independent human reviewer before being
  trusted — 3 of 4 rubric criteria confirmed (tone, editability, genericness), the 4th kept in the
  prompt but excluded from the trusted report after a genuinely surprising finding: removing it
  destabilized the other three on unchanged drafts, meaning this model doesn't judge multi-criteria
  rubrics independently — see [`docs/DESIGN.md`](docs/DESIGN.md) for the full story
- Fixed the generic/templated-reply finding: `response_draft.py`'s prompt now requires a concrete
  fact → impact → action structure and bans the boilerplate phrases every original draft used.
  Human-confirmed improvement; the judge's `not_generic` criterion still disagrees on the new
  drafts, and the design doc explains why that's an expected limit of the validation, not a new
  bug — a validated judge is only validated for the output distribution it saw
- End-to-end pipeline script (`scripts/run_pipeline.py`): the first place `orchestrator.process_reviews()`
  actually runs outside of tests — classify (or `--demo`) → route → draft/escalate → aggregate →
  report, in one command
- Escalation delivery: `src/aurapulse/escalation_delivery.py` appends each escalation as a JSON
  line to a local file — zero-cost, no Slack/email integration (yet); see `docs/DESIGN.md`
- Streamlit demo UI (`app/streamlit_app.py`): interactive view of routing/aggregation/escalation
  — business reports, escalations, and (unlike a plain report) drafts a human can actually act on
- Reject → regenerate loop for drafts (`src/aurapulse/draft_graph.py`): a human rejecting a draft in
  the Streamlit UI regenerates it with their feedback, up to 3 attempts, before falling back to
  "needs a human-written reply." This is this project's **first and only use of LangGraph** —
  every decision leading up to it kept routing as a plain `if/elif`; this loop's pause/resume
  requirement (a human can close the browser mid-review and come back) is the concrete trigger
  condition `docs/DESIGN.md` had been describing abstractly since Hito 1 kicked off. See
  `docs/DESIGN.md` for the full design and why routing itself still doesn't need it
- Consolidated results table (see "Results" below) — every eval number obtained so far in one place
- Streamlit Cloud replay mode: a third "🧊 Instant demo" data source (`app/frozen_demo.json`,
  produced by `scripts/freeze_demo_run.py`) that serves one real, previously-captured run instead
  of calling anything live — the cloud deployment's container can't reach a local Ollama server,
  so this is what makes a public link possible without breaking the zero-cost/local-model
  constraint. Its own tab is a per-review walkthrough (sentiment, aspects, the actual routing
  decision, and its outcome) covering all 8 reviews, not just the ones that got a draft — read-only
  (no reject/regenerate — there's no reachable model to regenerate against). Deployed and live —
  see "Deploying" below
- Adaptive onboarding: the sidebar detects whether a local Ollama server is actually reachable and
  defaults to whichever data source will work; the instant demo auto-runs on page load so a cloud
  visitor sees real results with zero clicks. "Run pipeline" is disabled outright (not just a
  warning) whenever the current selection is known to fail before anyone clicks it. See
  `docs/DESIGN.md`'s UI-clarity entries

**Not done yet:**
- A real escalation delivery channel (email/Slack/dashboard) beyond the local JSONL log
- The `aspect` enum's `other` category — still the weakest performer, unrevisited since Hito 0

## Results

Every number below carries its own denominator (see CLAUDE.md's "never report a metric without
visible case count" rule) and links to the full breakdown of what was tried, what didn't work, and
why in [`docs/DESIGN.md`](docs/DESIGN.md) — this table is a summary, not the evidence itself.

| Metric | Result | Sample | Validated against |
|---|---|---|---|
| Sentiment accuracy (deterministic ground truth) | 100% | 8/8 fake reviews | Known-correct labels ([`fake_reviews.py`](src/aurapulse/fake_reviews.py)) |
| Sentiment agreement (real reviews) | 85% | 51/60 | Yelp star-rating proxy, stratified sample |
| Aspect-set exact match | 44% | 44/100 | 100 hand-labeled real reviews |
| Aspect+sentiment exact match | 34% | 34/100 | Same 100 hand-labeled reviews |
| Aspect classification failures | 0% | 0/100 | Same eval, after the `other_detail` normalization fix |
| `severity_flag` accuracy (balanced set) | 100% | 16/16 (8 true + 8 near-miss) | Hand-designed genuine-vs-emotional-language cases |
| Draft structural/policy compliance | 100% | 6/6 negative fake reviews | Hard constraints: word limit, no promised remedy, no false fix claim |
| Draft `appropriate_tone` (LLM-judge, human-validated) | 100% | 14/14 | Independent human reviewer |
| Draft `usable_with_minor_edits` (LLM-judge, human-validated) | 100% | 14/14 | Independent human reviewer |

**Explicitly not in this table:** the judge's `addresses_specific_complaint` and `not_generic`
criteria — both have documented human disagreement and are excluded from the trusted report; see
`docs/DESIGN.md`'s draft-quality-eval and genericness-fix entries for exactly why a number here
would be misleading rather than informative.

## Why this project exists

Portfolio project demonstrating agent/LLM orchestration and production practices. Its angle is
conditional routing: when a graph orchestrator earns its complexity versus a plain sequential
pipeline, decided and documented as the project grows rather than assumed up front.

## Non-negotiable constraints

- **Zero cost.** No paid API keys or cloud services, for anything — including classification.
- **Human in the loop.** Any future "respond to review" feature will only ever produce a draft;
  it will never auto-publish a response.

## Architecture

Every review passes through classification once, then a pure routing decision picks what (if
anything) happens next. Aggregation always runs, regardless of route:

```
Yelp restaurant reviews (data/processed/*.csv, built by scripts/build_subset.py)
        │
        ▼
classifier.classify_review()            local Ollama call, structured output → ReviewAnalysis
        │
        ▼
routing.decide_route()                  pure if/elif, no LLM call, no I/O
        │
        ├── AGGREGATE          (positive or neutral review) ─────────────────┐
        │                                                                    │
        ├── DRAFT_RESPONSE ──► response_draft.generate_draft_response()      │
        │   (negative, not severe)      local Ollama call → DraftResponse    │
        │                                                                    │
        ├── ESCALATE ────────► response_draft.flag_for_escalation()          │
        │   (negative, severity_flag set)  deterministic → EscalationFlag    │
        ▼                                                                    ▼
orchestrator.process_reviews()  (dispatches every review above, collects the results)
        │
        ├──────────────────────────────────────────────────────────────────┐
        ▼                                                                  ▼
aggregation.aggregate_reviews()          escalation_delivery.write_escalations()
  groups by business_id,                   appends to data/escalations/escalations.jsonl
  flags inconsistencies                    (zero-cost: local file, no Slack/email yet)
        │
        ▼
reporting.format_full_report()           human-readable text
        │
        ▼
scripts/run_pipeline.py                  the one command that runs all of the above
```

`decide_route` is the project's concrete answer to "when does a graph orchestrator earn its
complexity" for *routing*: with 3 known routes it stays a 4-line `if/elif` (Option B in
[`docs/DESIGN.md`](docs/DESIGN.md) — decision kept separate from execution, no framework needed).
Routing here is unchanged and still doesn't use LangGraph.

A *different* piece of Hito 1 does: the DRAFT_RESPONSE path, when driven from
`app/streamlit_app.py`, runs through `draft_graph.py`'s reject/regenerate loop instead of a single
`generate_draft_response()` call:

```
routing.decide_route() → DRAFT_RESPONSE
        │
        ▼
draft_graph.build_draft_graph()          LangGraph state machine, MemorySaver checkpointer
        │
        ▼
generate_draft ──► human_review (interrupt() — pauses here until a human responds)
    ▲                    │
    │        reject, attempts remain
    └────────────────────┘
                         │
              approve, or attempts exhausted
                         ▼
                        END                 outcome: approved | needs_human_rewrite
```

Why this needed LangGraph and routing doesn't: a human can reject a draft, close the browser, and
come back the next day to reject it again — the graph's state has to survive across Streamlit
reruns and pause indefinitely between clicks, which a plain Python loop can't do. `MemorySaver` is
in-process only (no persistence past a process restart — an accepted limit for a local single-user
demo); see `docs/DESIGN.md` for the full design and trade-offs.

## Tech stack

- **Python 3.11+**, [Pydantic v2](https://docs.pydantic.dev/) for every structured contract
  (`ReviewAnalysis`, `BusinessReport`, `DraftResponse`, `EscalationFlag`, ...)
- **[Ollama](https://ollama.com)** (`llama3.1:8b` by default) for sentiment/aspect classification
  and draft-reply generation — the only "LLM" in the system, run locally, zero cost
- **pandas** for the Yelp dataset subset/sampling steps, **openpyxl** for the aspect
  hand-labeling spreadsheet
- **pytest**, **ruff**, **mypy** for tests, linting, and type checking — all three run in CI
  ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) and are required to be green before
  a PR merges (see `CLAUDE.md`)
- **[Streamlit](https://streamlit.io)** (optional `[ui]` extra) for an interactive demo of the
  pipeline — the only UI layer in the project; everything else is a CLI script
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** (optional `[ui]` extra) — this
  project's one and only orchestration-framework use, scoped to the draft reject/regenerate loop
  (`src/aurapulse/draft_graph.py`). Routing itself is still a plain `if/elif`; see "Architecture"
  above and `docs/DESIGN.md` for why this specific loop earned the framework and routing didn't
- No database — this is a CLI/script pipeline plus one thin demo UI

## Folder structure

```
AuraPulse/
├── src/aurapulse/          # Library code: schemas, classifier, routing, aggregation, reporting...
├── scripts/                # CLI entry points — one script per pipeline step or offline eval
├── app/                    # Streamlit demo UI
│   ├── streamlit_app.py    # interactive view of the pipeline (live + frozen-replay modes)
│   └── frozen_demo.json    # committed snapshot for the frozen-replay data source (not gitignored)
├── tests/                  # pytest suite, one test_*.py per src/aurapulse/*.py module
├── data/
│   ├── raw/                # Yelp Open Dataset goes here (gitignored, manual download)
│   ├── processed/          # Built review subsets / classified output (gitignored)
│   ├── escalations/        # escalations.jsonl, written by run_pipeline.py (gitignored)
│   └── decisions/          # draft_decisions.jsonl, written by app/streamlit_app.py (gitignored)
├── docs/
│   ├── DESIGN.md           # Architectural decision log, one entry per decision + trade-offs
│   └── assets/AuraPulse_logo.png
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt        # Streamlit Community Cloud's default dependency lookup; `.[ui]`, see "Deploying"
├── .env.example
└── LICENSE
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # .venv\Scripts\activate.bat on Windows cmd.exe
pip install -e ".[dev]"
pip install -e ".[ui]"   # optional, only needed for the Streamlit demo (app/streamlit_app.py)
```

Install [Ollama](https://ollama.com) and pull the model used by default:

```bash
ollama pull llama3.1:8b
```

Ollama's desktop install runs as a background service that auto-starts with your machine. If
`ollama serve` prints `bind: address already in use` (Windows) / `address already in use` (macOS/
Linux), that's not an error — it means the server is already up on port 11434. Skip straight to
running a script; there's nothing to start.

Download the [Yelp Open Dataset](https://www.yelp.com/dataset) manually (requires accepting
Yelp's terms — not scriptable) and place it anywhere under `data/raw/`; the loader searches
recursively for `yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json`.

## Configuration

All configuration is optional environment variables (see [`.env.example`](.env.example)). No API
keys are required for anything in this repo.

**Note:** nothing in this codebase currently calls `load_dotenv()` (the `python-dotenv` dependency
in `pyproject.toml` isn't wired in yet), so a plain `.env` file is *not* picked up automatically —
copying `.env.example` to `.env` and editing it has no effect on its own. To actually override a
default, export the variable in your shell before running a script, e.g.:

```bash
export OLLAMA_MODEL=llama3.1:70b
python scripts/eval_fake_reviews.py
```

or source `.env` yourself (`set -a; source .env; set +a` on bash/zsh). Treat `.env.example` as
documentation of the two variables that matter, not a drop-in config file — yet.

| Variable       | Default                 | Purpose                                    |
|----------------|--------------------------|---------------------------------------------|
| `OLLAMA_HOST`  | `http://localhost:11434` | URL of the local Ollama server              |
| `OLLAMA_MODEL` | `llama3.1:8b`             | Model tag used for classification and drafts |

## Usage

```bash
python scripts/build_subset.py              # filter Yelp for restaurants, build the review subset
python scripts/eval_fake_reviews.py         # offline eval against deterministic ground truth
python scripts/validate_sentiment_proxy.py  # validate real reviews against the star-rating proxy
python scripts/build_labeling_sheet.py      # generate the aspect hand-labeling spreadsheet
python scripts/validate_aspect_proxy.py     # validate aspect extraction against hand-labeled ground truth
python scripts/eval_draft_responses.py      # structural/policy checks on generated draft replies
python scripts/eval_draft_quality.py        # human-validated LLM-judge quality eval for drafts
python scripts/eval_severity_fake_reviews.py  # severity_flag accuracy on a balanced deterministic set
python scripts/generate_report.py --demo    # print the aggregated report (no dataset or Ollama needed)
python scripts/run_pipeline.py --demo       # full Hito 1 flow: route -> draft/escalate -> report (needs Ollama)
python -m streamlit run app/streamlit_app.py  # same flow, drafts with reject-and-regenerate (needs Ollama + `.[ui]`)
```

`build_subset.py`, `eval_fake_reviews.py`, `validate_sentiment_proxy.py`,
`validate_aspect_proxy.py`, `eval_draft_responses.py`, `eval_draft_quality.py`,
`eval_severity_fake_reviews.py`, `run_pipeline.py`, and `streamlit_app.py` (even in demo mode —
drafting always calls the local model) need the raw Yelp dataset and/or a running local Ollama
server.

**Windows note:** `streamlit run ...` (the bare command) can fail with `"streamlit" no se reconoce
como un comando...` even right after `pip install -e ".[ui]"` — pip installed it to your Python's
`Scripts` folder, which isn't always on `PATH`. `python -m streamlit run ...` (used above) sidesteps
that entirely by not depending on `PATH` at all, and is the more portable form in general.
`generate_report.py --demo` needs neither of those — it runs the same
aggregation/reporting code against the project's own deterministic fake-review dataset
([`src/aurapulse/fake_reviews.py`](src/aurapulse/fake_reviews.py)), so the report format is
checkable without any setup. Real output from that command, against the 8 fake reviews checked
into the repo:

```
$ python scripts/generate_report.py --demo
[DEMO MODE] Using 8 deterministic fake reviews, not real data.

=== biz-alpha (3 reviews) ===
Sentiment: positive 1 | neutral 1 | negative 1

Aspects (by mention count):
  food        : 3 mentions | 1 positive, 1 neutral, 1 negative (33% negative)
  service     : 2 mentions | 1 positive, 0 neutral, 1 negative (50% negative)
  ambience    : 1 mention | 1 positive, 0 neutral, 0 negative (0% negative)
  cleanliness : 1 mention | 0 positive, 0 neutral, 1 negative (100% negative)
  price       : 1 mention | 0 positive, 1 neutral, 0 negative (0% negative)

=== biz-beta (2 reviews) ===
Sentiment: positive 0 | neutral 0 | negative 2

Aspects (by mention count):
  food        : 1 mention | 1 positive, 0 neutral, 0 negative (0% negative)
  wait_time   : 1 mention | 0 positive, 0 neutral, 1 negative (100% negative)
  price       : 1 mention | 1 positive, 0 neutral, 0 negative (0% negative)
  cleanliness : 1 mention | 0 positive, 0 neutral, 1 negative (100% negative)

=== biz-gamma (3 reviews) ===
Sentiment: positive 0 | neutral 0 | negative 3

Aspects (by mention count):
  service     : 2 mentions | 0 positive, 1 neutral, 1 negative (50% negative)
  wait_time   : 1 mention | 1 positive, 0 neutral, 0 negative (0% negative)
  ambience    : 1 mention | 0 positive, 0 neutral, 1 negative (100% negative)
  other       : 1 mention | 0 positive, 0 neutral, 1 negative (100% negative)
  cleanliness : 1 mention | 0 positive, 0 neutral, 1 negative (100% negative)

=== Aspect enum coverage ===
'other' used in 1/18 aspect mentions (5.6%)
Sample of what fell through to 'other':
  - parking availability
```

Once a real classified dataset exists at `data/processed/classified_reviews.jsonl` (one
`ReviewAnalysis` JSON object per line — not built by this repo yet, see "Not done yet" above),
`python scripts/generate_report.py --input data/processed/classified_reviews.jsonl` renders the
same report against it.

## Deploying

The Streamlit app runs two ways: locally with the two live data sources (needs Ollama), or as a
public Streamlit Community Cloud link using the frozen-replay data source (needs no server on the
visitor's end at all — see `docs/DESIGN.md`'s "Streamlit Cloud replay mode" entry for why the cloud
container can't run the live routes). It's deployed — connecting a new repo is a one-time manual
step in the project owner's own [Streamlit Community Cloud](https://share.streamlit.io) account:

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub account that owns
   this repo (it's public, so no extra access grant is needed).
2. **New app** → pick this repository, branch `main`, main file path `app/streamlit_app.py`.
3. Streamlit Cloud installs from [`requirements.txt`](requirements.txt) at the repo root
   automatically — nothing to configure.
4. Deploy. The app detects that no local Ollama server is reachable and auto-selects **🧊 Instant
   demo** — it loads and runs on its own, no click needed. The other two data sources need a
   local Ollama server the cloud container can't reach, so **Run pipeline** is disabled outright
   if either is picked there — nobody visiting the link can click into a broken run.

To refresh what the frozen snapshot shows (e.g. after a prompt change in `response_draft.py`):

```bash
python scripts/freeze_demo_run.py   # needs a local Ollama server; overwrites app/frozen_demo.json
git add app/frozen_demo.json && git commit -m "chore: refresh frozen demo snapshot"
```

Streamlit Cloud redeploys automatically on a push to `main`.

## Tests and checks

```bash
pytest -q
ruff check src/ tests/ scripts/ app/
mypy src/ scripts/ tests/ app/
```

All three run in CI on every push/PR to `main` ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)); a PR only merges once all three are green (see `CLAUDE.md`).

## Design decisions

Every non-trivial architectural call — the aspect schema, the classification backend, ground-
truth conventions, what was tried and reverted and why — is logged with its trade-offs in
[`docs/DESIGN.md`](docs/DESIGN.md).

## Limitations

- **Local inference is slow.** ~20-45s/review on modest CPU hardware — a full classification
  pass over the ~700-review Hito 0 subset takes hours, and evals deliberately run on smaller
  stratified samples instead (see `docs/DESIGN.md`).
- **Aspect extraction accuracy is moderate, not high.** 44% aspect-set exact match / 34%
  aspect+sentiment exact match against 100 hand-labeled reviews. Per-aspect precision/recall
  breakdown and what was tried is in `docs/DESIGN.md` — this is disclosed, not hidden.
- **Escalation delivery is a local file, not a real channel.** `escalation_delivery.py` appends
  to `data/escalations/escalations.jsonl` — no email, Slack, or dashboard integration yet. Drafts
  aren't delivered anywhere at all (by design — a human reads `run_pipeline.py`'s stdout, or the
  Streamlit UI, and copies what they want to send; see CLAUDE.md's non-negotiable no-auto-publish
  rule).
- **The draft reject/regenerate loop's checkpointer is in-process only.** `draft_graph.py` uses
  LangGraph's `MemorySaver` — zero-cost, no external store, but state is lost on a server restart
  mid-review (the human would have to click Run pipeline again). Fine for a local single-user demo;
  a real deployment would need a persistent checkpointer.
- **The public demo is frozen-replay only.** The deployed Streamlit Community Cloud app can only
  serve `app/frozen_demo.json` — the cloud container can't reach a local Ollama server, so the two
  live data sources (and the reject/regenerate loop that depends on them) are reachable only when
  running the app locally. Refreshing what the public link shows means re-running
  `scripts/freeze_demo_run.py` locally and committing the snapshot.
- **A multi-criteria LLM-judge prompt doesn't judge criteria independently.** Removing one
  unreliable rubric question destabilized verdicts on unrelated, already-validated ones —
  see `docs/DESIGN.md`. The whole prompt has to be re-validated after any change, not just the
  part that changed.
- **A validated LLM-judge is only validated for the output distribution it saw.** After fixing
  the generic-draft finding above, the judge's `not_generic` criterion kept flagging the
  *improved* drafts as generic too — human-confirmed disagreement. It was validated against
  boilerplate-phrase genericness specifically, never against the new fact→impact→action
  structure, so that disagreement doesn't carry the same weight as the original validation. See
  `docs/DESIGN.md`.
- **`.env` isn't auto-loaded.** `python-dotenv` is a dependency but nothing calls `load_dotenv()`
  yet, so `OLLAMA_HOST`/`OLLAMA_MODEL` must be real exported environment variables, not just lines
  in a `.env` file — see "Configuration" above.
- **Classification trace logs aren't wired to a file by default.** `classifier.py` emits structured
  JSON via the standard `logging` module (see `docs/DESIGN.md`'s "Observability" entry), but no
  script currently configures a handler or writes to `data/logs/` — that path in `docs/DESIGN.md`'s
  example `grep` command is illustrative of the intended workflow, not something that exists yet
  out of the box. Configure a `logging.FileHandler` yourself (or run with `2>logfile`) to capture it.
- **The Yelp dataset isn't shipped.** It must be downloaded manually after accepting Yelp's
  terms of use; nothing in this repo can fetch it for you.

## FAQ

**Why a local model via Ollama instead of an API like OpenAI or Anthropic?**
The project's cost constraint is non-negotiable: zero paid API keys, anywhere (see `CLAUDE.md`).
A rule-based/lexicon classifier would also be free, but this project's portfolio purpose is
specifically demonstrating LLM-based classification and routing, so the free option that keeps
an LLM in the loop — a local model via Ollama — was chosen over both alternatives (see
`docs/DESIGN.md`).

**Can AuraPulse reply to reviews automatically?**
No, and it never will. `response_draft.generate_draft_response` only ever produces a
`DraftResponse` for a human to read, edit, and send themselves — there is no "publish" or
"send" capability anywhere in this codebase. This is a product boundary, not a technical
limitation (see `CLAUDE.md`).

**Does this use LangGraph or another agent-orchestration framework?**
In exactly one place, deliberately. Routing between the 3 known outcomes (aggregate / draft /
escalate) is still a plain `if/elif` function
([`src/aurapulse/routing.py`](src/aurapulse/routing.py)) — 3 routes stayed legible without a
framework, so none was introduced there. The draft reject/regenerate loop is different: a human
can reject a draft in `app/streamlit_app.py`, have it regenerated with their feedback, reject
again, and the whole thing has to pause indefinitely between clicks (even across closing and
reopening the browser) — state a plain Python loop can't hold across reruns. That's
[`src/aurapulse/draft_graph.py`](src/aurapulse/draft_graph.py), built on LangGraph's
`interrupt()`/`Command(resume=...)` mechanism. See `docs/DESIGN.md` for the full design and why
the same "does a framework earn its complexity" question got two different answers in one project.

**Where's the actual review data?**
Not in this repo. Download the [Yelp Open Dataset](https://www.yelp.com/dataset) yourself
(requires accepting Yelp's terms) and drop it under `data/raw/`; `data/raw/` and
`data/processed/` are both gitignored.

## License and legal notice

Copyright © 2026 Sergio Peigneux d'Egmont ([@serpeigd](https://github.com/serpeigd)).

The source code in this repository is released under the [MIT License](LICENSE) — you may reuse,
modify, and redistribute it, including commercially, provided the copyright notice and the licence
text are kept. It is provided **as is, without warranty of any kind**; see the LICENSE file for the
full disclaimer.

That licence covers this repository's own code and documentation only. It does **not** extend to:

- **The Yelp Open Dataset.** Not distributed here and not covered by this licence — it is
  downloaded manually by each user under [Yelp's own Dataset Terms of Use](https://www.yelp.com/dataset),
  which restrict how the data may be used and redistributed. `data/raw/` and `data/processed/` are
  gitignored precisely so no Yelp content ever lands in this repository.
- **The models and tools this project runs on.** [Ollama](https://ollama.com) and the model weights
  it serves (`llama3.1:8b` by default) carry their own separate licences and acceptable-use terms.
- **Any review text or business name** appearing in output produced from the Yelp dataset — that
  content belongs to its original authors and to Yelp, not to this project.

Nothing here is a legal opinion. If you plan to use this project on real customer reviews, review
the applicable data-protection obligations (GDPR and equivalents) for the reviews you process
yourself: AuraPulse is a portfolio project and makes no compliance claim.
