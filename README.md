<p align="center">
  <img src="docs/assets/logo.svg" alt="AuraPulse logo" width="120" height="120">
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
  rubrics independently — see [`docs/DESIGN.md`](docs/DESIGN.md) for the full story. The validated
  result itself is a real quality finding: every generated draft reads as generic/templated, a
  target for a future prompt pass on `response_draft.py`

**Not done yet:**
- A prompt pass on `response_draft.py` to fix the generic/templated-reply finding above
- Any delivery mechanism for escalations (email/Slack/dashboard) — currently just returned as data
- Any orchestration framework — LangGraph is deliberately not introduced yet; see
  `docs/DESIGN.md` for when it would be justified over the current `if/elif` routing

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
        ▼
aggregation.aggregate_reviews()          groups by business_id, flags inconsistencies
        │
        ▼
reporting.format_full_report()           human-readable text (scripts/generate_report.py)
```

`decide_route` is the project's concrete answer, so far, to "when does a graph orchestrator earn
its complexity": with 3 known routes it stays a 4-line `if/elif` (Option B in
[`docs/DESIGN.md`](docs/DESIGN.md) — decision kept separate from execution, but no framework
introduced). LangGraph is deliberately not used.

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
- No web framework, no database, no orchestration framework — this is a CLI/script pipeline

## Folder structure

```
AuraPulse/
├── src/aurapulse/          # Library code: schemas, classifier, routing, aggregation, reporting...
├── scripts/                # CLI entry points — one script per pipeline step or offline eval
├── tests/                  # pytest suite, one test_*.py per src/aurapulse/*.py module
├── data/
│   ├── raw/                # Yelp Open Dataset goes here (gitignored, manual download)
│   └── processed/          # Built review subsets / classified output (gitignored)
├── docs/
│   ├── DESIGN.md           # Architectural decision log, one entry per decision + trade-offs
│   └── assets/logo.svg
├── .github/workflows/ci.yml
├── pyproject.toml
├── .env.example
└── LICENSE
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # .venv\Scripts\activate.bat on Windows cmd.exe
pip install -e ".[dev]"
```

Install [Ollama](https://ollama.com) and pull the model used by default:

```bash
ollama pull llama3.1:8b
```

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
```

`build_subset.py`, `eval_fake_reviews.py`, `validate_sentiment_proxy.py`,
`validate_aspect_proxy.py`, `eval_draft_responses.py`, `eval_draft_quality.py`, and
`eval_severity_fake_reviews.py` need the raw Yelp dataset and/or a running local Ollama server.
`generate_report.py --demo` needs neither — it runs the same
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

## Tests and checks

```bash
pytest -q
ruff check src/ tests/ scripts/
mypy src/ scripts/ tests/
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
- **No delivery mechanism yet.** Drafts and escalations are returned as in-memory Pydantic
  objects (`DraftResponse`, `EscalationFlag`) — no email, Slack, or dashboard integration.
- **Generated draft replies read as generic/templated.** Confirmed by a human-validated
  LLM-judge eval (`scripts/eval_draft_quality.py`), not just a hunch — see `docs/DESIGN.md`. Tone
  and editability are solid; specificity/personalization is the known gap for a future prompt pass.
- **A multi-criteria LLM-judge prompt doesn't judge criteria independently.** Removing one
  unreliable rubric question destabilized verdicts on unrelated, already-validated ones —
  see `docs/DESIGN.md`. The whole prompt has to be re-validated after any change, not just the
  part that changed.
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
Not yet, and not by default. Routing between the 3 known outcomes (aggregate / draft / escalate)
is a plain `if/elif` function ([`src/aurapulse/routing.py`](src/aurapulse/routing.py)) — see
`docs/DESIGN.md` for the reasoning and what would make a graph orchestrator worth the added
complexity.

**Where's the actual review data?**
Not in this repo. Download the [Yelp Open Dataset](https://www.yelp.com/dataset) yourself
(requires accepting Yelp's terms) and drop it under `data/raw/`; `data/raw/` and
`data/processed/` are both gitignored.

## License

MIT — see [`LICENSE`](LICENSE).
