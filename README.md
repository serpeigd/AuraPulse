<p align="center">
  <img src="docs/assets/logo.svg" alt="AuraPulse logo" width="120" height="120">
</p>

# AuraPulse

[![CI](https://github.com/serpeigd/AuraPulse/actions/workflows/ci.yml/badge.svg)](https://github.com/serpeigd/AuraPulse/actions/workflows/ci.yml)

Detects recurring operational inconsistencies in restaurant reviews — e.g. food praised
consistently while wait time is consistently criticized — and turns that into actionable
product-improvement signal for the business. Built entirely on free, local tooling: **no paid
API keys anywhere**, sentiment/aspect classification runs on a local LLM via
[Ollama](https://ollama.com).

## Status: 🚧 Work in progress (Hito 0)

This is a portfolio project, developed incrementally with documented design decisions (see
[`docs/DESIGN.md`](docs/DESIGN.md)). Current milestone (Hito 0) target: a working pipeline from
raw Yelp reviews to an aggregated reputation/inconsistency report.

## Features

- **Structured, validated classification.** Every review is classified against a Pydantic
  schema (`ReviewAnalysis`): overall sentiment, zero or more aspect mentions (each with its own
  local sentiment), and a `severity_flag` reserved for future escalation routing — see
  [`src/aurapulse/schemas.py`](src/aurapulse/schemas.py).
- **Zero-cost classification.** Runs against a local model served by Ollama, with structured
  output enforced on the wire (not just prompted), retries on schema-invalid output, and a
  code-level normalization fix for a known model quirk — see
  [`src/aurapulse/classifier.py`](src/aurapulse/classifier.py).
- **Ground truth without an LLM.** A fully deterministic fake-review generator provides known
  sentiment/aspect ground truth to test the pipeline before a single real model call — see
  [`src/aurapulse/fake_reviews.py`](src/aurapulse/fake_reviews.py). Real-data evals layer on top
  of that: the free Yelp star-rating proxy for sentiment, and a hand-labeled 100-review sample
  for aspect (which has no free proxy).
- **Per-business aggregation and inconsistency flagging.** Groups classified reviews by
  business, summarizes sentiment distribution and per-aspect negative share, and flags aspects
  whose negative share is disproportionately worse than the business's own baseline — the core
  "food good, wait time bad" signal — see
  [`src/aurapulse/aggregation.py`](src/aurapulse/aggregation.py).
- **Human-readable reporting.** Renders aggregated results as plain text, including a
  dataset-wide summary of how often the `other` aspect escape hatch is used (a signal for
  whether the aspect enum needs a new category) — see
  [`src/aurapulse/reporting.py`](src/aurapulse/reporting.py) and
  [`scripts/generate_report.py`](scripts/generate_report.py).
- **Structured observability.** One JSON trace line per classification call (latency, retry
  count, outcome), no dashboard or paid service required — see
  [`src/aurapulse/classifier.py`](src/aurapulse/classifier.py).
- **Hand-labeling tooling.** Generates a dropdown-backed `.xlsx` sheet for hand-labeling the
  100-review aspect ground-truth sample — see
  [`scripts/build_labeling_sheet.py`](scripts/build_labeling_sheet.py).

**Evals run so far** (see [`docs/DESIGN.md`](docs/DESIGN.md) for the full breakdown and what
didn't work along the way):
- 100% sentiment accuracy against the deterministic fake-review ground truth (8/8).
- 85% (51/60) agreement with the free Yelp star-rating proxy on a stratified real-review sample.
- Aspect extraction validated against 100 hand-labeled reviews: 44% aspect-set exact match, 34%
  aspect+sentiment exact match (up from a 36%/24% baseline before few-shot tuning), 0
  classification failures.

**Explicitly out of scope until Hito 0 closes** (per [`CLAUDE.md`](CLAUDE.md)):
- Response-draft generation and escalation routing (Hito 1/2).
- Any orchestration framework — LangGraph is deliberately not introduced yet; see
  [`docs/DESIGN.md`](docs/DESIGN.md) for when it would be justified over the current plain
  pipeline.

## Why this project exists

Portfolio project demonstrating agent/LLM orchestration and production practices, run in
parallel with a sibling project ("Pre-Show Reels") that covers a purely sequential pipeline.
AuraPulse's angle is conditional routing: when a graph orchestrator earns its complexity versus
a plain sequential pipeline, decided and documented as the project grows rather than assumed
up front.

## Non-negotiable constraints

- **Zero cost.** No paid API keys or cloud services, for anything — including classification.
- **Human in the loop.** Any future "respond to review" feature will only ever produce a draft;
  it will never auto-publish a response.

## Architecture

Hito 0's pipeline, stage by stage (each stage is a separate module — see
[`docs/DESIGN.md`](docs/DESIGN.md) for why this is a plain sequential pipeline rather than a
graph orchestrator at this stage):

```
raw Yelp dataset (data/raw/)
      │  data_loader.py — filter for restaurants, deterministic business/review subset
      ▼
data/processed/*.csv (business_subset.csv, review_subset.csv)
      │  classifier.py — per-review structured classification via local Ollama model
      ▼
ReviewAnalysis records (sentiment, aspects, severity_flag)
      │  aggregation.py — group by business, summarize, flag inconsistencies
      ▼
BusinessReport + OtherAspectSummary
      │  reporting.py — render as human-readable text
      ▼
console / exported report
```

`fake_reviews.py` feeds deterministic, known-ground-truth synthetic reviews into the same
`ReviewAnalysis` shape, so every downstream stage (classifier eval, aggregation, reporting) is
testable without a live Ollama call or real Yelp data.

## Tech stack

- **Python 3.11+**, [Pydantic v2](https://docs.pydantic.dev/) for schema-first structured data.
- [Ollama](https://ollama.com) (local LLM serving, default model `llama3.1:8b`) — the only
  "model" dependency, zero cost, no API key.
- [pandas](https://pandas.pydata.org/) for CSV-based dataset handling.
- [openpyxl](https://openpyxl.readthedocs.io/) for the hand-labeling `.xlsx` sheet.
- [pytest](https://pytest.org/), [ruff](https://docs.astral.sh/ruff/), and
  [mypy](https://mypy-lang.org/) for tests, linting, and static typing — all enforced in CI.

## Folder structure

```
src/aurapulse/       Library code: schemas, classifier, data loader, aggregation, reporting,
                      fake-review generator.
scripts/              Runnable entry points (dataset subset, evals, labeling sheet, report).
tests/                pytest suite, one test module per src/aurapulse module.
data/raw/             Raw Yelp Open Dataset (gitignored — download it yourself, see below).
data/processed/       Derived CSVs/artifacts (gitignored — built by scripts/, not committed).
docs/DESIGN.md         Architectural decisions and their trade-offs, most recent first.
docs/assets/           Static assets (logo).
.github/workflows/     CI (pytest, ruff, mypy on every push/PR).
```

## Installation

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

## Configuration / environment variables

No API keys required anywhere in this project. Copy `.env.example` to `.env` only if you need
to override the defaults below (both already work out of the box for a local `ollama serve`):

| Variable       | Default                 | Purpose                                      |
|----------------|--------------------------|-----------------------------------------------|
| `OLLAMA_HOST`  | `http://localhost:11434` | URL of the local Ollama server.               |
| `OLLAMA_MODEL` | `llama3.1:8b`             | Ollama model tag used for classification.     |

## Usage

Run these in order to go from raw Yelp data to a real aggregated report (each step's output
feeds the next):

```bash
python scripts/build_subset.py              # filter Yelp for restaurants, build the review subset
python scripts/eval_fake_reviews.py         # offline eval against deterministic ground truth
python scripts/validate_sentiment_proxy.py  # validate real reviews against the star-rating proxy
python scripts/build_labeling_sheet.py      # generate the aspect hand-labeling spreadsheet
python scripts/validate_aspect_proxy.py     # validate aspect extraction against hand-labeled ground truth
python scripts/generate_report.py --demo    # aggregated report against the deterministic fake-review dataset
```

`generate_report.py --demo` is runnable today with no dataset and no Ollama server — it's the
quickest way to see the aggregation/reporting format. Running it against real classified Yelp
reviews (`python scripts/generate_report.py --input data/processed/classified_reviews.jsonl`)
needs that JSONL file, which isn't produced by any script yet — see
[Limitations](#limitations).

## Tests and checks

```bash
pytest -q
ruff check src/ tests/ scripts/
mypy src/ scripts/ tests/
```

All three run in CI (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)) on every push
to `main` and on every pull request.

## Roadmap

- **Hito 0 (current):** dataset loading → structured classification → per-business aggregation
  → text reporting. Every stage above is implemented and evaluated individually; the one
  missing piece is a script that runs `classify_review` over the *entire* review subset and
  writes the `classified_reviews.jsonl` file `generate_report.py` expects — see
  [Limitations](#limitations).
- **Hito 1:** response-draft generation for negative reviews without a severity signal, and
  escalation flagging for negative reviews with one — routing written first as plain
  `if`/`elif`, not a graph orchestrator (see [`docs/DESIGN.md`](docs/DESIGN.md)). Drafts only;
  the agent never auto-publishes a response.
- **Hito 2:** revisit whether the `if`/`elif` routing from Hito 1 still stays legible with more
  routes, and only then consider introducing LangGraph — documented explicitly when (if) that
  call is made.

## Limitations

- **No end-to-end script for the real dataset yet.** `scripts/validate_sentiment_proxy.py` and
  `scripts/validate_aspect_proxy.py` classify small stratified *samples* for evaluation
  purposes; nothing yet classifies the full ~500-1000-review subset and writes it to
  `data/processed/classified_reviews.jsonl` for `generate_report.py` to consume. Until then,
  the aggregation/reporting stages are only demonstrable via `--demo` mode (deterministic fake
  reviews) or by pointing `--input` at a manually assembled JSONL file.
- **Local-model classification is imperfect and slow.** ~20-45s/review on CPU inference; aspect
  extraction is at 44%/34% exact-match rates against hand-labeled ground truth (see
  [`docs/DESIGN.md`](docs/DESIGN.md) for the per-aspect precision/recall breakdown). `other`
  aspect precision/recall (33.3%/7.1%) is the weakest category and hasn't been investigated
  further yet.
- **`severity_flag` is not reliable.** Accuracy against the fake-review ground truth sat at 50%
  across two eval runs, with no stable pattern to which reviews it got right — acceptable to
  defer since no Hito 0 logic consumes it, but it needs work before Hito 1's escalation routing
  can depend on it.
- **No temporal-evolution reporting yet.** The Hito 0 scope allows for it "if data supports it"
  (per `CLAUDE.md`), but it isn't implemented in `aggregation.py` yet — only aggregate sentiment
  distribution and per-aspect summaries are.
- **Inconsistency-flagging thresholds are fixed constants**
  (`MIN_MENTIONS_FOR_FLAG = 3`, `INCONSISTENCY_THRESHOLD = 0.30` in
  [`src/aurapulse/aggregation.py`](src/aurapulse/aggregation.py)), not yet tuned against real
  aggregated data — they were chosen for sanity, not validated against how business owners
  would judge "actionable" vs. "noise".

## Design decisions

Every non-trivial architectural call — the aspect schema, the classification backend, ground-
truth conventions, what was tried and reverted and why — is logged with its trade-offs in
[`docs/DESIGN.md`](docs/DESIGN.md).

## License

MIT — see [`LICENSE`](LICENSE).
