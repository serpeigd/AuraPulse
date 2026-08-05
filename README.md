<p align="center">
  <img src="docs/assets/logo.svg" alt="AuraPulse logo" width="120" height="120">
</p>

# AuraPulse

Detects recurring operational inconsistencies in restaurant reviews — e.g. food praised
consistently while wait time is consistently criticized — and turns that into actionable
product-improvement signal for the business. Built entirely on free, local tooling: **no paid
API keys anywhere**, sentiment/aspect classification runs on a local LLM via
[Ollama](https://ollama.com).

## Status: 🚧 Work in progress (Hito 0)

This is a portfolio project, developed incrementally with documented design decisions (see
[`docs/DESIGN.md`](docs/DESIGN.md)). Current milestone (Hito 0) target: a working pipeline from
raw Yelp reviews to an aggregated reputation/inconsistency report. Not complete yet.

**Done:**
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

**Not done yet (explicitly out of scope until Hito 0 closes):**
- Aggregation and reporting across a business's reviews
- Response-draft generation and escalation routing (Hito 1/2)
- Any orchestration framework — LangGraph is deliberately not introduced yet; see
  `docs/DESIGN.md` for when it would be justified over a plain pipeline

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

## Usage

```bash
python scripts/build_subset.py              # filter Yelp for restaurants, build the review subset
python scripts/eval_fake_reviews.py         # offline eval against deterministic ground truth
python scripts/validate_sentiment_proxy.py  # validate real reviews against the star-rating proxy
python scripts/build_labeling_sheet.py      # generate the aspect hand-labeling spreadsheet
python scripts/validate_aspect_proxy.py     # validate aspect extraction against hand-labeled ground truth
```

## Tests and checks

```bash
pytest -q
ruff check src/ tests/ scripts/
mypy src/ scripts/ tests/
```

## Design decisions

Every non-trivial architectural call — the aspect schema, the classification backend, ground-
truth conventions, what was tried and reverted and why — is logged with its trade-offs in
[`docs/DESIGN.md`](docs/DESIGN.md).

## License

MIT — see [`LICENSE`](LICENSE).
