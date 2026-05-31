# Support Triage Agent — Setup & Run

## Prerequisites

- Python 3.11+
- An OpenAI API key (`gpt-4o`)

## Installation

```bash
# From repo root

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r code/requirements.txt
```

## Configuration

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your key:
#   OPENAI_API_KEY=sk-...
```

The agent reads `OPENAI_API_KEY` from the environment automatically.
**Never hardcode keys.**

## Running the agent

```bash
# From repo root
python code/main.py
```

Output is written to `support_tickets/output.csv`.

## Validating output format

```bash
python code/validate_output.py
```

This checks structural compliance (columns, enums, row count). It does **not**
evaluate quality.

## Reproducing results exactly

The agent is fully deterministic:

- `temperature=0`, `seed=42` on all OpenAI calls
- SHA-256 keyed persistent LLM response cache (`code/llm_cache.json`) — same prompt always returns the same cached response
- BM25 corpus is sorted before indexing (stable chunk IDs)
- `langdetect.DetectorFactory.seed = 42`
- No random sampling anywhere in the pipeline
- `ThreadPoolExecutor` results collected by index (not completion order)

Running the agent twice on the same input produces byte-identical output.

## Architecture

See `code/ARCHITECTURE.md` for the full design walkthrough.

## File layout

```
code/
├── main.py           # Entry point — run this
├── config.py         # All constants, thresholds, paths
├── retriever.py      # Corpus manifest + BM25 index
├── safety.py         # PII detection, injection detection, language ID
├── llm.py            # System prompt, user prompt, Anthropic API wrapper
├── validator.py      # Post-generation manifest check + output cleaning
├── actions.py        # API spec loader + actions_taken validator
├── requirements.txt  # Pinned dependencies
├── README.md         # This file
└── ARCHITECTURE.md   # Design documentation
```
