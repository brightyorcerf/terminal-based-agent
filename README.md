# Support Triage Agent — Setup & Run

## Prerequisites

- Python 3.11+
- An Anthropic API key

## Installation

```bash
# From repo root
cd MLE-hiring

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
#   ANTHROPIC_API_KEY=sk-ant-...
```

The agent reads `ANTHROPIC_API_KEY` from the environment automatically.
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

The agent is deterministic:

- `temperature=0` on all LLM calls
- BM25 corpus is sorted before indexing (stable chunk IDs)
- `langdetect.DetectorFactory.seed = 42`
- No random sampling anywhere in the pipeline

Running the agent twice on the same input will produce byte-identical output.

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
