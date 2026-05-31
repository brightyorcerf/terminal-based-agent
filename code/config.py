"""
config.py — Global constants, paths, and shared configuration.
All tuneable thresholds live here so they're easy to adjust without
touching the logic files.
"""

import os
from pathlib import Path

# ── Repo layout ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.resolve()
DATA_ROOT = REPO_ROOT / "data"
TICKETS_PATH = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_PATH = REPO_ROOT / "support_tickets" / "output.csv"
API_SPEC_PATH = DATA_ROOT / "api_specs" / "internal_tools.json"

# ── LLM ─────────────────────────────────────────────────────────────────────
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0          # determinism — never change this
LLM_RETRY_WAIT = 60          # seconds to wait on RateLimitError
LLM_RETRY_ATTEMPTS = 2

# ── Retrieval ────────────────────────────────────────────────────────────────
BM25_TOP_K = 5               # chunks returned per query
BM25_CHUNK_SIZE = 400        # characters per chunk
BM25_CHUNK_OVERLAP = 80      # overlap between adjacent chunks
BM25_MIN_CHUNK_LEN = 20      # skip chunks shorter than this
RETRIEVAL_THRESHOLD = 1.5    # below this → low confidence, lean toward escalate

# ── Parallelism ──────────────────────────────────────────────────────────────
MAX_WORKERS = 10             # ThreadPoolExecutor workers

# ── Confidence ranges (for calibration) ─────────────────────────────────────
# Used in the system prompt AND in post-validation overrides
CONF_ADVERSARIAL  = 0.15     # adversarial detected
CONF_PII_ESCALATE = 0.25     # PII-forced escalation
CONF_LOW_RETRIEVAL = 0.35    # corpus didn't match well
CONF_MEDIUM       = 0.55     # reasonable answer, some uncertainty
CONF_HIGH         = 0.82     # strong corpus match, clear answer

# ── Output schema ────────────────────────────────────────────────────────────
# ORDER MATTERS — must match the actual output.csv header exactly.
# If output.csv has more columns, add them here in order.
OUTPUT_COLUMNS = [
    "status",
    "product_area",
    "response",
    "justification",
    "request_type",
    "confidence_score",
    "source_documents",
    "risk_level",
    "pii_detected",
    "language",
    "actions_taken",
]

# ── Enum allowlists ──────────────────────────────────────────────────────────
VALID_STATUSES      = frozenset({"replied", "escalated"})
VALID_REQUEST_TYPES = frozenset({"product_issue", "feature_request", "bug", "invalid"})
VALID_RISK_LEVELS   = frozenset({"low", "medium", "high", "critical"})

# ── Safe fallback row ────────────────────────────────────────────────────────
# Used when something crashes hard enough that even the normal escalation
# path can't run. Never changes — fully deterministic.
FALLBACK_ROW = {
    "status":           "escalated",
    "product_area":     "unknown",
    "response":         "This request has been escalated to our support team for review. A human agent will follow up with you shortly.",
    "justification":    "System error during processing — escalated as precaution.",
    "request_type":     "invalid",
    "confidence_score": 0.2,
    "source_documents": "",
    "risk_level":       "high",
    "pii_detected":     "false",
    "language":         "en",
    "actions_taken":    "[]",
}

# ── Actions requiring identity verification first ────────────────────────────
DESTRUCTIVE_ACTIONS = frozenset({
    "issue_refund",
    "lock_account",
    "unlock_account",
    "delete_account",
    "modify_subscription",
    "reset_credentials",
    "close_ticket",
    "escalate_to_human",
    "chargeback",
    "reverse_transaction",
})

IDENTITY_ACTIONS = frozenset({
    "verify_identity",
    "check_identity_status",
})
