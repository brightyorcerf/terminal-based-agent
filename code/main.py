"""
main.py — Entry point for the MLE Hiring Challenge support triage agent.

Usage:
    python code/main.py

Reads:  support_tickets/support_tickets.csv
Writes: support_tickets/output.csv

Architecture (5 stages per ticket):
  Stage 0 — Parse ticket (JSON conversation array, subject, company)
  Stage 1 — Pre-screen (PII scan, injection scan, gibberish check) — pure Python
  Stage 2 — BM25 Retrieval — pure Python, no API calls
  Stage 3 — LLM Generation — XML-bounded prompt, temperature=0
  Stage 4 — Post-validation — manifest check, PII-in-output, enum guard
  Stage 5 — Write row to output

Parallelism: ThreadPoolExecutor with MAX_WORKERS workers.
No-crash guarantee: every ticket is wrapped in a try/except that returns
a hardcoded escalation row on any unhandled exception.
"""

import json
import os
import sys
import traceback
import concurrent.futures

# Load .env before any module that reads env vars (anthropic client reads ANTHROPIC_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; user can export ANTHROPIC_API_KEY directly

import pandas as pd

from config import (
    TICKETS_PATH,
    OUTPUT_PATH,
    OUTPUT_COLUMNS,
    FALLBACK_ROW,
    CONF_ADVERSARIAL,
    CONF_PII_ESCALATE,
    MAX_WORKERS,
)
from retriever import Retriever
from safety import (
    scan_pii,
    is_high_risk_pii,
    scan_injection,
    detect_language,
    is_gibberish,
)
from llm import call_llm
from validator import validate_and_clean


# ── Globals (initialised once, shared across threads) ────────────────────────
# The Retriever object is read-only after __init__ so thread-safe.
_retriever: Retriever | None = None


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _make_escalation_row(
    reason: str,
    *,
    pii_detected: bool = False,
    language: str = "en",
    risk_level: str = "high",
    confidence: float | None = None,
) -> dict:
    """
    Produce a deterministic escalation row without calling the LLM.
    Used by the pre-screen gate and hard-error fallbacks.
    """
    if confidence is None:
        confidence = CONF_PII_ESCALATE if pii_detected else CONF_ADVERSARIAL
    row = dict(FALLBACK_ROW)
    row.update(
        {
            "justification": f"Escalated: {reason}",
            "pii_detected": "true" if pii_detected else "false",
            "language": language,
            "risk_level": risk_level,
            "confidence_score": round(confidence, 3),
        }
    )
    return row


def _parse_issue(raw: str) -> tuple[str, str]:
    """
    Parse the issue column (JSON array of conversation turns).

    Returns:
        (last_user_turn, full_conversation_text)

    Falls back gracefully to raw string if JSON parse fails.
    """
    try:
        turns = json.loads(raw)
        if not isinstance(turns, list):
            raise ValueError("Not a list")

        lines = []
        last_user = ""
        for turn in turns:
            role = str(turn.get("role", "user")).upper()
            content = str(turn.get("content", ""))
            lines.append(f"{role}: {content}")
            if turn.get("role") == "user":
                last_user = content

        full = "\n".join(lines)
        return (last_user or full), full

    except Exception:
        return raw, raw


# ════════════════════════════════════════════════════════════════════════════
# SINGLE TICKET PROCESSOR
# ════════════════════════════════════════════════════════════════════════════

def _process_ticket(row: "pd.Series") -> dict:
    """
    Full 5-stage pipeline for one ticket row.

    NEVER raises — all exceptions are caught and return FALLBACK_ROW.
    """
    try:
        # ── Stage 0: Parse ───────────────────────────────────────────────
        raw_issue = str(row.get("issue") or "")
        subject   = str(row.get("subject") or "")
        company   = str(row.get("company") or "")

        last_user_turn, full_conversation = _parse_issue(raw_issue)

        # ── Language detection (fast, no API) ────────────────────────────
        language = detect_language(full_conversation)

        # ── PII scan on full conversation ─────────────────────────────────
        pii_found, pii_types = scan_pii(full_conversation)

        # ── Stage 1: Pre-screen ──────────────────────────────────────────

        # 1a. Gibberish check
        if is_gibberish(full_conversation):
            return _make_escalation_row(
                "Empty or non-printable ticket content",
                language=language,
                risk_level="low",
                confidence=0.3,
            )

        # 1b. Injection check (highest priority)
        injection_found, injection_category = scan_injection(full_conversation)
        if injection_found:
            return _make_escalation_row(
                f"Adversarial pattern detected ({injection_category})",
                language=language,
                risk_level="critical",
                confidence=CONF_ADVERSARIAL,
            )

        # 1c. High-risk PII → auto-escalate
        if pii_found and is_high_risk_pii(pii_types):
            return _make_escalation_row(
                f"High-risk PII detected ({', '.join(pii_types)}) — escalated for safe handling",
                pii_detected=True,
                language=language,
                risk_level="high",
                confidence=CONF_PII_ESCALATE,
            )

        # ── Stage 2: Retrieval ───────────────────────────────────────────
        # Use last user turn + subject as query.
        # Do NOT gate retrieval on the company field — it may be wrong.
        query = f"{last_user_turn} {subject}"[:800]
        retrieved_chunks, best_score = _retriever.retrieve(query)

        # ── Stage 3: LLM Generation ──────────────────────────────────────
        llm_output = call_llm(
            full_conversation=full_conversation,
            subject=subject,
            company=company,
            retrieved_chunks=retrieved_chunks,
            pii_types=pii_types,
            language=language,
        )

        if llm_output is None:
            return _make_escalation_row(
                "LLM processing error — escalated as precaution",
                pii_detected=pii_found,
                language=language,
                risk_level="medium",
                confidence=0.2,
            )

        # ── Stage 4: Post-validation ─────────────────────────────────────
        final = validate_and_clean(
            llm_output=llm_output,
            corpus_manifest=_retriever.manifest,
            real_pii_detected=pii_found,
            pii_types=pii_types,
            best_retrieval_score=best_score,
        )

        return final

    except Exception:
        tb = traceback.format_exc()
        print(f"  [CRITICAL] Unhandled exception in ticket processor:\n{tb}", flush=True)
        return dict(FALLBACK_ROW)


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def run() -> None:
    global _retriever

    print("=" * 60, flush=True)
    print("MLE Hiring Challenge — Support Triage Agent", flush=True)
    print("=" * 60, flush=True)

    # ── Startup: build index ─────────────────────────────────────────────
    print("\n[1/4] Initialising retriever …", flush=True)
    _retriever = Retriever()

    # ── Load tickets ─────────────────────────────────────────────────────
    print("\n[2/4] Loading tickets …", flush=True)
    if not TICKETS_PATH.exists():
        print(f"  [ERROR] Tickets file not found: {TICKETS_PATH}", flush=True)
        sys.exit(1)

    df = pd.read_csv(TICKETS_PATH)
    df.columns = [c.lower() for c in df.columns]   # normalise Title/UPPER case headers
    total = len(df)
    print(f"  {total} tickets loaded.", flush=True)

    # ── Determine output column order ────────────────────────────────────
    # Always use OUTPUT_COLUMNS order from config as the authoritative source.
    # If output.csv exists but has extra/different columns, we still emit only
    # OUTPUT_COLUMNS in their defined order — never infer order from the file.
    output_cols = OUTPUT_COLUMNS

    # ── Process tickets in parallel ──────────────────────────────────────
    print(f"\n[3/4] Processing {total} tickets (workers={MAX_WORKERS}) …", flush=True)

    results: list[dict | None] = [None] * total

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(_process_ticket, row): idx
            for idx, row in df.iterrows()
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = dict(FALLBACK_ROW)
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"  Progress: {completed}/{total}", flush=True)

    # ── Build output DataFrame ───────────────────────────────────────────
    print("\n[4/4] Writing output …", flush=True)
    output_df = pd.DataFrame(results)

    # Ensure all required columns are present with defaults
    for col in output_cols:
        if col not in output_df.columns:
            output_df[col] = FALLBACK_ROW.get(col, "")

    # Write in correct column order
    output_df[output_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"\n  Done. Output written to: {OUTPUT_PATH}", flush=True)
    print(f"  Rows written: {len(output_df)}", flush=True)

    # Quick sanity stats
    if "status" in output_df.columns:
        counts = output_df["status"].value_counts().to_dict()
        print(f"  Status distribution: {counts}", flush=True)
    if "risk_level" in output_df.columns:
        risk_counts = output_df["risk_level"].value_counts().to_dict()
        print(f"  Risk distribution:   {risk_counts}", flush=True)


if __name__ == "__main__":
    run()
