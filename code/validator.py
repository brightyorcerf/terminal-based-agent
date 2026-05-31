"""
validator.py — Post-generation validation and output cleaning.

Runs AFTER the LLM responds, BEFORE writing to output.csv.

Checks:
  1. Citation validation — strip any source_documents path not in corpus manifest
  2. PII-in-response check — escalate if LLM echoed PII back
  3. Enum enforcement — force valid values for status / request_type / risk_level
  4. Confidence override — lower confidence when retrieval was weak
  5. pii_detected canonicalisation — always string "true" / "false"
  6. actions_taken delegation to actions.validate_actions()
  7. actions_taken prerequisite injection (via actions.validate_actions)
  8. language fallback
  9. confidence clamp to [0.05, 0.95]
"""

from config import (
    VALID_STATUSES,
    VALID_REQUEST_TYPES,
    VALID_RISK_LEVELS,
    RETRIEVAL_THRESHOLD,
    CONF_PII_ESCALATE,
    CONF_LOW_RETRIEVAL,
    CONF_ADVERSARIAL,
    FALLBACK_ROW,
)
from safety import scan_pii
from actions import validate_actions


def validate_and_clean(
    llm_output: dict,
    corpus_manifest: frozenset[str],
    real_pii_detected: bool,
    pii_types: list[str],
    best_retrieval_score: float,
) -> dict:
    """
    Validate and sanitise the LLM output dict in-place.

    Args:
        llm_output          : dict parsed from LLM JSON response
        corpus_manifest     : frozenset of real relative paths in data/
        real_pii_detected   : result of our own PII scan (authoritative)
        pii_types           : list of detected PII type labels
        best_retrieval_score: BM25 top score (used to calibrate confidence)

    Returns:
        The mutated dict (same object, returned for convenience).
    """

    # ── 1. Citation validation ───────────────────────────────────────────
    raw_sources: str = str(llm_output.get("source_documents") or "")
    if raw_sources:
        cited = [p.strip() for p in raw_sources.split("|") if p.strip()]
        valid_cited = [p for p in cited if p in corpus_manifest]
        hallucinated = [p for p in cited if p not in corpus_manifest]

        llm_output["source_documents"] = "|".join(valid_cited)

        if hallucinated:
            # Penalise confidence for hallucinated citations
            current = _safe_float(llm_output.get("confidence_score"), 0.5)
            llm_output["confidence_score"] = max(0.1, current - 0.2)
            _append_justification(
                llm_output,
                f"[{len(hallucinated)} hallucinated citation(s) removed]",
            )
    else:
        llm_output["source_documents"] = ""

    # ── 2. PII-in-response check ─────────────────────────────────────────
    response_text: str = str(llm_output.get("response") or "")
    pii_in_response, _ = scan_pii(response_text)
    if pii_in_response:
        llm_output["status"] = "escalated"
        llm_output["response"] = (
            "This request has been escalated to our support team. "
            "A human agent will contact you securely."
        )
        llm_output["risk_level"] = "high"
        llm_output["confidence_score"] = CONF_PII_ESCALATE
        _append_justification(
            llm_output,
            "[Escalated: PII detected in generated response — redacted]",
        )

    # ── 3. Enum enforcement ──────────────────────────────────────────────
    if llm_output.get("status") not in VALID_STATUSES:
        llm_output["status"] = "escalated"

    if llm_output.get("request_type") not in VALID_REQUEST_TYPES:
        llm_output["request_type"] = "invalid"

    if llm_output.get("risk_level") not in VALID_RISK_LEVELS:
        llm_output["risk_level"] = "medium"

    # ── 4. Low retrieval score → cap confidence, lean toward escalation ──
    if best_retrieval_score < RETRIEVAL_THRESHOLD:
        current = _safe_float(llm_output.get("confidence_score"), 0.5)
        # Cap confidence at CONF_LOW_RETRIEVAL when corpus support is weak
        if current > CONF_LOW_RETRIEVAL:
            llm_output["confidence_score"] = CONF_LOW_RETRIEVAL
        # If LLM said "replied" but retrieval was very weak, escalate
        if (
            llm_output.get("status") == "replied"
            and best_retrieval_score < RETRIEVAL_THRESHOLD / 2
        ):
            llm_output["status"] = "escalated"
            _append_justification(
                llm_output,
                "[Escalated: insufficient corpus support for reliable answer]",
            )

    # ── 5. pii_detected canonicalisation ────────────────────────────────
    # Our regex scan is authoritative; the LLM's guess is secondary.
    llm_output["pii_detected"] = "true" if real_pii_detected else "false"

    # ── 6 & 7. actions_taken validation + prerequisite injection ─────────
    llm_output["actions_taken"] = validate_actions(
        llm_output.get("actions_taken", [])
    )

    # ── 8. language fallback ─────────────────────────────────────────────
    if not llm_output.get("language"):
        llm_output["language"] = "en"

    # ── 9. Confidence clamp ──────────────────────────────────────────────
    llm_output["confidence_score"] = round(
        max(0.05, min(0.95, _safe_float(llm_output.get("confidence_score"), 0.3))),
        3,
    )

    # ── 10. Ensure all required keys present (defensive) ─────────────────
    for key, default in FALLBACK_ROW.items():
        if key not in llm_output or llm_output[key] is None:
            llm_output[key] = default

    return llm_output


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _append_justification(row: dict, note: str) -> None:
    existing = str(row.get("justification") or "").strip()
    row["justification"] = f"{existing} {note}".strip() if existing else note
