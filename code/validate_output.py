"""
validate_output.py — Structural compliance check for output.csv.

Checks:
  1. File exists and is readable
  2. All required columns are present (in correct order)
  3. Row count matches support_tickets.csv (input)
  4. status values are all valid
  5. request_type values are all valid
  6. risk_level values are all valid
  7. pii_detected values are all "true" or "false" (lowercase strings)
  8. confidence_score is numeric and in [0.0, 1.0]
  9. actions_taken is valid JSON array per row
 10. source_documents — no hallucinated file paths

This script checks structural compliance ONLY. It does NOT evaluate
correctness, quality, or scoring dimensions.

Exit codes:
  0 — all checks pass
  1 — one or more checks FAIL
"""

import json
import os
import sys

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    sys.exit(1)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "support_tickets", "output.csv")
TICKETS_PATH = os.path.join(REPO_ROOT, "support_tickets", "support_tickets.csv")
DATA_ROOT = os.path.join(REPO_ROOT, "data")

REQUIRED_COLUMNS = [
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

VALID_STATUSES = {"replied", "escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
VALID_RISK_LEVELS = {"low", "medium", "high", "critical"}


def _load_corpus_paths() -> set[str]:
    """Walk data/ and collect all real relative paths."""
    paths: set[str] = set()
    for root, dirs, files in os.walk(DATA_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for fname in files:
            if fname.startswith("."):
                continue
            abs_path = os.path.join(root, fname)
            rel = os.path.relpath(abs_path, REPO_ROOT).replace("\\", "/")
            paths.add(rel)
    return paths


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return ok


def main() -> int:
    print("=" * 60)
    print("output.csv structural compliance check")
    print("=" * 60)
    failures = 0

    # ── 1. File exists ───────────────────────────────────────────────────────
    if not check("output.csv exists", os.path.exists(OUTPUT_PATH)):
        print("\nFATAL: output.csv not found. Cannot continue.")
        return 1

    try:
        df = pd.read_csv(OUTPUT_PATH)
    except Exception as exc:
        print(f"FATAL: Cannot read output.csv: {exc}")
        return 1

    # ── 2. Required columns ──────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if not check(
        "All required columns present",
        not missing,
        f"Missing: {missing}" if missing else "",
    ):
        failures += 1
    if extra:
        print(f"  [WARN] Extra columns (will be ignored by evaluator): {extra}")

    # Check column order
    actual_order = [c for c in df.columns if c in REQUIRED_COLUMNS]
    order_ok = actual_order == REQUIRED_COLUMNS
    if not check(
        "Column order matches OUTPUT_COLUMNS",
        order_ok,
        f"Got: {actual_order}" if not order_ok else "",
    ):
        failures += 1

    # ── 3. Row count ─────────────────────────────────────────────────────────
    if os.path.exists(TICKETS_PATH):
        input_df = pd.read_csv(TICKETS_PATH)
        expected_rows = len(input_df)
        actual_rows = len(df)
        if not check(
            f"Row count matches input ({expected_rows} expected)",
            actual_rows == expected_rows,
            f"Got {actual_rows} rows" if actual_rows != expected_rows else "",
        ):
            failures += 1
    else:
        print("  [WARN] support_tickets.csv not found — skipping row count check")

    if len(df) == 0:
        print("\n  [WARN] output.csv is empty — run main.py first to generate outputs")
        print("=" * 60)
        print(f"RESULT: {failures} failure(s)")
        return 1 if failures else 0

    # ── 4. status values ─────────────────────────────────────────────────────
    bad_status = df[~df["status"].isin(VALID_STATUSES)]
    if not check(
        "status values valid",
        len(bad_status) == 0,
        f"{len(bad_status)} invalid rows: {bad_status['status'].value_counts().to_dict()}" if len(bad_status) else "",
    ):
        failures += 1

    # ── 5. request_type values ───────────────────────────────────────────────
    bad_rt = df[~df["request_type"].isin(VALID_REQUEST_TYPES)]
    if not check(
        "request_type values valid",
        len(bad_rt) == 0,
        f"{len(bad_rt)} invalid rows" if len(bad_rt) else "",
    ):
        failures += 1

    # ── 6. risk_level values ─────────────────────────────────────────────────
    bad_rl = df[~df["risk_level"].isin(VALID_RISK_LEVELS)]
    if not check(
        "risk_level values valid",
        len(bad_rl) == 0,
        f"{len(bad_rl)} invalid rows" if len(bad_rl) else "",
    ):
        failures += 1

    # ── 7. pii_detected values ───────────────────────────────────────────────
    bad_pii = df[~df["pii_detected"].isin(["true", "false"])]
    if not check(
        'pii_detected is always "true" or "false" (lowercase string)',
        len(bad_pii) == 0,
        f"{len(bad_pii)} invalid rows: {bad_pii['pii_detected'].value_counts().to_dict()}" if len(bad_pii) else "",
    ):
        failures += 1

    # ── 8. confidence_score numeric in [0, 1] ────────────────────────────────
    try:
        scores = pd.to_numeric(df["confidence_score"], errors="coerce")
        out_of_range = df[(scores < 0) | (scores > 1) | scores.isna()]
        if not check(
            "confidence_score numeric in [0.0, 1.0]",
            len(out_of_range) == 0,
            f"{len(out_of_range)} invalid rows" if len(out_of_range) else "",
        ):
            failures += 1

        unique_scores = scores.nunique()
        if unique_scores <= 3:
            print(f"  [WARN] Only {unique_scores} unique confidence_score values — calibration will score near zero")
    except Exception as exc:
        print(f"  [FAIL] confidence_score check error: {exc}")
        failures += 1

    # ── 9. actions_taken valid JSON array ────────────────────────────────────
    bad_actions = []
    for i, val in enumerate(df["actions_taken"]):
        try:
            parsed = json.loads(str(val))
            if not isinstance(parsed, list):
                bad_actions.append(i)
        except (json.JSONDecodeError, TypeError):
            bad_actions.append(i)
    if not check(
        "actions_taken is valid JSON array on all rows",
        not bad_actions,
        f"Invalid on rows: {bad_actions[:10]}" if bad_actions else "",
    ):
        failures += 1

    # ── 10. source_documents — no hallucinated paths ─────────────────────────
    corpus = _load_corpus_paths()
    hallucinated: list[tuple[int, str]] = []
    for i, val in enumerate(df["source_documents"]):
        sources = str(val or "").strip()
        if not sources:
            continue
        for path in sources.split("|"):
            path = path.strip()
            if path and path not in corpus:
                hallucinated.append((i, path))

    if not check(
        "No hallucinated source_documents citations",
        not hallucinated,
        f"{len(hallucinated)} hallucinated path(s). First 5: {hallucinated[:5]}" if hallucinated else "",
    ):
        failures += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    if failures == 0:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print(f"RESULT: {failures} FAILURE(S) — fix before submitting")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
