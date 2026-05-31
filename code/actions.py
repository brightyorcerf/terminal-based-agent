"""
actions.py — Load internal_tools.json and validate actions_taken arrays.

Responsibilities:
  - Parse the API spec at startup
  - Validate LLM-generated action arrays against the spec schema
  - Inject verify_identity before any destructive action if missing
  - Return safe JSON string (never crashes)
"""

import json
from pathlib import Path

from config import (
    API_SPEC_PATH,
    DESTRUCTIVE_ACTIONS,
    IDENTITY_ACTIONS,
)


# ── Load spec at import time ─────────────────────────────────────────────────

def _load_spec() -> tuple[frozenset[str], dict[str, dict]]:
    """
    Returns (valid_action_names, action_schemas_by_name).
    Gracefully returns empty structures if the file doesn't exist yet —
    this prevents crashes during development before the corpus is present.
    """
    if not API_SPEC_PATH.exists():
        print(
            f"  [WARN] API spec not found at {API_SPEC_PATH}. "
            "Tool validation will be permissive.",
            flush=True,
        )
        return frozenset(), {}

    with open(API_SPEC_PATH, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    # The spec is either a bare list of tool objects OR {"tools": [...]}
    if isinstance(spec, list):
        tools = spec
    else:
        tools = spec.get("tools", [])

    names = frozenset(t["name"] for t in tools if "name" in t)
    schemas = {
        t["name"]: t.get("parameters", {})
        for t in tools
        if "name" in t
    }
    return names, schemas


VALID_ACTIONS, ACTION_SCHEMAS = _load_spec()


# ── Validation ───────────────────────────────────────────────────────────────

def validate_actions(
    raw_actions: list | str | None,
) -> str:
    """
    Validate and sanitize an actions_taken value produced by the LLM.

    Rules:
      1. Must be a JSON array. If not parseable, return "[]".
      2. Each element must be a dict with "action" in VALID_ACTIONS.
         Unknown or malformed entries are silently dropped.
      3. If any destructive action is present and no identity-check action
         precedes it, inject {"action": "verify_identity", "parameters": {}}
         before the first destructive action.
      4. Parameter dicts are kept as-is (deep schema validation is
         aspirational; doing it robustly requires the full JSON Schema spec).

    Returns a compact JSON string, e.g. '[{"action":"verify_identity","parameters":{}}]'
    """
    # ── Parse ────────────────────────────────────────────────────────────
    if raw_actions is None:
        return "[]"

    if isinstance(raw_actions, str):
        raw_actions = raw_actions.strip()
        if not raw_actions or raw_actions == "null":
            return "[]"
        try:
            raw_actions = json.loads(raw_actions)
        except json.JSONDecodeError:
            return "[]"

    if not isinstance(raw_actions, list):
        return "[]"

    # ── Filter valid actions ─────────────────────────────────────────────
    valid: list[dict] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        name = item.get("action", "")
        if not name:
            continue
        # If VALID_ACTIONS is empty (spec missing) accept anything to avoid
        # false negatives; otherwise enforce the allowlist.
        if VALID_ACTIONS and name not in VALID_ACTIONS:
            continue
        params = item.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        valid.append({"action": name, "parameters": params})

    if not valid:
        return "[]"

    # ── Prerequisite injection ───────────────────────────────────────────
    has_identity = any(a["action"] in IDENTITY_ACTIONS for a in valid)

    if not has_identity:
        for i, action in enumerate(valid):
            if action["action"] in DESTRUCTIVE_ACTIONS:
                # Inject verify_identity immediately before first destructive
                valid.insert(i, {"action": "verify_identity", "parameters": {}})
                break

    return json.dumps(valid, separators=(",", ":"))
