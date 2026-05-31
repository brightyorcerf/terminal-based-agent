"""
safety.py — Pre-LLM safety gates. Pure Python, zero API calls.

Three concerns:
  1. PII detection  — find sensitive personal data in ticket text
  2. Injection detection — find adversarial / prompt-injection patterns
  3. Language detection — ISO 639-1 code for the primary language

All functions are deterministic and stateless.
"""

import re


# ════════════════════════════════════════════════════════════════════════════
# 1. PII DETECTION
# ════════════════════════════════════════════════════════════════════════════

_PII_PATTERNS: dict[str, re.Pattern] = {
    # ── Payment cards ──────────────────────────────────────────────────────
    # Matches Visa / Mastercard / Amex / Discover with optional spaces/dashes
    "credit_card": re.compile(
        r"\b(?:"
        r"4[0-9]{12}(?:[0-9]{3})?"            # Visa 13 or 16
        r"|5[1-5][0-9]{14}"                    # Mastercard
        r"|3[47][0-9]{13}"                     # Amex
        r"|6(?:011|5[0-9]{2})[0-9]{12}"       # Discover
        r"|(?:\d{4}[\s\-]){3}\d{4}"           # generic spaced 16-digit
        r")\b"
    ),

    # ── US Social Security Number ──────────────────────────────────────────
    "ssn": re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
    ),

    # ── Email address ──────────────────────────────────────────────────────
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),

    # ── US phone (various formats) ─────────────────────────────────────────
    "phone_us": re.compile(
        r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
    ),

    # ── Indian mobile (10 digits, starts 6-9) ─────────────────────────────
    "phone_india": re.compile(
        r"\b[6-9]\d{9}\b"
    ),

    # ── Aadhaar (12 digits, optionally spaced 4-4-4) ──────────────────────
    "aadhaar": re.compile(
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    ),

    # ── Street address heuristic ───────────────────────────────────────────
    "address": re.compile(
        r"\b\d{1,5}\s+[A-Za-z0-9\s]{3,40}"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd"
        r"|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b",
        re.IGNORECASE,
    ),

    # ── Passport number heuristic ─────────────────────────────────────────
    "passport": re.compile(
        r"\b[A-Z]{1,2}[0-9]{6,9}\b"
    ),
}

# PII types that are HIGH risk and warrant automatic escalation
_HIGH_RISK_PII = frozenset({"credit_card", "ssn", "aadhaar", "passport"})


def scan_pii(text: str) -> tuple[bool, list[str]]:
    """
    Scan text for PII.

    Returns:
        (pii_found, list_of_pii_type_names)

    NOTE: never returns the actual matched values — we never log PII.
    """
    found: list[str] = []
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            found.append(pii_type)
    return (bool(found), found)


def is_high_risk_pii(pii_types: list[str]) -> bool:
    """True if any detected PII type warrants automatic escalation."""
    return bool(_HIGH_RISK_PII.intersection(set(pii_types)))


# ════════════════════════════════════════════════════════════════════════════
# 2. INJECTION / ADVERSARIAL DETECTION
# ════════════════════════════════════════════════════════════════════════════

_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ── Direct instruction override ────────────────────────────────────────
    (re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+"
        r"(instructions?|prompts?|rules?|context|constraints?)",
        re.IGNORECASE,
    ), "instruction_override"),

    (re.compile(
        r"disregard\s+(all\s+)?(previous|prior|above|earlier)",
        re.IGNORECASE,
    ), "instruction_override"),

    (re.compile(
        r"forget\s+(all\s+)?(previous|prior|your)\s+"
        r"(instructions?|context|training|rules?)",
        re.IGNORECASE,
    ), "instruction_override"),

    (re.compile(
        r"override\s+(your\s+)?(instructions?|rules?|guidelines?|constraints?)",
        re.IGNORECASE,
    ), "instruction_override"),

    # ── Role / persona hijacking ───────────────────────────────────────────
    (re.compile(
        r"you\s+are\s+now\s+(a\s+)?(?:different|new|another|evil|uncensored)",
        re.IGNORECASE,
    ), "persona_hijack"),

    (re.compile(
        r"\bDAN\b",   # "Do Anything Now" jailbreak keyword
    ), "persona_hijack"),

    (re.compile(
        r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(?:a\s+)?(?:different|uncensored|unrestricted|jailbroken)",
        re.IGNORECASE,
    ), "persona_hijack"),

    (re.compile(
        r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(?:AI|assistant|bot|system)\s+"
        r"(?:without|that\s+ignores|that\s+has\s+no)",
        re.IGNORECASE,
    ), "persona_hijack"),

    # ── System prompt / internals extraction ──────────────────────────────
    (re.compile(
        r"(print|output|reveal|show|display|repeat|tell\s+me|what\s+(is|are))"
        r"\s+(your\s+)?(system\s+prompt|instructions?|rules?|corpus|training\s+data)",
        re.IGNORECASE,
    ), "exfiltration"),

    (re.compile(
        r"what\s+were\s+you\s+told|what\s+are\s+your\s+(instructions?|rules?|guidelines?)",
        re.IGNORECASE,
    ), "exfiltration"),

    (re.compile(
        r"(list|dump|output|print)\s+(all\s+)?(files?|documents?|corpus|tickets?|training)",
        re.IGNORECASE,
    ), "exfiltration"),

    (re.compile(
        r"(send|email|forward|transmit|exfiltrate)\s+(all\s+)?(data|tickets?|corpus|documents?)\s+to",
        re.IGNORECASE,
    ), "exfiltration"),

    # ── Direct output manipulation ─────────────────────────────────────────
    (re.compile(
        r"(output|respond\s+with|return|print|write)\s+(only|just)?\s*"
        r"[\"']?status\s*:\s*(replied|escalated)",
        re.IGNORECASE,
    ), "output_manipulation"),

    (re.compile(
        r"classify\s+this\s+(ticket\s+)?as\s+(replied|escalated|invalid|a\s+bug)",
        re.IGNORECASE,
    ), "output_manipulation"),

    (re.compile(
        r"set\s+(the\s+)?(status|risk_level|confidence_score|request_type)\s+(to|=)",
        re.IGNORECASE,
    ), "output_manipulation"),

    (re.compile(
        r"your\s+(final\s+)?answer\s+(should\s+be|must\s+be|is)\s+(replied|escalated)",
        re.IGNORECASE,
    ), "output_manipulation"),

    # ── Authority / privilege claim ────────────────────────────────────────
    (re.compile(
        r"(i\s+am|i'm)\s+(your\s+)?(developer|admin|administrator|creator|"
        r"anthropic|openai|supervisor|operator|owner)",
        re.IGNORECASE,
    ), "false_authority"),

    (re.compile(
        r"(authorized|permission|allowed)\s+to\s+(bypass|override|ignore|skip)",
        re.IGNORECASE,
    ), "false_authority"),

    (re.compile(
        r"(maintenance|test|debug)\s+mode",
        re.IGNORECASE,
    ), "false_authority"),

    # ── Multilingual injections (common attacks in other languages) ────────
    (re.compile(
        r"ignorez\s+(les?\s+)?instructions",
        re.IGNORECASE,
    ), "multilingual_injection"),

    (re.compile(
        r"ignorar\s+(las?\s+)?instrucciones",
        re.IGNORECASE,
    ), "multilingual_injection"),

    (re.compile(
        r"ignoriere\s+(alle?\s+)?anweisungen",
        re.IGNORECASE,
    ), "multilingual_injection"),

    (re.compile(
        r"ignorer\s+les\s+instructions",
        re.IGNORECASE,
    ), "multilingual_injection"),

    # ── Delimiter / structural injection ──────────────────────────────────
    (re.compile(
        r"<\s*/?\s*(system|instruction|prompt|ticket_data)\s*>",
        re.IGNORECASE,
    ), "delimiter_injection"),

    (re.compile(
        r"\]\s*\]\s*\]\s*ignore",
        re.IGNORECASE,
    ), "delimiter_injection"),

    # ── Jailbreak catch-alls ───────────────────────────────────────────────
    (re.compile(
        r"jailbreak|jail\s*break",
        re.IGNORECASE,
    ), "jailbreak"),

    (re.compile(
        r"developer\s+mode\s+enabled|enable\s+developer\s+mode",
        re.IGNORECASE,
    ), "jailbreak"),
]


def scan_injection(text: str) -> tuple[bool, str]:
    """
    Scan text for adversarial prompt injection patterns.

    Returns:
        (found: bool, category: str)

    category is one of the string labels in _INJECTION_PATTERNS,
    or "" if nothing found.
    """
    for pattern, category in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True, category
    return False, ""


# ════════════════════════════════════════════════════════════════════════════
# 3. LANGUAGE DETECTION
# ════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Returns an ISO 639-1 language code.
    Falls back to 'en' if detection fails or text is too short.

    Uses langdetect with seed=42 for determinism.
    """
    if len(text.strip()) < 20:
        return "en"

    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42          # determinism — required
        code = detect(text[:500])          # only first 500 chars for speed
        return code if code else "en"
    except Exception:
        # Heuristic fallback: high non-ASCII ratio → mark as unknown
        non_ascii = sum(1 for c in text if ord(c) > 127)
        ratio = non_ascii / max(len(text), 1)
        return "unknown" if ratio > 0.4 else "en"


# ════════════════════════════════════════════════════════════════════════════
# 4. MISC HELPERS
# ════════════════════════════════════════════════════════════════════════════

def is_gibberish(text: str) -> bool:
    """
    True if text is too short or contains too many non-printable characters
    to be a meaningful support ticket.
    """
    stripped = text.strip()
    if len(stripped) < 3:
        return True
    printable_ratio = sum(1 for c in stripped if c.isprintable()) / len(stripped)
    return printable_ratio < 0.7
