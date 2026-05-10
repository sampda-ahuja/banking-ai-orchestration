"""
app/policies/pii_rules.py
Deterministic PII detection via regex patterns.
No LLM involved — fast, reliable, zero-hallucination.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ─── Compiled patterns ───────────────────────────────────────────────────────

_PATTERNS: dict[str, re.Pattern[str]] = {
    # US Social Security Number  e.g. 123-45-6789 or 123456789
    "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    # Indian Aadhaar (12-digit)
    "aadhaar": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    # Indian PAN  e.g. ABCDE1234F
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    # Bank account number (8–17 digits)
    "bank_account": re.compile(r"\b\d{8,17}\b"),
    # US ABA routing number (9 digits)
    "routing_number": re.compile(r"\b\d{9}\b"),
    # Credit/debit card number: 16 consecutive digits OR groups of 4 separated by spaces/dashes
    # Uses a specific pattern to avoid false-positives with Aadhaar (12 digits)
    "card_number": re.compile(
        r"\b(?:\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}(?:[ -]\d{1,3})?|\d{16,19})\b"
    ),
    # CVV (3–4 digits preceded by keyword like 'CVV is', 'CVV:', 'CVV2')
    "cvv": re.compile(r"\b(?:cvv2?|cvc|security\s+code)\s*(?:is\s*|[:\s])*\d{3,4}\b", re.IGNORECASE),
    # Password patterns
    "password": re.compile(r"\b(?:password|passwd|pwd)[:\s]+\S+", re.IGNORECASE),
}

# Labels to display / log for each pattern key
_LABEL_MAP: dict[str, str] = {
    "ssn": "SSN",
    "aadhaar": "Aadhaar",
    "pan": "PAN",
    "bank_account": "Bank Account",
    "routing_number": "Routing Number",
    "card_number": "Card Number",
    "cvv": "CVV",
    "password": "Password",
}


@dataclass
class PiiScanResult:
    """Result of a deterministic PII scan."""

    has_pii: bool = False
    detected_types: list[str] = field(default_factory=list)
    redacted_text: str = ""


def scan_for_pii(text: str) -> PiiScanResult:
    """
    Scan *text* for known PII patterns.

    Partial masking is applied (first/last chars preserved) so that
    redacted text remains useful in logs without exposing raw PII.
    """
    detected: list[str] = []
    redacted = text

    # Skip already-masked values like ****1234, XXXX-1234, etc.
    _masked_pattern = re.compile(r"[\*xX]{2,}[-\s]?\d{1,4}")

    for key, pattern in _PATTERNS.items():
        matches = pattern.findall(redacted)
        if matches:
            # Filter out matches that are part of already-masked values
            real_matches = [
                m for m in matches
                if not _masked_pattern.search(text[max(0, text.find(m) - 6):text.find(m) + len(m)])
            ]
            if real_matches:
                detected.append(_LABEL_MAP[key])
                # Replace each match with a masked version
                redacted = pattern.sub(lambda m: _mask(m.group()), redacted)

    return PiiScanResult(
        has_pii=bool(detected),
        detected_types=detected,
        redacted_text=redacted,
    )


def _mask(value: str) -> str:
    """
    Partially mask a detected PII string.
    Keeps first 2 and last 2 chars; replaces the rest with '*'.
    Single-char values are fully masked.
    """
    clean = value.strip()
    if len(clean) <= 4:
        return "*" * len(clean)
    return clean[:2] + "*" * (len(clean) - 4) + clean[-2:]
