"""
app/policies/compliance_rules.py
Deterministic compliance rule-checks — no LLM required.
Covers ECOA, UDAAP, and disclaimer enforcement.
"""
from __future__ import annotations

import re

# ─── Guaranteed-approval phrases ─────────────────────────────────────────────

_GUARANTEE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bguaranteed?\s+approv(?:al|ed)\b", re.IGNORECASE),
    re.compile(r"\bdefinitely\s+qualif(?:y|ied)\b", re.IGNORECASE),
    re.compile(r"\byou\s+will\s+(?:definitely\s+)?(?:be\s+)?approv(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:credit\s+check|rejection)\s+guaranteed\b", re.IGNORECASE),
    re.compile(r"\b100\s*%\s+approv(?:al|ed)\b", re.IGNORECASE),
    re.compile(r"\binstant\s+approv(?:al|ed)\b", re.IGNORECASE),
]

# ─── ECOA-protected class references ─────────────────────────────────────────

_PROTECTED_CLASS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:race|racial|ethnicity|ethnic)\b", re.IGNORECASE),
    re.compile(r"\b(?:gender|sex|female|male|woman|man)\b", re.IGNORECASE),
    re.compile(r"\b(?:age|elderly|young|minor)\b", re.IGNORECASE),
    re.compile(r"\b(?:marital\s+status|married|single|divorced|widowed)\b", re.IGNORECASE),
    re.compile(r"\b(?:religion|religious|faith|church|mosque|temple)\b", re.IGNORECASE),
    re.compile(r"\b(?:disability|disabled|handicap(?:ped)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:national\s+origin|nationality|immigrant|citizen(?:ship)?)\b", re.IGNORECASE),
]

# ─── UDAAP / deceptive language patterns ─────────────────────────────────────

_UDAAP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\blowest\s+rate\s+(?:in\s+(?:the\s+)?(?:market|country|state))\b", re.IGNORECASE),
    re.compile(r"\bbest\s+deal\s+(?:available|guaranteed)\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:fees?|hidden\s+fees?|costs?)\s+(?:ever|at\s+all)\b", re.IGNORECASE),
]

# ─── Required disclaimer ──────────────────────────────────────────────────────

REQUIRED_DISCLAIMER = (
    "All loan offers and estimates are subject to formal credit review and approval"
)


def check_guarantee_claims(text: str) -> list[str]:
    """Return list of violation strings for guarantee/approval claims."""
    return [
        f"Guaranteed approval language detected: '...{m.group()}...'"
        for p in _GUARANTEE_PATTERNS
        for m in p.finditer(text)
    ]


def check_ecoa_violations(text: str) -> list[str]:
    """Return list of ECOA violations (protected-class references)."""
    violations: list[str] = []
    for p in _PROTECTED_CLASS_PATTERNS:
        m = p.search(text)
        if m:
            violations.append(f"ECOA violation — protected class reference: '{m.group()}'")
    return violations


def check_udaap_violations(text: str) -> list[str]:
    """Return list of UDAAP violations (deceptive/misleading claims)."""
    return [
        f"UDAAP violation — deceptive language detected: '...{m.group()}...'"
        for p in _UDAAP_PATTERNS
        for m in p.finditer(text)
    ]


def check_disclaimer_present(text: str) -> list[str]:
    """Return a violation if the required disclaimer is missing."""
    if REQUIRED_DISCLAIMER.lower() not in text.lower():
        return [
            "Missing required disclaimer: "
            "'All loan offers and estimates are subject to formal credit review and approval.'"
        ]
    return []
