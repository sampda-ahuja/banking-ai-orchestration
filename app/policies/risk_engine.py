"""
app/policies/risk_engine.py
Deterministic underwriting risk engine — zero LLM involvement.
Produces a RiskAssessment based on structured inquiry data.
"""
from __future__ import annotations

from app.schemas.models import InquiryExtraction, RiskAssessment

# ─── Thresholds ───────────────────────────────────────────────────────────────

# If loan-to-income ratio exceeds this, escalate risk
_HIGH_DTI_RATIO = 5.0   # loan amount > 5× annual income → HIGH risk
_MED_DTI_RATIO  = 3.0   # loan amount > 3× annual income → MEDIUM risk

# Stable full-time employment keywords
_STABLE_EMPLOYMENT = {
    "full-time",
    "full time",
    "employed full time",
    "salaried",
    "permanent",
    "government",
    "federal",
    "stable job",
    "stable",
    "stable employment",
}

# Part-time / uncertain employment keywords
_UNSTABLE_EMPLOYMENT = {
    "part-time",
    "part time",
    "freelance",
    "contractor",
    "contract",
    "self-employed",
    "self employed",
    "gig",
    "seasonal",
    "temporary",
}

# Unemployed keywords (highest risk)
_UNEMPLOYED = {
    "unemployed",
    "not employed",
    "no job",
    "job search",
    "looking for work",
    "between jobs",
}


def _classify_employment(status: str | None) -> str:
    """Classify raw employment string into 'stable' | 'unstable' | 'unemployed' | 'unknown'."""
    if status is None:
        return "unknown"
    lower = status.lower()
    if any(k in lower for k in _UNEMPLOYED):
        return "unemployed"
    if any(k in lower for k in _UNSTABLE_EMPLOYMENT):
        return "unstable"
    if any(k in lower for k in _STABLE_EMPLOYMENT):
        return "stable"
    return "unknown"


def assess_risk(inquiry: InquiryExtraction) -> RiskAssessment:
    """
    Deterministic risk scoring.

    Rules (applied in priority order):
    HIGH  – unemployed OR excessive loan-to-income ratio (>5×)
    MEDIUM – part-time / unknown employment OR moderate DTI (>3×) OR missing info
    LOW   – stable employment AND acceptable DTI (≤3×)
    """
    employment_class = _classify_employment(inquiry.employment_status)

    # Compute loan-to-income ratio when both values are available
    dti_ratio: float | None = None
    if inquiry.loan_amount_requested and inquiry.annual_income and inquiry.annual_income > 0:
        dti_ratio = inquiry.loan_amount_requested / inquiry.annual_income

    # ── HIGH risk ─────────────────────────────────────────────────────────────
    if employment_class == "unemployed":
        return RiskAssessment(
            risk_score_estimate="HIGH",
            risk_reason="Applicant is unemployed — significant repayment risk.",
        )

    if dti_ratio is not None and dti_ratio > _HIGH_DTI_RATIO:
        return RiskAssessment(
            risk_score_estimate="HIGH",
            risk_reason=(
                f"Loan-to-income ratio is {dti_ratio:.1f}× "
                f"(threshold: {_HIGH_DTI_RATIO}×) — excessive loan request relative to income."
            ),
        )

    # ── MEDIUM risk ───────────────────────────────────────────────────────────
    if employment_class in ("unstable", "unknown"):
        return RiskAssessment(
            risk_score_estimate="MEDIUM",
            risk_reason=(
                f"Employment status is '{inquiry.employment_status or 'not provided'}' "
                "— income stability cannot be confirmed."
            ),
        )

    if dti_ratio is not None and dti_ratio > _MED_DTI_RATIO:
        return RiskAssessment(
            risk_score_estimate="MEDIUM",
            risk_reason=(
                f"Loan-to-income ratio is {dti_ratio:.1f}× "
                f"(threshold: {_MED_DTI_RATIO}×) — moderate debt burden concern."
            ),
        )

    # Missing critical info — if employment is stable, default to LOW; otherwise MEDIUM
    if inquiry.annual_income is None or inquiry.loan_amount_requested is None:
        if employment_class == "stable":
            return RiskAssessment(
                risk_score_estimate="LOW",
                risk_reason="Stable employment confirmed; income details not yet provided.",
            )
        return RiskAssessment(
            risk_score_estimate="MEDIUM",
            risk_reason="Incomplete financial information provided — cannot fully assess affordability.",
        )

    # ── LOW risk ──────────────────────────────────────────────────────────────
    return RiskAssessment(
        risk_score_estimate="LOW",
        risk_reason=(
            "Stable employment with an acceptable loan-to-income ratio "
            f"({dti_ratio:.1f}× < {_MED_DTI_RATIO}×)."
        ),
    )
