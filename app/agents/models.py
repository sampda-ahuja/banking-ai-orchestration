"""
app/schemas/models.py
All shared Pydantic v2 schemas used across agents and the graph.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 – Guardrails
# ─────────────────────────────────────────────────────────────────────────────

class GuardrailResponse(BaseModel):
    """Structured output from the guardrails agent."""

    is_banking_related: bool = Field(
        description="True when the user message is related to banking or lending."
    )
    no_pii: bool = Field(
        description="True when NO PII (SSN, Aadhaar, card numbers, etc.) is present."
    )
    needs_escalation: bool = Field(
        description="True when the user shows NO distress or escalation signals."
    )
    escalation_reason: str | None = Field(
        default=None,
        description="Short description of the escalation trigger, if any.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 – Inquiry Parser
# ─────────────────────────────────────────────────────────────────────────────

class InquiryExtraction(BaseModel):
    """Structured output from the inquiry parser agent."""

    intent: Literal["Mortgage", "Auto", "Personal", "Refinance"] = Field(
        description="The classified loan intent."
    )
    loan_amount_requested: int | None = Field(
        default=None,
        description="The loan amount the user requested, in USD.",
    )
    employment_status: str | None = Field(
        default=None,
        description="Employment status as described by the user.",
    )
    annual_income: int | None = Field(
        default=None,
        description="Self-reported annual income in USD, if mentioned.",
    )
    summary_response: str = Field(
        description=(
            "A professional, compliance-safe response to the user. "
            "Must include the disclaimer: "
            "'All loan offers and estimates are subject to formal credit review and approval.'"
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Policy Engine – Risk Assessment
# ─────────────────────────────────────────────────────────────────────────────

class RiskAssessment(BaseModel):
    """Deterministic risk assessment produced by the policy engine."""

    risk_score_estimate: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Estimated risk tier based on deterministic underwriting rules."
    )
    risk_reason: str = Field(
        description="Human-readable explanation of the risk decision."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 – Compliance
# ─────────────────────────────────────────────────────────────────────────────

class ComplianceResponse(BaseModel):
    """Structured output from the compliance agent."""

    compliance_pass: bool = Field(
        description="True when the generated response passes all compliance checks."
    )
    violations: list[str] = Field(
        default_factory=list,
        description="List of identified compliance violations.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# API – Request / Response
# ─────────────────────────────────────────────────────────────────────────────

class LoanInquiryRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's free-text loan inquiry.",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str
