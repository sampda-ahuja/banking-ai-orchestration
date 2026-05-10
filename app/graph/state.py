"""
app/graph/state.py
LangGraph workflow state definition.
Uses TypedDict for compatibility with LangGraph's state management.
"""
from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from app.schemas.models import (
    ComplianceResponse,
    GuardrailResponse,
    InquiryExtraction,
    RiskAssessment,
)


class NodeMetrics(TypedDict, total=False):
    """Per-node performance metrics."""
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class WorkflowState(TypedDict, total=False):
    """
    Shared mutable state passed between LangGraph nodes.

    Fields are optional (total=False) to allow partial initialization.
    Each node reads what it needs and writes its result back.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    user_input: str

    # ── Per-agent outputs ────────────────────────────────────────────────────
    guardrail_result: GuardrailResponse
    inquiry_result: InquiryExtraction
    risk_result: RiskAssessment
    compliance_result: ComplianceResponse

    # ── Final synthesised response ───────────────────────────────────────────
    final_response: str

    # ── Streaming event log ──────────────────────────────────────────────────
    # Each entry is a dict emitted to the SSE stream
    events: list[dict[str, Any]]

    # ── Per-node metrics (latency + token usage) ─────────────────────────────
    metrics: dict[str, NodeMetrics]
