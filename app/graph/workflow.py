"""
app/graph/workflow.py
LangGraph workflow definition.

Topology:
    START → guardrail_node → inquiry_parser_node → risk_policy_node → compliance_node → END

Conditional routing:
    • PII detected          → safe_terminate_node → END
    • Non-banking message   → safe_terminate_node → END
    • Emotional escalation  → escalation_node     → END
    • Compliance failure    → fallback_node        → END
"""
from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from app.schemas.models import ComplianceResponse
from app.agents.compliance import run_compliance
from app.agents.guardrails import run_guardrails
from app.agents.inquiry_parser import run_inquiry_parser
from app.graph.state import NodeMetrics, WorkflowState
from app.policies.risk_engine import assess_risk
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Helper: event appender ──────────────────────────────────────────────────


def _append_event(state: WorkflowState, agent: str, status: str, output: Any) -> None:
    """Append a structured event to state['events']."""
    events: list[dict[str, Any]] = list(state.get("events", []))
    events.append({"agent": agent, "status": status, "output": output})
    state["events"] = events  # type: ignore[typeddict-unknown-key]


def _record_metrics(state: WorkflowState, node_name: str, metrics: NodeMetrics) -> None:
    """Record per-node metrics into state."""
    all_metrics: dict[str, NodeMetrics] = dict(state.get("metrics", {}))
    all_metrics[node_name] = metrics
    state["metrics"] = all_metrics  # type: ignore[typeddict-unknown-key]


# ─── Node implementations ─────────────────────────────────────────────────────


def guardrail_node(state: WorkflowState) -> WorkflowState:
    """Agent 1: Run guardrails — PII scan + LLM relevance/escalation check."""
    logger.info("node_start", node="guardrail")
    t0 = time.perf_counter()
    result, llm_metrics = run_guardrails(state["user_input"])
    latency_ms = (time.perf_counter() - t0) * 1000

    state["guardrail_result"] = result
    _append_event(state, "guardrails", "completed", result.model_dump())
    _record_metrics(state, "guardrails", NodeMetrics(
        latency_ms=latency_ms,
        prompt_tokens=llm_metrics.prompt_tokens,
        completion_tokens=llm_metrics.completion_tokens,
        total_tokens=llm_metrics.total_tokens,
    ))
    logger.info("node_end", node="guardrail", latency_ms=round(latency_ms))
    return state


def inquiry_parser_node(state: WorkflowState) -> WorkflowState:
    """Agent 2: Parse loan intent, extract structured fields, generate response."""
    logger.info("node_start", node="inquiry_parser")
    t0 = time.perf_counter()
    result, llm_metrics = run_inquiry_parser(state["user_input"])
    latency_ms = (time.perf_counter() - t0) * 1000

    state["inquiry_result"] = result
    _append_event(state, "inquiry_parser", "completed", result.model_dump())
    _record_metrics(state, "inquiry_parser", NodeMetrics(
        latency_ms=latency_ms,
        prompt_tokens=llm_metrics.prompt_tokens,
        completion_tokens=llm_metrics.completion_tokens,
        total_tokens=llm_metrics.total_tokens,
    ))
    logger.info("node_end", node="inquiry_parser", latency_ms=round(latency_ms))
    return state


def risk_policy_node(state: WorkflowState) -> WorkflowState:
    """Deterministic policy engine: score risk from inquiry extraction."""
    logger.info("node_start", node="risk_policy")
    t0 = time.perf_counter()
    risk = assess_risk(state["inquiry_result"])
    latency_ms = (time.perf_counter() - t0) * 1000

    state["risk_result"] = risk
    _append_event(state, "risk_policy", "completed", risk.model_dump())
    _record_metrics(state, "risk_policy", NodeMetrics(
        latency_ms=latency_ms,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    ))
    logger.info("node_end", node="risk_policy", score=risk.risk_score_estimate)
    return state


def compliance_node(state: WorkflowState) -> WorkflowState:
    """Agent 3: Validate generated response against compliance rules."""
    logger.info("node_start", node="compliance")
    t0 = time.perf_counter()
    response_text = state["inquiry_result"].summary_response
    result, llm_metrics = run_compliance(response_text)
    latency_ms = (time.perf_counter() - t0) * 1000

    state["compliance_result"] = result

    if result.compliance_pass:
        state["final_response"] = response_text

    _append_event(state, "compliance", "completed", result.model_dump())
    _record_metrics(state, "compliance", NodeMetrics(
        latency_ms=latency_ms,
        prompt_tokens=llm_metrics.prompt_tokens,
        completion_tokens=llm_metrics.completion_tokens,
        total_tokens=llm_metrics.total_tokens,
    ))
    logger.info("node_end", node="compliance", pass_=result.compliance_pass)
    return state


# ─── Terminal / fallback nodes ────────────────────────────────────────────────


def safe_terminate_node(state: WorkflowState) -> WorkflowState:
    """
    Safe termination for PII-detected or non-banking messages.
    Emits a generic, non-committal response.
    """
    gr = state.get("guardrail_result")
    if gr and not gr.no_pii:
        reason = "PII detected in your message"
        response = (
            "For your security, we cannot process messages containing personal "
            "identification information. Please contact our secure banking portal "
            "or visit a branch directly. Do not share sensitive data via chat."
        )
    else:
        reason = "Message not related to banking services"
        response = (
            "I'm only able to assist with banking and loan-related inquiries. "
            "Please reach out if you have questions about our loan products."
        )

    state["final_response"] = response
    _append_event(
        state,
        "safe_terminate",
        "terminated",
        {"reason": reason, "response": response},
    )
    logger.warning("workflow_terminated", reason=reason)
    return state


def escalation_node(state: WorkflowState) -> WorkflowState:
    """
    Escalation path for detected emotional distress or crisis signals.
    Routes to human support without further automated processing.
    """
    gr = state.get("guardrail_result")
    reason = (gr.escalation_reason if gr else None) or "Emotional distress detected"

    response = (
        "We understand you may be going through a difficult time. "
        "A dedicated banking specialist will contact you shortly to provide "
        "personalised assistance. If this is an emergency, please call our "
        "24/7 support line. Your well-being is our priority."
    )
    state["final_response"] = response
    _append_event(
        state,
        "escalation",
        "escalated",
        {"reason": reason, "response": response},
    )
    logger.warning("workflow_escalated", reason=reason)
    return state


def fallback_node(state: WorkflowState) -> WorkflowState:
    """
    Generic fallback handler for compliance or jailbreak scenarios.
    Returns a safe, policy-aligned response.
    """
    cr = state.get("compliance_result")
    gr = state.get("guardrail_result")

    violations = cr.violations if cr else []

    # Jailbreak-specific fallback
    if gr and getattr(gr, "is_jailbreak", False):
        response = (
            "I'm unable to process requests that attempt to override "
            "banking policies, approval procedures, or safety guidelines. "
            "Please submit a standard loan-related inquiry."
        )
        state["compliance_result"] = ComplianceResponse(
    compliance_pass=False,
    violations=["jailbreak_attempt"],
)

        reason = "jailbreak_detected"

    # Generic compliance fallback
    else:
        response = (
            "Thank you for your loan inquiry. A banking specialist will review "
            "your request and provide a personalised, accurate response. "
            "All loan offers and estimates are subject to formal credit review and approval."
        )

        reason = "compliance_fallback"

    state["final_response"] = response

    _append_event(
        state,
        reason,
        "fallback",
        {
            "violations": violations,
            "response": response,
        },
    )

    logger.warning(reason, violations=violations)

    return state

# ─── Conditional edge functions ───────────────────────────────────────────────

def route_after_guardrail(state: WorkflowState) -> str:
    """
    Determine next node after the guardrail check.

    Priority:
        1. PII detected          → safe_terminate
        2. Not banking-related   → safe_terminate
        3. Emotional distress    → escalation
        4. All clear             → inquiry_parser
    """
    gr = state["guardrail_result"]

    if not gr.no_pii:
        return "safe_terminate"

    if gr.is_jailbreak:
        return "fallback"

    if not gr.is_banking_related:
        return "safe_terminate"

    if gr.needs_escalation:
        return "escalation"

    return "inquiry_parser"

def route_after_compliance(state: WorkflowState) -> str:
    """Route to fallback if compliance failed, otherwise end."""
    cr = state["compliance_result"]
    return "end" if cr.compliance_pass else "fallback"


# ─── Graph assembly ───────────────────────────────────────────────────────────


def build_workflow() -> StateGraph:
    """Assemble and compile the LangGraph workflow."""
    graph = StateGraph(WorkflowState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("inquiry_parser", inquiry_parser_node)
    graph.add_node("risk_policy", risk_policy_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("safe_terminate", safe_terminate_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("fallback", fallback_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "guardrail")

    # ── Conditional routing after guardrail ───────────────────────────────────
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "safe_terminate": "safe_terminate",
            "escalation": "escalation",
            "inquiry_parser": "inquiry_parser",
            "fallback": "fallback",
        },
    )

    # ── Linear happy path ─────────────────────────────────────────────────────
    graph.add_edge("inquiry_parser", "risk_policy")
    graph.add_edge("risk_policy", "compliance")

    # ── Conditional routing after compliance ──────────────────────────────────
    graph.add_conditional_edges(
        "compliance",
        route_after_compliance,
        {
            "end": END,
            "fallback": "fallback",
        },
    )

    # ── Terminal edges ────────────────────────────────────────────────────────
    graph.add_edge("safe_terminate", END)
    graph.add_edge("escalation", END)
    graph.add_edge("fallback", END)

    return graph


# Compiled graph — import this in route handlers
compiled_workflow = build_workflow().compile()
