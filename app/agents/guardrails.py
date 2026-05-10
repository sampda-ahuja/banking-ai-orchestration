"""
app/agents/guardrails.py
Agent 1 — Guardrails.

Hybrid design:
  • Deterministic PII scan via regex (app/policies/pii_rules.py)
  • LLM classification for banking relevance + emotional escalation only
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.policies.pii_rules import scan_for_pii
from app.schemas.models import GuardrailResponse
from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.utils.metrics import LLMMetrics, extract_token_usage

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "guardrails_prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def _build_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.guardrail_model,
        temperature=0,
        api_key=settings.groq_api_key,
    )


def run_guardrails(user_input: str) -> tuple[GuardrailResponse, LLMMetrics]:
    """
    Execute the guardrail pipeline:
    1. Deterministic PII scan (fast, reliable)
    2. LLM classification for banking relevance + escalation

    Returns a tuple of (GuardrailResponse, LLMMetrics).
    """
    # ── Step 1: Deterministic PII detection ──────────────────────────────────
    pii_result = scan_for_pii(user_input)
    if pii_result.has_pii:
        logger.warning(
            "pii_detected",
            types=pii_result.detected_types,
            redacted=pii_result.redacted_text,
        )
        # Short-circuit: no need to send PII to the LLM
        return (
            GuardrailResponse(
                is_banking_related=True,
                no_pii=False,
                needs_escalation=False,
                escalation_reason=None,
            ),
            LLMMetrics(),  # No LLM call — zero tokens
        )

    # ── Step 2: LLM classification ───────────────────────────────────────────
    llm = _build_llm()
    # Use include_raw=True to get the AIMessage with token metadata
    structured_llm = llm.with_structured_output(GuardrailResponse, include_raw=True)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"User message:\n\n{user_input}"),
    ]

    raw_response = structured_llm.invoke(messages)
    result: GuardrailResponse = raw_response["parsed"]  # type: ignore[index]
    raw_msg = raw_response["raw"]  # type: ignore[index]

    # Extract token usage from the raw AIMessage
    metrics = extract_token_usage(raw_msg)

    # Merge deterministic PII result (always no_pii=True here since we passed scan)
    if result.no_pii == False or result.is_jailbreak == True:
        result.needs_escalation = False

    logger.info(
        "guardrail_result",
        is_banking_related=result.is_banking_related,
        no_pii=result.no_pii,
        needs_escalation=result.needs_escalation,
        escalation_reason=result.escalation_reason,
        tokens=metrics.total_tokens,
    )
    return result, metrics
