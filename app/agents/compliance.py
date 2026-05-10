"""
app/agents/compliance.py
Agent 3 — Compliance Agent.

Hybrid design:
  • Deterministic rule checks (guarantee claims, ECOA, UDAAP, disclaimer)
  • LLM review for nuanced deceptive/manipulative language

The LLM is invoked ONLY when deterministic checks pass, to catch
subtle patterns that regex cannot reliably detect.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.policies.compliance_rules import (
    check_disclaimer_present,
    check_ecoa_violations,
    check_guarantee_claims,
    check_udaap_violations,
)
from app.schemas.models import ComplianceResponse
from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.utils.metrics import LLMMetrics, extract_token_usage

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "compliance_prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def _build_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.compliance_model,
        temperature=0,
        api_key=settings.groq_api_key,
    )


def run_compliance(generated_response: str) -> tuple[ComplianceResponse, LLMMetrics]:
    """
    Execute the compliance pipeline against the generated response text.

    Step 1 — Deterministic checks (fast, zero-hallucination):
      • Guarantee/approval language
      • ECOA protected-class references
      • UDAAP deceptive language patterns
      • Required disclaimer presence

    Step 2 — LLM review (nuanced semantic understanding):
      • Manipulative urgency
      • Implied guarantees not caught by regex
      • Bait-and-switch phrasing

    Returns a tuple of (ComplianceResponse, LLMMetrics).
    """
    # ── Step 1: Deterministic checks ─────────────────────────────────────────
    violations: list[str] = []
    violations.extend(check_guarantee_claims(generated_response))
    violations.extend(check_ecoa_violations(generated_response))
    violations.extend(check_udaap_violations(generated_response))
    violations.extend(check_disclaimer_present(generated_response))

    if violations:
        # Deterministic violations found — no need to invoke LLM
        logger.warning("compliance_deterministic_fail", violations=violations)
        return (
            ComplianceResponse(compliance_pass=False, violations=violations),
            LLMMetrics(),  # No LLM call — zero tokens
        )

    # ── Step 2: LLM nuanced review ────────────────────────────────────────────
    llm = _build_llm()
    structured_llm = llm.with_structured_output(ComplianceResponse, include_raw=True)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Review the following generated banking response for compliance issues:\n\n"
                f"---\n{generated_response}\n---"
            )
        ),
    ]

    raw_response = structured_llm.invoke(messages)
    llm_result: ComplianceResponse = raw_response["parsed"]  # type: ignore[index]
    raw_msg = raw_response["raw"]  # type: ignore[index]

    # Extract token usage
    metrics = extract_token_usage(raw_msg)

    if not llm_result.compliance_pass:
        logger.warning("compliance_llm_fail", violations=llm_result.violations)
    else:
        logger.info("compliance_pass", tokens=metrics.total_tokens)

    return llm_result, metrics
