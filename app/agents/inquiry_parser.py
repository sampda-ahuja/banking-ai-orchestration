"""
app/agents/inquiry_parser.py
Agent 2 — Inquiry Parser.

Uses llama-3.3-70b-versatile (Groq) with structured output to extract
intent, loan details, employment info, and generate a compliant response.

Risk computation is intentionally EXCLUDED from this agent —
that is handled deterministically by app/policies/risk_engine.py.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.schemas.models import InquiryExtraction
from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.utils.metrics import LLMMetrics, extract_token_usage

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "inquiry_prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


_DISCLAIMER = (
    "All loan offers and estimates are subject to formal credit review and approval."
)


def _enforce_disclaimer(result: InquiryExtraction) -> None:
    """
    Deterministic post-processing guard: if the LLM omitted the required
    disclaimer, append it to the summary response.
    """
    if _DISCLAIMER.lower() not in result.summary_response.lower():
        result.summary_response = result.summary_response.rstrip() + "\n\n" + _DISCLAIMER
        logger.warning("disclaimer_enforced", action="appended_missing_disclaimer")


def _build_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.parser_model,
        temperature=0,
        api_key=settings.groq_api_key,
    )


def run_inquiry_parser(user_input: str) -> tuple[InquiryExtraction, LLMMetrics]:
    """
    Parse the user inquiry into a structured InquiryExtraction.

    Returns a tuple of (InquiryExtraction, LLMMetrics).
    Raises ValueError if the LLM response cannot be validated by Pydantic.
    """
    llm = _build_llm()
    structured_llm = llm.with_structured_output(InquiryExtraction, include_raw=True)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"User inquiry:\n\n{user_input}"),
    ]

    raw_response = structured_llm.invoke(messages)
    result: InquiryExtraction = raw_response["parsed"]  # type: ignore[index]
    raw_msg = raw_response["raw"]  # type: ignore[index]

    # Extract token usage
    metrics = extract_token_usage(raw_msg)

    # Post-generation safety net: ensure disclaimer is present
    _enforce_disclaimer(result)

    logger.info(
        "inquiry_parsed",
        intent=result.intent,
        loan_amount=result.loan_amount_requested,
        employment=result.employment_status,
        tokens=metrics.total_tokens,
    )
    return result, metrics

