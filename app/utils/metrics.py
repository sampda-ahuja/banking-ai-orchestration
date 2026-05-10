"""
app/utils/metrics.py
Token usage and latency extraction from LangChain LLM responses.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from langchain_core.messages import BaseMessage


@dataclass
class LLMMetrics:
    """Captured metrics from a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


def extract_token_usage(response: Any) -> LLMMetrics:
    """
    Extract token usage from a LangChain response.

    Works with:
    - AIMessage (has response_metadata or usage_metadata)
    - Pydantic model returned by with_structured_output (check __dict__ for raw response)
    """
    metrics = LLMMetrics()

    # Case 1: AIMessage with response_metadata
    if isinstance(response, BaseMessage):
        meta = getattr(response, "response_metadata", {})
        usage = meta.get("token_usage", {}) or meta.get("usage", {})
        if usage:
            metrics.prompt_tokens = usage.get("prompt_tokens", 0)
            metrics.completion_tokens = usage.get("completion_tokens", 0)
            metrics.total_tokens = usage.get("total_tokens", 0)
        # Fallback: usage_metadata attribute (newer LangChain)
        elif hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            metrics.prompt_tokens = getattr(um, "input_tokens", 0)
            metrics.completion_tokens = getattr(um, "output_tokens", 0)
            metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens

    return metrics


@contextmanager
def timed() -> Generator[dict[str, float], None, None]:
    """Context manager that records elapsed time in ms."""
    result: dict[str, float] = {"latency_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["latency_ms"] = (time.perf_counter() - start) * 1000
