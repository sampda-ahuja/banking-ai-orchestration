"""
evals/graph_provider.py
─────────────────────────────────────────────────────────────────────────────
Promptfoo Python provider — calls compiled_workflow directly.

Promptfoo calls call_api(prompt, options, context) for each test case.
The `prompt` is the user_input from the test vars.
Returns a ProviderResponse dict with output + metadata including metrics.

Docs: https://www.promptfoo.dev/docs/providers/python/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load env (GROQ_API_KEY must be available)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "evals" / ".env")   # evals/.env takes precedence

from app.graph.workflow import compiled_workflow
from app.graph.state import WorkflowState


def call_api(
    prompt: str,
    options: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Promptfoo provider entry point.

    `prompt`  — the rendered user_input string from the test vars
    `options` — provider config (unused here)
    `context` — contains vars dict with expected_* fields for assertions

    Returns:
        {
            "output": "<json string with full state snapshot>",
            "tokenUsage": { prompt, completion, total }
        }
    """
    vars_: dict[str, Any] = context.get("vars", {})
    user_input: str = vars_.get("user_input", prompt)

    initial: WorkflowState = {"user_input": user_input, "events": [], "metrics": {}}

    try:
        final_state: WorkflowState = compiled_workflow.invoke(initial)  # type: ignore[assignment]

        # Determine which route was taken
        route = _detect_route(final_state)

        # Aggregate token usage across all nodes
        node_metrics = final_state.get("metrics", {})
        total_prompt = sum(m.get("prompt_tokens", 0) for m in node_metrics.values())
        total_completion = sum(m.get("completion_tokens", 0) for m in node_metrics.values())
        total_tokens = sum(m.get("total_tokens", 0) for m in node_metrics.values())

        return {
            "output": json.dumps({
                "response": final_state.get("final_response", ""),
                "route": route,
                "guardrails": (
                    final_state["guardrail_result"].model_dump()
                    if "guardrail_result" in final_state else None
                ),
                "inquiry_parser": (
                    final_state["inquiry_result"].model_dump()
                    if "inquiry_result" in final_state else None
                ),
                "risk_policy": (
                    final_state["risk_result"].model_dump()
                    if "risk_result" in final_state else None
                ),
                "compliance": (
                    final_state["compliance_result"].model_dump()
                    if "compliance_result" in final_state else None
                ),
                "node_metrics": node_metrics,
            }),
            "tokenUsage": {
                "prompt": total_prompt,
                "completion": total_completion,
                "total": total_tokens,
            },
        }

    except Exception as exc:
        return {
            "output": "",
            "error": str(exc),
            "tokenUsage": {"prompt": 0, "completion": 0, "total": 0},
        }


def _detect_route(state: WorkflowState) -> str:
    for event in state.get("events", []):
        if event.get("agent") == "safe_terminate":
            return "safe_terminate"
        if event.get("agent") == "escalation":
            return "escalation"
        if event.get("agent") == "compliance_fallback":
            return "fallback"
        if event.get("agent") == "jailbreak_detected":
            return "fallback"
    if "compliance_result" in state:
        return "happy_path"
    return "unknown"
