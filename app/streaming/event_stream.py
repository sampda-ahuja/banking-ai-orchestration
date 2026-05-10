"""
app/streaming/event_stream.py
Server-Sent Events (SSE) streaming for LangGraph workflow events.

Design:
  • Does NOT stream tokens — only streams agent-level completion events.
  • Each event is a JSON object: {"agent": ..., "status": ..., "output": {...}}
  • Uses an async generator consumed by sse-starlette's EventSourceResponse.
"""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from app.graph.state import WorkflowState
from app.graph.workflow import compiled_workflow
from app.utils.logger import get_logger

logger = get_logger(__name__)

# SSE comment to keep connection alive during processing
_KEEPALIVE_COMMENT = ": keepalive\n\n"


async def stream_workflow_events(user_input: str) -> AsyncGenerator[dict[str, str], None]:
    """
    Run the compiled LangGraph workflow and yield SSE-compatible event dicts.

    Each yielded dict contains:
        {"data": "<json-string>", "event": "<event-name>"}

    The workflow runs synchronously inside the async generator (Groq calls
    are synchronous); for high-throughput production use, run in a thread
    pool via asyncio.to_thread.
    """
    import asyncio

    initial_state: WorkflowState = {
        "user_input": user_input,
        "events": [],
        "metrics": {},
    }

    try:
        # Run the synchronous LangGraph workflow in a thread to avoid
        # blocking the event loop.
        final_state: WorkflowState = await asyncio.to_thread(
            compiled_workflow.invoke,  # type: ignore[arg-type]
            initial_state,
        )

        events: list[dict[str, Any]] = final_state.get("events", [])

        for event in events:
            payload = json.dumps(event, default=str)
            yield {"event": "agent_event", "data": payload}
            logger.debug("sse_event_sent", agent=event.get("agent"))

        # Emit a final "done" event with the overall result
        final_payload = json.dumps(
            {
                "agent": "workflow",
                "status": "done",
                "output": {
                    "final_response": final_state.get("final_response", ""),
                    "risk_score": (
                        final_state["risk_result"].risk_score_estimate
                        if "risk_result" in final_state
                        else None
                    ),
                    "compliance_pass": (
                        final_state["compliance_result"].compliance_pass
                        if "compliance_result" in final_state
                        else None
                    ),
                },
            },
            default=str,
        )
        yield {"event": "done", "data": final_payload}

    except Exception as exc:  # noqa: BLE001
        logger.error("workflow_error", error=str(exc))
        error_payload = json.dumps(
            {"agent": "workflow", "status": "error", "output": {"message": str(exc)}}
        )
        yield {"event": "error", "data": error_payload}
