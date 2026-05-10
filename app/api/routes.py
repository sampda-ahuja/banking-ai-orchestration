"""
app/api/routes.py
FastAPI route definitions.

Endpoints:
    POST /loan-inquiry  — SSE-streamed workflow execution
    GET  /health        — Health check
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.schemas.models import HealthResponse, LoanInquiryRequest
from app.streaming.event_stream import stream_workflow_events
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/loan-inquiry",
    summary="Submit a loan inquiry",
    description=(
        "Accepts a free-text loan inquiry and returns an SSE stream of agent events. "
        "Each event represents the completion of one workflow node."
    ),
    response_class=EventSourceResponse,
)
async def loan_inquiry(request: LoanInquiryRequest) -> EventSourceResponse:
    """
    POST /loan-inquiry

    Returns a Server-Sent Events stream.
    Events are structured as:
        event: agent_event | done | error
        data: <JSON>
    """
    logger.info("loan_inquiry_received", message_length=len(request.message))

    return EventSourceResponse(
        stream_workflow_events(request.message),
        media_type="text/event-stream",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health() -> JSONResponse:
    """GET /health — Returns the application health status."""
    settings = get_settings()
    return JSONResponse(
        content=HealthResponse(environment=settings.app_env).model_dump()
    )
