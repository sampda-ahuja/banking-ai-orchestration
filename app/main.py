"""
app/main.py
FastAPI application entry point.

Usage:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Application startup / shutdown lifecycle."""
    settings = get_settings()
    logger.info(
        "application_startup",
        environment=settings.app_env,
        guardrail_model=settings.guardrail_model,
        parser_model=settings.parser_model,
        compliance_model=settings.compliance_model,
    )
    yield
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Banking AI Orchestration",
        description=(
            "Production-style multi-agent AI system for banking loan inquiry workflows. "
            "Implements LangGraph orchestration, hybrid deterministic + LLM agents, "
            "structured outputs, SSE streaming, and enterprise compliance controls."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS (tighten origins in production) ──────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(router, prefix="")

    return app


app = create_app()

def main() -> None:
    """Entry point for `uv run start`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
