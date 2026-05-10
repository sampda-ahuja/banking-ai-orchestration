"""
app/utils/config.py
Centralised settings loaded from environment / .env file.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Groq ─────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key")

    # ── Model names ──────────────────────────────────────────────────────────
    guardrail_model: str = "llama-3.1-8b-instant"
    parser_model: str = "llama-3.3-70b-versatile"
    compliance_model: str = "llama-3.1-8b-instant"

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    app_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return Settings()  # type: ignore[call-arg]
