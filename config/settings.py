"""Centralized, typed configuration. Reads from environment / .env.

On Hugging Face Spaces these values come from Spaces Secrets, not a
committed .env file — see docs/04_deployment_and_environment.md.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Environment ---
    environment: str = Field(default="local", description="local | hf")

    # --- Database ---
    sqlite_path: Path = Field(default=BASE_DIR / "data" / "market.db")

    # --- Cache backend ---
    # "redis" (local dev) or "ttl" (in-process, HF-safe fallback).
    cache_backend: str = Field(default="ttl")
    redis_url: str = Field(default="redis://localhost:6379/0")
    cache_default_ttl_seconds: int = Field(default=300)

    # --- TradingView (BIST via tvDatafeed) ---
    tvdatafeed_username: str | None = None
    tvdatafeed_password: str | None = None

    # --- HF Hub (model/dataset artifact delivery) ---
    hf_hub_token: str | None = None
    hf_model_repo: str | None = None
    hf_dataset_repo: str | None = None

    # --- Logging ---
    log_level: str = Field(default="INFO")

    @field_validator("tvdatafeed_username", "tvdatafeed_password", "hf_hub_token", "hf_model_repo", "hf_dataset_repo", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        # An unset .env/Secret var (e.g. `HF_HUB_TOKEN=`) parses as "",
        # not absent — callers that pass these straight to huggingface_hub
        # (HfApi(token=...), snapshot_download(token=...)) build an
        # invalid empty "Bearer " auth header instead of falling back to
        # cached `hf auth login` credentials. Normalize once, here.
        return value or None


settings = Settings()
