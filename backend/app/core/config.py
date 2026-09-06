"""
Backend application configuration.
Reads values from environment variables (with .env file support via python-dotenv).
Falls back to defaults from configs/config.yaml when not overridden.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────
    app_name: str          = "Brain Tumor Segmentation API"
    app_version: str       = "1.0.0"
    debug: bool            = False
    environment: str       = "production"    # "development" | "production"

    # ── Server ────────────────────────────────────────────────────────
    host: str              = "0.0.0.0"
    port: int              = 8000

    # ── CORS ──────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.
    # Set the CORS_ORIGINS environment variable on Render (or in .env) to
    # override.  The deployed frontend URL is included in the default so the
    # service works out-of-the-box without any environment variable.
    cors_origins: str = (
        "http://localhost:3000,"
        "http://localhost:5173,"
        "https://brain-tumor-segmentation-1-savp.onrender.com"
    )

    # ── Model / Inference ─────────────────────────────────────────────
    config_path: str       = "configs/config.yaml"
    model_path: str        = "outputs/checkpoints/best_model.pth"
    device: str            = "auto"
    inference_threshold: float = 0.5

    # ── Upload ────────────────────────────────────────────────────────
    max_upload_mb: int         = 50
    allowed_extensions: str    = ".png,.jpg,.jpeg,.tif,.tiff,.nii,.nii.gz"
    temp_dir: str              = "outputs/temp"

    # ── History ───────────────────────────────────────────────────────
    history_limit: int         = 100
    history_file: str          = "outputs/logs/history.json"

    # ── Security ──────────────────────────────────────────────────────
    api_key: Optional[str]     = None          # if set, require X-API-Key header
    rate_limit_per_minute: int = 60

    # ── Computed properties ───────────────────────────────────────────
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
