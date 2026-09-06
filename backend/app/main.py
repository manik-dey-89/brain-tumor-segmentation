"""
backend/app/main.py
===================
FastAPI application entrypoint.

Creates the ASGI `app` instance that uvicorn targets via:
    uvicorn backend.app.main:app ...

Startup sequence (lifespan):
  1. Load settings from environment / .env
  2. Ensure required output directories exist
  3. Load SegmentationEngine via from_config()
  4. Register engine via set_engine() so all routes can access it
  5. Yield (server accepts requests)
  6. On shutdown: log teardown (engine itself is stateless/GC'd)
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import get_settings
from backend.app.core.engine_state import set_engine

logger = logging.getLogger("brain_tumor_api")


def _ensure_output_dirs(settings) -> None:
    """Create any output directories the app writes to, if they don't exist."""
    for dir_path in (
        Path(settings.temp_dir),
        Path(settings.history_file).parent,
    ):
        dir_path.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything before `yield` runs at startup; after `yield` runs at shutdown.
    """
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("  %s  -  Starting up", settings.app_name)
    logger.info("=" * 60)

    _ensure_output_dirs(settings)

    # --- Load inference engine -------------------------------------------
    from inference.engine import SegmentationEngine

    t0 = time.perf_counter()
    try:
        engine = SegmentationEngine.from_config(settings.config_path)
    except FileNotFoundError as exc:
        logger.error("Config or checkpoint not found: %s", exc)
        raise RuntimeError(
            f"Cannot start server: {exc}. "
            "Run training first or place a checkpoint at the configured path."
        ) from exc
    except Exception as exc:
        logger.error("Engine initialisation failed: %s", exc, exc_info=True)
        raise

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Build a small meta dict that /model-info and /health can read back
    meta: dict = {
        "architecture":  getattr(engine, "architecture", "attention_unet"),
        "device":        str(getattr(engine, "device", "cpu")),
        "checkpoint":    settings.model_path,
        "config_path":   settings.config_path,
        "threshold":     getattr(engine, "threshold", settings.inference_threshold),
        "tta_enabled":   getattr(engine, "tta_enabled", False),
        "mc_dropout_samples": getattr(engine, "mc_dropout_samples", 0),
        "params":        getattr(engine, "num_params", None),
        "load_ms":       round(elapsed_ms, 2),
    }

    set_engine(engine, meta)

    arch  = meta["architecture"]
    params = f"{meta['params']:,}" if meta["params"] else "?"
    logger.info(
        "Model loaded in %.2f ms | arch=%s | params=%s | device=%s",
        elapsed_ms, arch, params, meta["device"],
    )
    logger.info("API ready -> http://%s:%s", settings.host, settings.port)
    logger.info("Docs      -> http://%s:%s/docs", settings.host, settings.port)

    yield  # --- server is running ---

    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """
    Build and configure the FastAPI application.

    The module-level `app` is created by calling this function so that the
    app object is available at import time (required by uvicorn's string
    import: `backend.app.main:app`).
    """
    settings = get_settings()

    application = FastAPI(
        title       = settings.app_name,
        version     = settings.app_version,
        description = (
            "AI-assisted brain tumour segmentation using Attention U-Net. "
            "Research tool — not a medical device."
        ),
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        openapi_url = "/openapi.json",
        lifespan    = lifespan,
    )

    # --- CORS ---------------------------------------------------------------
    # Build the final origins list: settings value (env-configurable) plus
    # the deployed Render URLs hardcoded as a safety net so a misconfigured
    # CORS_ORIGINS env var never locks out the production frontend.
    _render_origins = [
        "https://brain-tumor-segmentation-1-savp.onrender.com",
    ]
    _origins = list(dict.fromkeys(settings.cors_origins_list + _render_origins))

    application.add_middleware(
        CORSMiddleware,
        allow_origins     = _origins,
        allow_credentials = True,
        allow_methods     = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers     = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-API-Key",
            "Cache-Control",
        ],
        expose_headers    = ["Content-Disposition"],
        max_age           = 600,   # preflight cache: 10 min
    )

    # --- Routes -------------------------------------------------------------
    # Import here (not at module top-level) to avoid circular imports if any
    # route module transitively imports from main.
    from backend.app.api.routes import router
    application.include_router(router, prefix="/api/v1")

    return application


# Module-level app instance — uvicorn imports this as `backend.app.main:app`
app: FastAPI = create_app()
