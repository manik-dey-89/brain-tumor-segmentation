"""
Global inference engine singleton for the FastAPI application.

The engine is initialised once at startup (lifespan event) and shared
across all requests through dependency injection.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_engine: Optional[object] = None
_engine_meta: dict = {}


def set_engine(engine: object, meta: dict) -> None:
    global _engine, _engine_meta
    _engine = engine
    _engine_meta = meta
    logger.info("Inference engine registered in app state.")


def get_engine() -> object:
    if _engine is None:
        raise RuntimeError(
            "Inference engine is not initialised. "
            "Ensure the model checkpoint exists and the server started correctly."
        )
    return _engine


def get_engine_meta() -> dict:
    return _engine_meta


def is_engine_ready() -> bool:
    return _engine is not None
