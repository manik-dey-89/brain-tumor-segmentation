"""
Prediction History Service
==========================
In-memory + file-backed store for recent prediction results.

Thread-safe (uses a lock) and limited to `history_limit` entries
to prevent unbounded memory growth.
"""

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HistoryService:
    """
    Manages a bounded, persistent prediction history.

    Parameters
    ----------
    max_entries  : maximum number of history records to keep in memory
    history_file : optional JSON file for persistence across restarts
    """

    def __init__(
        self,
        max_entries: int = 100,
        history_file: Optional[str | Path] = None,
    ) -> None:
        self._lock        = threading.Lock()
        self._max         = max_entries
        self._history: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._file        = Path(history_file) if history_file else None

        if self._file and self._file.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------
    def add(self, prediction: dict[str, Any]) -> None:
        """
        Add a prediction record to the history.
        Only stores JSON-serialisable metadata (no image arrays).
        """
        record = self._strip_arrays(prediction)

        with self._lock:
            self._history.appendleft(record)

        # Async-friendly: persist to disk without blocking (best-effort)
        self._save_to_disk()

    def get_all(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Return history records (most recent first).

        Parameters
        ----------
        limit  : max records to return (None = all)
        offset : skip first N records
        """
        with self._lock:
            records = list(self._history)

        sliced = records[offset:]
        if limit is not None:
            sliced = sliced[:limit]
        return sliced

    def get_by_id(self, prediction_id: str) -> Optional[dict[str, Any]]:
        """Find a specific prediction by its UUID."""
        with self._lock:
            for r in self._history:
                if r.get("prediction_id") == prediction_id:
                    return r
        return None

    def clear(self) -> int:
        """Clear all history. Returns count of cleared records."""
        with self._lock:
            count = len(self._history)
            self._history.clear()
        self._save_to_disk()
        return count

    def count(self) -> int:
        with self._lock:
            return len(self._history)

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics over stored history."""
        with self._lock:
            records = list(self._history)

        if not records:
            return {"total": 0}

        tumor_detected    = [r for r in records if r.get("tumor_detected")]
        dice_values       = [
            r["metrics"]["dice"] for r in records
            if r.get("metrics") and r["metrics"].get("dice") is not None
        ]
        confidence_values = [r["confidence"] for r in records if "confidence" in r]
        tumor_pcts        = [r["tumor_percentage"] for r in records if "tumor_percentage" in r]

        return {
            "total":               len(records),
            "tumor_detected_count": len(tumor_detected),
            "tumor_detection_rate": round(len(tumor_detected) / len(records), 4),
            "avg_confidence":       round(sum(confidence_values) / len(confidence_values), 4)
                                    if confidence_values else None,
            "avg_dice":             round(sum(dice_values) / len(dice_values), 4)
                                    if dice_values else None,
            "avg_tumor_percentage": round(sum(tumor_pcts) / len(tumor_pcts), 4)
                                    if tumor_pcts else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_to_disk(self) -> None:
        if self._file is None:
            return
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = list(self._history)
            with open(self._file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not save history to disk: %s", exc)

    def _load_from_disk(self) -> None:
        try:
            with open(self._file) as f:  # type: ignore[arg-type]
                data = json.load(f)
            with self._lock:
                for record in reversed(data[-self._max:]):
                    self._history.appendleft(record)
            logger.info("Loaded %d history records from %s", len(self._history), self._file)
        except Exception as exc:
            logger.warning("Could not load history from disk: %s", exc)

    # ------------------------------------------------------------------
    @staticmethod
    def _strip_arrays(prediction: dict[str, Any]) -> dict[str, Any]:
        """Remove NumPy arrays and large binary blobs from a prediction dict."""
        import numpy as np

        STUDY_PRESERVE_KEYS = {"study_id", "patient_id", "full_name", "mri_modality", "scan_date"}

        def _clean(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return "<array>"
            if isinstance(v, dict):
                return {kk: _clean(vv) for kk, vv in v.items()}
            if isinstance(v, list) and v and isinstance(v[0], list):
                return "<contours>"          # skip polygon data
            return v

        def _clean_study_dict(study_dict: dict[str, Any]) -> dict[str, Any]:
            cleaned = {}
            for k, v in study_dict.items():
                if isinstance(v, np.ndarray):
                    cleaned[k] = "<array>"
                else:
                    cleaned[k] = v
            return cleaned

        cleaned: dict[str, Any] = {}
        study_data = prediction.get("study")
        if isinstance(study_data, dict):
            preserved_study = _clean_study_dict(study_data)
            cleaned["study"] = preserved_study
            for key in STUDY_PRESERVE_KEYS:
                if key in preserved_study:
                    cleaned[key] = preserved_study[key]

        for k, v in prediction.items():
            if k == "study":
                continue
            if k in STUDY_PRESERVE_KEYS and k in cleaned:
                continue
            if k == "images":
                cleaned[k] = {ik: "<base64>" for ik in v} if isinstance(v, dict) else "<images>"
            elif k == "contours":
                cleaned[k] = f"<{len(v) if isinstance(v, list) else 0} contours>"
            else:
                cleaned[k] = _clean(v)
        return cleaned


# Module-level singleton

_history_service: Optional[HistoryService] = None


def get_history_service() -> HistoryService:
    global _history_service
    if _history_service is None:
        from backend.app.core.config import get_settings
        s = get_settings()
        _history_service = HistoryService(
            max_entries  = s.history_limit,
            history_file = s.history_file,
        )
    return _history_service
