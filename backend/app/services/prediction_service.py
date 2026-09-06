"""
Prediction Service
==================
Orchestrates the full pipeline:
  upload -> validate -> preprocess -> inference -> postprocess -> response

Used by the /predict and /segment API routes.
"""

import base64
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.app.core.config import get_settings
from inference.postprocessing import (
    array_to_base64,
    encode_mask_to_png_bytes,
    encode_image_to_png_bytes,
    mask_to_contours,
    postprocess_mask,
)

logger = logging.getLogger(__name__)
settings = get_settings()


def run_prediction_pipeline(
    engine: Any,
    image_array: np.ndarray,
    gt_mask_array: Optional[np.ndarray] = None,
    filename: str = "upload",
    apply_postprocessing: bool = True,
    study_meta: Optional[dict] = None,
    model_meta: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Execute full prediction pipeline and build API response payload.

    Parameters
    ----------
    engine           : SegmentationEngine instance
    image_array      : raw input image (H, W) or (H, W, C)
    gt_mask_array    : optional ground-truth mask for metric computation
    filename         : original upload filename
    apply_postprocessing: run morphological cleanup on predicted mask

    Returns
    -------
    JSON-serialisable dict suitable for API response.
    """
    t0 = time.perf_counter()

    # ── Run inference ─────────────────────────────────────────────────
    result = engine.predict_array(
        image=image_array,
        gt_mask=gt_mask_array,
        filename=filename,
    )

    # ── Post-process mask ─────────────────────────────────────────────
    pred_mask = result["pred_mask"]
    if apply_postprocessing:
        pred_mask = postprocess_mask(pred_mask, min_component_size=100)
        result["pred_mask"]  = pred_mask
        result["overlay_rgb"] = engine._make_overlay(
            result["processed_image"], pred_mask
        )

    # ── Encode images to base64 for JSON transport ─────────────────────
    proc_image   = result["processed_image"]
    prob_map     = result["prob_map"]
    overlay_rgb  = result["overlay_rgb"]
    unc_map      = result["uncertainty_map"]

    # Normalise processed image for display
    img_disp = proc_image.squeeze()
    if img_disp.ndim == 3:
        img_disp = img_disp.mean(axis=0)
    img_disp = ((img_disp - img_disp.min()) /
                (img_disp.max() - img_disp.min() + 1e-8) * 255).astype(np.uint8)

    images_b64 = {
        "original":    array_to_base64(img_disp,        fmt="jpeg"),
        "pred_mask":   array_to_base64(pred_mask * 255, fmt="png"),
        "prob_map":    array_to_base64((prob_map * 255).astype(np.uint8), fmt="png"),
        "overlay":     array_to_base64(overlay_rgb,     fmt="jpeg"),
    }

    # Uncertainty map (only if meaningful)
    if unc_map.max() > 0:
        unc_disp = (unc_map / (unc_map.max() + 1e-8) * 255).astype(np.uint8)
        images_b64["uncertainty"] = array_to_base64(unc_disp, fmt="png")

    # ── Contours for vector overlay ───────────────────────────────────
    contours = mask_to_contours(pred_mask)

    # ── Build metrics payload ─────────────────────────────────────────
    metrics_payload: Optional[dict] = None
    if result.get("metrics"):
        m = result["metrics"]
        metrics_payload = {
            "dice":            m.get("dice"),
            "iou":             m.get("iou"),
            "precision":       m.get("precision"),
            "recall":          m.get("recall"),
            "f1":              m.get("f1"),
            "sensitivity":     m.get("sensitivity"),
            "specificity":     m.get("specificity"),
            "tumor_pixels":    m.get("tumor_pixels"),
            "tumor_percentage": m.get("tumor_percentage"),
        }

    elapsed_total = (time.perf_counter() - t0) * 1000.0

    result_dict: dict[str, Any] = {
        "prediction_id":    str(uuid.uuid4()),
        "filename":         filename,
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        # Quantification
        "tumor_detected":   bool(result["tumor_pixels"] > 0),
        "tumor_pixels":     result["tumor_pixels"],
        "total_pixels":     result["total_pixels"],
        "tumor_percentage": result["tumor_percentage"],

        # Confidence
        "confidence":       result["confidence"],
        "is_confident":     result["is_confident"],

        # Images (base64-encoded)
        "images":           images_b64,

        # Vector contours
        "contours":         contours,

        # Metrics (when GT provided)
        "metrics":          metrics_payload,

        # Performance
        "inference_ms":     result["elapsed_ms"],
        "total_ms":         round(elapsed_total, 2),

        # Safety disclaimer
        "disclaimer": (
            "⚠️ AI-assisted research tool. "
            "This output is NOT a medical diagnosis. "
            "Results must be reviewed by a qualified radiologist."
        ),
    }

    if study_meta is not None:
        result_dict["study"] = study_meta
    if model_meta is not None:
        result_dict["model_info"] = model_meta

    return result_dict
