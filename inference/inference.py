#!/usr/bin/env python3
"""
inference.py – Programmatic inference helper
=============================================
Provides a clean functional API wrapping SegmentationEngine, used by:
  - FastAPI backend routes
  - Jupyter notebooks
  - Testing utilities

Functions
---------
- run_inference(image_array, cfg, model)   : core prediction call
- predict_from_path(path, engine)          : file-based convenience wrapper
- generate_report(result)                  : build structured JSON report
- save_prediction_artifacts(result, dir)   : persist all output files
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def run_inference(
    image: np.ndarray,
    engine: Any,
    gt_mask: Optional[np.ndarray] = None,
    apply_postprocessing: bool = True,
    min_component_size: int = 100,
) -> dict[str, Any]:
    """
    Run full inference pipeline on a raw image array.

    Parameters
    ----------
    image                : (H, W) or (H, W, C) MRI image array
    engine               : SegmentationEngine instance
    gt_mask              : optional ground-truth mask for metrics
    apply_postprocessing : run morphological post-processing on prediction
    min_component_size   : minimum tumour component size (pixels)

    Returns
    -------
    Enriched result dict (engine result + post-processed mask).
    """
    from inference.postprocessing import postprocess_mask

    result = engine.predict_array(image, gt_mask=gt_mask)

    if apply_postprocessing:
        result["pred_mask_raw"]  = result["pred_mask"].copy()
        result["pred_mask"]      = postprocess_mask(
            result["pred_mask"],
            min_component_size=min_component_size,
        )
        # Recompute overlay with post-processed mask
        result["overlay_rgb"] = engine._make_overlay(
            result["processed_image"], result["pred_mask"]
        )

    return result


def predict_from_path(
    image_path: str | Path,
    engine: Any,
    gt_mask_path: Optional[str | Path] = None,
    apply_postprocessing: bool = True,
) -> dict[str, Any]:
    """Wrapper around run_inference that accepts file paths."""
    image_path = Path(image_path)
    image = engine.preprocessor.load_image(image_path)

    gt_mask: Optional[np.ndarray] = None
    if gt_mask_path:
        raw_mask = engine.preprocessor.load_mask(gt_mask_path)
        gt_mask  = engine.preprocessor.process_mask(raw_mask)

    result = run_inference(
        image=image,
        engine=engine,
        gt_mask=gt_mask,
        apply_postprocessing=apply_postprocessing,
    )
    result["filename"] = image_path.name
    return result


def generate_report(result: dict[str, Any]) -> dict[str, Any]:
    """
    Build a clean JSON-serialisable report from an inference result.

    Strips NumPy arrays and adds human-readable fields.
    """
    metrics = result.get("metrics") or {}

    report: dict[str, Any] = {
        "filename":         result.get("filename", "unknown"),
        "image_size":       result.get("image_size"),
        "elapsed_ms":       result.get("elapsed_ms"),

        # Tumour quantification
        "tumor_detected":   result.get("tumor_pixels", 0) > 0,
        "tumor_pixels":     result.get("tumor_pixels", 0),
        "total_pixels":     result.get("total_pixels", 0),
        "tumor_percentage": result.get("tumor_percentage", 0.0),

        # Model confidence / uncertainty
        "confidence":       result.get("confidence", 0.0),
        "is_confident":     result.get("is_confident", False),

        # Performance metrics (only when GT is available)
        "metrics": {
            "dice":        metrics.get("dice"),
            "iou":         metrics.get("iou"),
            "precision":   metrics.get("precision"),
            "recall":      metrics.get("recall"),
            "f1":          metrics.get("f1"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
        } if metrics else None,

        # Legal / safety
        "disclaimer": result.get(
            "disclaimer",
            "AI-assisted research tool only. NOT a medical diagnosis.",
        ),
    }
    return report


def save_prediction_artifacts(
    result: dict[str, Any],
    output_dir: str | Path,
    stem: Optional[str] = None,
    fmt: str = "png",
) -> dict[str, str]:
    """
    Save all inference outputs to disk.

    Parameters
    ----------
    result     : dict from run_inference / predict_from_path
    output_dir : directory path
    stem       : filename stem (defaults to result["filename"])
    fmt        : image format ("png" or "jpeg")

    Returns
    -------
    dict mapping artifact name -> saved file path.
    """
    from inference.postprocessing import encode_mask_to_png_bytes, encode_image_to_png_bytes

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if stem is None:
        stem = Path(result.get("filename", "result")).stem

    saved: dict[str, str] = {}

    def _write(name: str, arr: np.ndarray, is_mask: bool = False) -> str:
        path = out_dir / f"{stem}_{name}.{fmt}"
        if is_mask:
            data = encode_mask_to_png_bytes(arr)
            (out_dir / f"{stem}_{name}.png").write_bytes(data)
            return str(out_dir / f"{stem}_{name}.png")
        else:
            data = encode_image_to_png_bytes(arr)
            path = out_dir / f"{stem}_{name}.png"
            path.write_bytes(data)
            return str(path)

    # Predicted mask
    if "pred_mask" in result:
        saved["pred_mask"] = _write("pred_mask", result["pred_mask"], is_mask=True)

    # Probability map
    if "prob_map" in result:
        saved["prob_map"] = _write("prob_map", result["prob_map"])

    # Overlay
    if "overlay_rgb" in result:
        saved["overlay"] = _write("overlay", result["overlay_rgb"])

    # Uncertainty map
    if "uncertainty_map" in result and result["uncertainty_map"].max() > 0:
        saved["uncertainty"] = _write("uncertainty", result["uncertainty_map"])

    # JSON report
    report = generate_report(result)
    report_path = out_dir / f"{stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    saved["report"] = str(report_path)

    logger.info("Saved %d artifacts to %s", len(saved), out_dir)
    return saved
