"""
FastAPI route definitions.

Endpoints
---------
GET  /health          – liveness / readiness probe
GET  /model-info      – architecture details and config
POST /predict         – upload MRI -> JSON prediction (base64 images + metrics)
POST /segment         – upload MRI -> binary mask PNG (raw bytes)
GET  /metrics         – aggregate metrics from history
GET  /history         – paginated prediction history
DELETE /history       – clear history
GET  /history/{id}    – single prediction record

All image uploads go through file validation (extension, size, decodability).
"""

import base64
import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from backend.app.core.config import Settings, get_settings
from backend.app.core.engine_state import get_engine, get_engine_meta, is_engine_ready
from backend.app.schemas import StudyMeta
from backend.app.services.history_service import get_history_service
from backend.app.services.prediction_service import run_prediction_pipeline
from backend.app.utils.file_utils import TempFileContext, validate_upload
from inference.postprocessing import decode_image_from_bytes, encode_mask_to_png_bytes

logger = logging.getLogger(__name__)
router = APIRouter()


# Dependency: settings

def _settings() -> Settings:
    return get_settings()


# GET /health

@router.get(
    "/health",
    summary="Health check",
    tags=["System"],
    response_model=dict,
)
async def health_check() -> dict[str, Any]:
    """
    Liveness and readiness probe.

    Returns 200 when the API is up and the model is loaded.
    Returns 503 when the model is not ready yet.
    """
    if not is_engine_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet. Check server logs.",
        )
    return {
        "status":      "healthy",
        "model_ready": True,
        "message":     "Brain Tumour Segmentation API is running.",
    }


# GET /model-info

@router.get(
    "/model-info",
    summary="Model architecture and configuration",
    tags=["Model"],
    response_model=dict,
)
async def model_info() -> dict[str, Any]:
    """
    Return model architecture details, parameter count,
    training configuration, and inference settings.
    """
    meta = get_engine_meta()
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata not available.",
        )
    return meta


# POST /predict

@router.post(
    "/predict",
    summary="Predict tumour segmentation (JSON response with base64 images)",
    tags=["Inference"],
    response_model=dict,
)
async def predict(
    file: UploadFile = File(..., description="MRI brain scan image (PNG/JPG/TIFF/NIfTI)"),
    gt_mask: Optional[UploadFile] = File(
        default=None,
        description="Optional ground-truth mask for metric computation",
    ),
    study_meta: Optional[str] = Form(
        default=None,
        description="Optional StudyMeta as JSON string (patient/study metadata)",
    ),
    settings: Settings = Depends(_settings),
) -> dict[str, Any]:
    """
    Upload an MRI image and receive:

    - Base64-encoded images: original, predicted mask, probability map, overlay
    - Tumour area percentage
    - Confidence score and uncertainty map
    - Dice / IoU / Precision / Recall (when ground-truth mask is provided)
    - Vector contours of tumour boundary

    ⚠️ **Not a medical diagnosis. Research tool only.**
    """
    if not is_engine_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. The server started but engine initialisation failed. Check logs.",
        )
    engine = get_engine()
    history = get_history_service()

    parsed_study_meta: Optional[dict] = None
    if study_meta is not None:
        try:
            sm_raw = json.loads(study_meta)
            sm = StudyMeta(**sm_raw)
            parsed_study_meta = sm.model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid study_meta JSON: {str(exc)}",
            )

    model_meta = get_engine_meta()

    # ── Validate and read image ───────────────────────────────────────
    content = await validate_upload(
        file,
        allowed_extensions=settings.allowed_extensions_set,
        max_bytes=settings.max_upload_bytes,
    )

    suffix = Path(file.filename or "upload.png").suffix.lower()

    # ── Optional ground-truth mask ────────────────────────────────────
    gt_content: Optional[bytes] = None
    if gt_mask is not None:
        gt_content = await validate_upload(
            gt_mask,
            allowed_extensions=settings.allowed_extensions_set,
            max_bytes=settings.max_upload_bytes,
        )

    # ── Process via temp files ────────────────────────────────────────
    async with TempFileContext(content, suffix, settings.temp_dir) as tmp_path:
        image_array = _decode_upload(content, suffix)

        gt_array: Optional[np.ndarray] = None
        if gt_content is not None:
            gt_suffix = Path(gt_mask.filename or "mask.png").suffix.lower()  # type: ignore[union-attr]
            gt_array  = _decode_upload(gt_content, gt_suffix)

        try:
            result = run_prediction_pipeline(
                engine       = engine,
                image_array  = image_array,
                gt_mask_array= gt_array,
                filename     = file.filename or "upload",
                study_meta   = parsed_study_meta,
                model_meta   = model_meta if model_meta else None,
            )
        except Exception as exc:
            logger.error("Prediction failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {str(exc)}",
            )

    # ── Persist to history ────────────────────────────────────────────
    history.add(result)
    logger.info(
        "Prediction complete: %s | tumor=%.2f%% | confidence=%.3f | %dms",
        result["filename"], result["tumor_percentage"],
        result["confidence"], result["inference_ms"],
    )

    return _json_safe(result)


# POST /segment

@router.post(
    "/segment",
    summary="Segment tumour mask (returns raw PNG mask bytes)",
    tags=["Inference"],
    response_class=Response,
)
async def segment(
    file: UploadFile = File(..., description="MRI brain scan image"),
    settings: Settings = Depends(_settings),
) -> Response:
    """
    Upload an MRI image and receive the binary segmentation mask
    as a PNG image (grayscale, white = tumour).

    Useful for programmatic mask download / further processing.
    """
    if not is_engine_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Engine initialisation failed at startup.",
        )
    engine = get_engine()

    content = await validate_upload(
        file,
        allowed_extensions=settings.allowed_extensions_set,
        max_bytes=settings.max_upload_bytes,
    )
    suffix      = Path(file.filename or "upload.png").suffix.lower()
    image_array = _decode_upload(content, suffix)

    try:
        result = run_prediction_pipeline(
            engine=engine,
            image_array=image_array,
            filename=file.filename or "upload",
        )
    except Exception as exc:
        logger.error("Segmentation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation error: {str(exc)}",
        )

    mask_bytes = encode_mask_to_png_bytes(result["pred_mask"])
    return Response(
        content=mask_bytes,
        media_type="image/png",
        headers={
            "X-Tumor-Percentage":  str(result["tumor_percentage"]),
            "X-Confidence":        str(result["confidence"]),
            "X-Tumor-Detected":    str(result["tumor_detected"]).lower(),
        },
    )


# POST /report/pdf

class ReportPdfRequest(BaseModel):
    prediction_result: dict
    images: Optional[dict] = None


@router.post(
    "/report/pdf",
    summary="Generate PDF report from prediction result",
    tags=["Reporting"],
    response_class=Response,
)
async def report_pdf(
    body: ReportPdfRequest = Body(
        ...,
        description="Prediction result dict (from /predict) and optional images dict with base64 data",
    ),
) -> Response:
    """
    Compose and stream a professional radiology-style PDF report based on a
    Brain Tumor Segmentation prediction result payload.

    Accepts:
      - prediction_result: dict (required) – the output of /predict, including
        study, model_info, images, and metrics.
      - images:            dict (optional) – if provided, overrides the
        prediction_result['images'] bundle for embedded figures.

    Returns HTTP 200 with `Content-Type: application/pdf`.
    """
    from backend.app.services.report_service import build_report_pdf

    try:
        pdf_bytes = build_report_pdf(body.prediction_result, body.images)
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF generation failed: {str(exc)}",
        )

    study = body.prediction_result.get("study") or {}
    study_id = study.get("study_id") or body.prediction_result.get("prediction_id") or "REPORT"
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    filename = f"BrainTumorSeg_Report_{study_id}_{ts}.pdf"

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers=headers,
    )


# GET /metrics

@router.get(
    "/metrics",
    summary="Aggregate performance metrics from history",
    tags=["Analytics"],
    response_model=dict,
)
async def get_metrics() -> dict[str, Any]:
    """
    Return aggregate statistics computed over the prediction history:
    detection rate, average confidence, average Dice (when GT was provided).
    """
    history = get_history_service()
    summary = history.summary()
    return {
        "history_count": history.count(),
        **summary,
        "note": (
            "Dice/IoU metrics are only available for predictions where "
            "a ground-truth mask was uploaded alongside the MRI."
        ),
    }


# GET /history

@router.get(
    "/history",
    summary="Paginated prediction history",
    tags=["Analytics"],
    response_model=dict,
)
async def get_history(
    limit:  int = Query(default=20, ge=1, le=100, description="Max records to return"),
    offset: int = Query(default=0,  ge=0,           description="Number of records to skip"),
) -> dict[str, Any]:
    """Return the most recent prediction records (most recent first)."""
    history = get_history_service()
    records = history.get_all(limit=limit, offset=offset)
    return {
        "total":   history.count(),
        "offset":  offset,
        "limit":   limit,
        "records": records,
    }


@router.delete(
    "/history",
    summary="Clear prediction history",
    tags=["Analytics"],
    response_model=dict,
)
async def clear_history() -> dict[str, Any]:
    """Remove all stored prediction records."""
    history = get_history_service()
    count   = history.clear()
    return {"cleared": count, "message": f"Deleted {count} prediction records."}


@router.get(
    "/history/{prediction_id}",
    summary="Get a single prediction record by ID",
    tags=["Analytics"],
    response_model=dict,
)
async def get_history_record(prediction_id: str) -> dict[str, Any]:
    """Retrieve a single prediction record by its UUID."""
    history = get_history_service()
    record  = history.get_by_id(prediction_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction ID {prediction_id!r} not found in history.",
        )
    return record


# Internal helpers

def _json_safe(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to JSON-serializable Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        # Skip raw ndarrays at top level; they should not leak via JSON.
        return None
    return obj


def _decode_upload(content: bytes, suffix: str) -> np.ndarray:
    """Decode raw upload bytes to a NumPy image array."""
    from inference.postprocessing import decode_image_from_bytes
    import nibabel as nib  # type: ignore[import]
    import numpy as np
    import io

    if suffix in (".nii", ".gz"):
        try:
            import nibabel as nib
            nii  = nib.load(io.BytesIO(content))
            data = nii.get_fdata().astype(np.float32)
            if data.ndim == 3:
                data = data[:, :, data.shape[2] // 2]
            return data
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse NIfTI file: {exc}",
            )

    return decode_image_from_bytes(content)
