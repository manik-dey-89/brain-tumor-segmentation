"""
File handling utilities for the FastAPI backend.

Provides:
- validate_upload        : check extension, size, and basic image integrity
- save_temp_file         : write UploadFile to a secure temp location
- cleanup_temp_file      : delete a temp file safely
- TempFileContext        : async context manager for temp-file lifecycle
"""

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

logger = logging.getLogger(__name__)


# Validation

async def validate_upload(
    file: UploadFile,
    allowed_extensions: set[str],
    max_bytes: int,
) -> bytes:
    """
    Read and validate an uploaded file.

    Checks
    ------
    1. File extension is in allowed_extensions
    2. File size ≤ max_bytes
    3. File is a valid image (decodable by OpenCV)

    Returns
    -------
    Raw file bytes.

    Raises
    ------
    HTTPException 400 on validation failure.
    """
    filename = file.filename or ""
    suffix   = Path(filename).suffix.lower()

    # 1. Extension check
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type: {suffix!r}. "
                f"Allowed: {sorted(allowed_extensions)}"
            ),
        )

    # 2. Read content + size check
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File too large: {len(content) / (1024*1024):.1f} MB. "
                f"Maximum: {max_bytes // (1024*1024)} MB."
            ),
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 3. Image decodability check (skip for NIfTI)
    if suffix not in (".nii", ".gz"):
        buf = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode image. Ensure the file is a valid image.",
            )

    logger.debug("Validated upload: %s (%d bytes)", filename, len(content))
    return content


def save_temp_file(
    content: bytes,
    suffix: str,
    temp_dir: str | Path,
) -> Path:
    """
    Save raw bytes to a uniquely-named temporary file.

    Parameters
    ----------
    content  : file bytes
    suffix   : file extension including dot (e.g. ".png")
    temp_dir : directory for temporary files

    Returns
    -------
    Path to the saved temporary file.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path   = temp_dir / unique_name
    temp_path.write_bytes(content)

    logger.debug("Temp file saved: %s", temp_path)
    return temp_path


def cleanup_temp_file(path: Optional[str | Path]) -> None:
    """Delete a temporary file, silently ignoring errors."""
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
        logger.debug("Cleaned up temp file: %s", path)
    except Exception as exc:
        logger.warning("Could not delete temp file %s: %s", path, exc)


def file_hash(content: bytes) -> str:
    """Return an MD5 hex digest of file content (for deduplication)."""
    return hashlib.md5(content).hexdigest()


# Async context manager for temp-file lifecycle

class TempFileContext:
    """
    Async context manager that saves an UploadFile to a temp location
    and cleans it up when the block exits.

    Usage
    -----
        async with TempFileContext(file, settings) as tmp_path:
            result = engine.predict_file(tmp_path)
    """

    def __init__(
        self,
        content: bytes,
        suffix: str,
        temp_dir: str | Path,
    ) -> None:
        self.content  = content
        self.suffix   = suffix
        self.temp_dir = temp_dir
        self._path: Optional[Path] = None

    async def __aenter__(self) -> Path:
        self._path = save_temp_file(self.content, self.suffix, self.temp_dir)
        return self._path

    async def __aexit__(self, *args) -> None:
        cleanup_temp_file(self._path)
