"""
Post-processing utilities applied to raw prediction masks.

Functions
---------
- remove_small_components   : remove connected components below a pixel threshold
- fill_holes                : fill holes inside predicted tumour region
- smooth_mask               : morphological closing to smooth jagged boundaries
- apply_crf                 : Dense CRF refinement (optional, requires pydensecrf)
- mask_to_contours          : extract contour coordinates for frontend overlay
- encode_mask_to_png_bytes  : encode mask to PNG bytes for API response
- encode_image_to_png_bytes : encode any array to PNG bytes
- decode_image_from_bytes   : decode PNG/JPEG bytes to numpy array
"""

import io
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Morphological post-processing

def remove_small_components(
    mask: np.ndarray,
    min_size: int = 100,
) -> np.ndarray:
    """
    Remove connected components smaller than min_size pixels.

    Parameters
    ----------
    mask     : (H, W) binary uint8 mask
    min_size : minimum component size in pixels

    Returns
    -------
    Cleaned binary uint8 mask.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    out = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_size:
            out[labels == label_id] = 1
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes (enclosed background regions) inside the tumour mask."""
    mask    = mask.astype(np.uint8)
    flooded = mask.copy()
    h, w    = mask.shape
    flood   = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flooded, flood, (0, 0), 1)
    holes   = (flooded == 0).astype(np.uint8)
    return np.logical_or(mask, holes).astype(np.uint8)


def smooth_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply morphological closing to smooth mask boundaries.

    Parameters
    ----------
    mask        : (H, W) binary uint8 mask
    kernel_size : structuring element size
    iterations  : number of closing iterations
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iterations
    )


def postprocess_mask(
    mask: np.ndarray,
    min_component_size: int = 100,
    do_fill_holes: bool = True,
    do_smooth: bool = True,
    smooth_kernel: int = 5,
) -> np.ndarray:
    """Full post-processing pipeline for a predicted binary mask."""
    out = mask.astype(np.uint8)
    out = remove_small_components(out, min_size=min_component_size)
    if do_fill_holes:
        out = fill_holes(out)
    if do_smooth:
        out = smooth_mask(out, kernel_size=smooth_kernel)
    return out


# Contour extraction (for frontend polygon overlay)

def mask_to_contours(
    mask: np.ndarray,
) -> list[list[list[int]]]:
    """
    Extract contour coordinates from a binary mask.

    Returns
    -------
    List of contours, each contour is a list of [x, y] integer points.
    (Suitable for JSON serialisation.)
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    result: list[list[list[int]]] = []
    for c in contours:
        pts = c.reshape(-1, 2).tolist()
        result.append(pts)
    return result


# Image encoding / decoding helpers (for API transport)

def encode_mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """
    Encode a binary mask (0/1 uint8) to PNG bytes (0/255 grayscale).

    Returns
    -------
    PNG-encoded bytes.
    """
    mask_255 = (mask.astype(np.uint8) * 255)
    success, buf = cv2.imencode(".png", mask_255)
    if not success:
        raise RuntimeError("Failed to encode mask to PNG.")
    return buf.tobytes()


def encode_image_to_png_bytes(image: np.ndarray) -> bytes:
    """
    Encode an arbitrary image array to PNG bytes.

    Handles:
      - (H, W)        -> grayscale
      - (H, W, 1)     -> grayscale
      - (H, W, 3) RGB -> BGR for OpenCV, then PNG
      - (C, H, W)     -> permute to HWC first
    """
    img = image.copy()

    # (C, H, W) -> (H, W, C)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    # Squeeze single channel
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(2)

    # Normalise float to uint8
    if img.dtype != np.uint8:
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    # RGB -> BGR for OpenCV
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image to PNG.")
    return buf.tobytes()


def encode_image_to_jpeg_bytes(
    image: np.ndarray, quality: int = 90
) -> bytes:
    """Encode image array to JPEG bytes (smaller payload for API)."""
    img = image.copy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(2)
    if img.dtype != np.uint8:
        lo, hi = img.min(), img.max()
        img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Failed to encode image to JPEG.")
    return buf.tobytes()


def decode_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Decode PNG/JPEG image bytes to a NumPy array.

    Handles RGBA PNGs by stripping the alpha channel.

    Returns
    -------
    np.ndarray  (H, W)    for grayscale
                (H, W, 3) for RGB
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def array_to_base64(image: np.ndarray, fmt: str = "png") -> str:
    """
    Encode image array to a base64 data-URI string for embedding in JSON.

    Parameters
    ----------
    image : numpy array (any shape / dtype supported by encode functions)
    fmt   : "png" or "jpeg"
    """
    import base64
    if fmt == "jpeg":
        raw = encode_image_to_jpeg_bytes(image)
        mime = "image/jpeg"
    else:
        raw = encode_image_to_png_bytes(image)
        mime = "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"
