"""Visualization utilities for segmentation results."""
import numpy as np
import cv2


def create_overlay_array(
    image: np.ndarray,
    mask: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a segmentation mask onto an MRI image as a coloured overlay.

    Parameters
    ----------
    image    : (C, H, W) or (H, W) float array, intensity range any
    mask     : (H, W) binary uint8/float array  {0, 1}
    colormap : OpenCV colormap name ('jet', 'hot', etc.) — fallback to jet
    alpha    : overlay blend strength  [0, 1]

    Returns
    -------
    (H, W, 3) uint8 RGB array
    """
    # Normalise image to [0, 255] uint8
    if image.ndim == 3:
        # (C, H, W) -> (H, W)
        img2d = image[0] if image.shape[0] in (1, 3) else image.mean(axis=0)
    else:
        img2d = image.copy().astype(np.float32)

    img_min, img_max = img2d.min(), img2d.max()
    if img_max - img_min > 1e-8:
        img_norm = ((img2d - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_norm = np.zeros_like(img2d, dtype=np.uint8)

    # RGB base
    base_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

    # Build coloured mask
    _COLORMAPS = {
        "jet":   cv2.COLORMAP_JET,
        "hot":   cv2.COLORMAP_HOT,
        "cool":  cv2.COLORMAP_COOL,
        "bone":  cv2.COLORMAP_BONE,
    }
    cv_cmap = _COLORMAPS.get(colormap.lower(), cv2.COLORMAP_JET)
    mask_u8 = (mask * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(mask_u8, cv_cmap)
    coloured_rgb = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    # Blend only where mask > 0
    mask_bool = (mask > 0).astype(np.float32)[..., np.newaxis]
    overlay = (base_rgb.astype(np.float32) * (1.0 - alpha * mask_bool)
               + coloured_rgb.astype(np.float32) * alpha * mask_bool)
    return overlay.clip(0, 255).astype(np.uint8)
