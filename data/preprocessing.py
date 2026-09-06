"""
Preprocessing pipeline for brain MRI images and segmentation masks.

Handles:
  - Loading (PNG/JPG/TIFF/NIfTI)
  - Resizing
  - Intensity normalisation (z-score, min-max, percentile)
  - Denoising (Gaussian, Median, Bilateral)
  - Value clipping
  - Mask validation and binarisation
  - Channel standardisation
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Optional NIfTI support
try:
    import nibabel as nib
    _NIBABEL_AVAILABLE = True
except ImportError:
    _NIBABEL_AVAILABLE = False


class Preprocessor:
    """
    Stateless preprocessing pipeline driven by config dict.

    Parameters
    ----------
    cfg : dict
        The ``preprocessing:`` section from config.yaml, plus
        ``image_size``, ``num_channels``, ``num_classes`` from ``data:``.
    """

    def __init__(self, cfg: dict) -> None:
        self.image_size: Tuple[int, int] = tuple(cfg.get("image_size", [256, 256]))  # type: ignore[assignment]
        self.num_channels: int           = cfg.get("num_channels", 1)
        self.num_classes: int            = cfg.get("num_classes", 2)

        # Normalisation
        self.normalize: bool             = cfg.get("normalize", True)
        self.norm_method: str            = cfg.get("normalization_method", "zscore")
        self.p_low: float                = cfg.get("percentile_low", 1.0)
        self.p_high: float               = cfg.get("percentile_high", 99.0)

        # Denoising
        self.denoise: bool               = cfg.get("denoise", True)
        self.denoise_method: str         = cfg.get("denoise_method", "gaussian")
        self.denoise_sigma: float        = cfg.get("denoise_sigma", 1.0)

        # Clipping
        self.clip_values: bool           = cfg.get("clip_values", True)
        self.clip_min: float             = cfg.get("clip_min", 0.0)
        self.clip_max: float             = cfg.get("clip_max", 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to a single MRI image array.

        Parameters
        ----------
        image : np.ndarray
            Raw image array, any dtype, shape (H, W) or (H, W, C).

        Returns
        -------
        np.ndarray
            Float32 array, shape (C, H, W), values in [clip_min, clip_max].
        """
        img = self._to_float32(image)
        img = self._ensure_channels(img)       # -> (H, W, C)
        img = self._resize(img)                # -> (H, W, C)

        if self.denoise:
            img = self._denoise(img)

        if self.normalize:
            img = self._normalize(img)

        if self.clip_values:
            img = np.clip(img, self.clip_min, self.clip_max)

        return np.transpose(img, (2, 0, 1)).astype(np.float32)  # (C, H, W)

    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Validate and binarise a segmentation mask.

        Parameters
        ----------
        mask : np.ndarray
            Raw mask array, shape (H, W) or (H, W, 1).

        Returns
        -------
        np.ndarray
            Long (int64) array, shape (H, W), values in {0, …, num_classes-1}.
        """
        msk = mask.squeeze()
        if msk.ndim != 2:
            raise ValueError(f"Mask must be 2-D after squeezing; got shape {msk.shape}")

        msk = cv2.resize(
            msk.astype(np.float32),
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        msk = self._validate_mask(msk)
        return msk.astype(np.int64)

    def load_image(self, path: str | Path) -> np.ndarray:
        """Load an image file (PNG/JPG/TIFF/NIfTI) as a NumPy array."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in (".nii", ".gz"):
            return self._load_nifti(path)

        flag = cv2.IMREAD_GRAYSCALE if self.num_channels == 1 else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        if self.num_channels == 3 and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def load_mask(self, path: str | Path) -> np.ndarray:
        """Load a mask file (PNG/NIfTI) as a NumPy array."""
        path = Path(path)
        if path.suffix.lower() in (".nii", ".gz"):
            return self._load_nifti(path)

        msk = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        return msk

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float32(arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float32)

    def _ensure_channels(self, img: np.ndarray) -> np.ndarray:
        """Ensure shape is (H, W, C). Handles 4-channel RGBA images."""
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if img.shape[2] == 4:
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            if self.num_channels == 1:
                img = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2GRAY)
                img = img[:, :, np.newaxis].astype(np.float32)
            else:
                img = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2RGB).astype(np.float32)
        elif img.shape[2] == 3 and self.num_channels == 1:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            img = img[:, :, np.newaxis].astype(np.float32)
        elif img.shape[2] == 1 and self.num_channels == 3:
            img = np.repeat(img, 3, axis=2)
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = self.image_size
        if img.shape[:2] == (h, w):
            return img
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        return resized

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        method = self.denoise_method.lower()
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            ch = img[:, :, c]
            if method == "gaussian":
                k = int(2 * round(2 * self.denoise_sigma) + 1)
                out[:, :, c] = cv2.GaussianBlur(ch, (k, k), self.denoise_sigma)
            elif method == "median":
                k = max(3, int(self.denoise_sigma * 2 + 1))
                k = k if k % 2 == 1 else k + 1
                out[:, :, c] = cv2.medianBlur(ch, k)
            elif method == "bilateral":
                out[:, :, c] = cv2.bilateralFilter(ch, 9, 75, 75)
            else:
                out[:, :, c] = ch
        return out

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        method = self.norm_method.lower()
        out = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            ch = img[:, :, c]
            if method == "zscore":
                mu, sigma = ch.mean(), ch.std()
                out[:, :, c] = (ch - mu) / (sigma + 1e-8)
            elif method == "minmax":
                lo, hi = ch.min(), ch.max()
                out[:, :, c] = (ch - lo) / (hi - lo + 1e-8)
            elif method == "percentile":
                lo = np.percentile(ch, self.p_low)
                hi = np.percentile(ch, self.p_high)
                out[:, :, c] = (ch - lo) / (hi - lo + 1e-8)
            else:
                out[:, :, c] = ch
        return out

    def _validate_mask(self, msk: np.ndarray) -> np.ndarray:
        """
        Validate and clean a segmentation mask:
          - binary: threshold at 0.5 -> {0, 1}
          - multi-class: ensure integer labels in [0, num_classes-1]
        Emits a warning when unexpected values are found.
        """
        unique = np.unique(msk)

        if self.num_classes == 2:
            # Binary tumour mask
            if msk.max() > 1.5:
                # Pixel values likely 0/255 – normalise
                msk = (msk > 127).astype(np.float32)
            else:
                msk = (msk > 0.5).astype(np.float32)
        else:
            # Multi-class
            unexpected = [v for v in unique if v < 0 or v >= self.num_classes]
            if unexpected:
                logger.warning(
                    "Mask contains unexpected label values %s; clipping to [0, %d].",
                    unexpected,
                    self.num_classes - 1,
                )
                msk = np.clip(np.round(msk), 0, self.num_classes - 1)

        return msk

    def _load_nifti(self, path: Path) -> np.ndarray:
        if not _NIBABEL_AVAILABLE:
            raise ImportError(
                "nibabel is required to load NIfTI files.  "
                "Install it with:  pip install nibabel"
            )
        nii = nib.load(str(path))
        data = nii.get_fdata()

        # For 3-D volumes return the middle slice
        if data.ndim == 3:
            mid = data.shape[2] // 2
            data = data[:, :, mid]

        return data.astype(np.float32)
