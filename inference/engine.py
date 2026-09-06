"""
Inference Engine
================
Central inference class used by both the CLI scripts and the FastAPI backend.

Capabilities
------------
- Single image and batch inference
- Test-Time Augmentation (TTA) with horizontal/vertical flips
- MC-Dropout uncertainty estimation (multiple forward passes with dropout active)
- Confidence / quality scoring
- Returns rich result dict with:
    prob_map, pred_mask, overlay_rgb, metrics (if GT supplied),
    uncertainty_map, confidence_score, tumor_area_px, tumor_percentage

Usage
-----
    from inference.engine import SegmentationEngine

    engine = SegmentationEngine.from_config("configs/config.yaml")

    result = engine.predict_file("path/to/mri.png")
    result = engine.predict_array(image_array, gt_mask=gt_array)
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


class SegmentationEngine:
    """
    Wraps a trained segmentation model for inference.

    Parameters
    ----------
    model         : trained nn.Module (already loaded + to device)
    preprocessor  : Preprocessor instance
    cfg           : full config dict
    device        : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: Any,
        cfg: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model        = model
        self.preprocessor = preprocessor
        self.cfg          = cfg
        self.device       = device

        infer_cfg = cfg.get("inference", {})
        eval_cfg  = cfg.get("evaluation", {})
        viz_cfg   = cfg.get("visualization", {})

        self.threshold          = eval_cfg.get("threshold", 0.5)
        self.tta_enabled        = infer_cfg.get("tta", True)
        self.tta_flips          = infer_cfg.get("tta_flips", ["horizontal", "vertical"])
        self.mc_dropout         = infer_cfg.get("mc_dropout", True)
        self.mc_samples         = infer_cfg.get("mc_samples", 20)
        self.confidence_thresh  = infer_cfg.get("confidence_threshold", 0.6)
        self.colormap           = viz_cfg.get("colormap", "jet")
        self.overlay_alpha      = viz_cfg.get("overlay_alpha", 0.45)
        self.num_classes        = cfg.get("data", {}).get("num_classes", 1)

        # Image size for preprocessing
        self.image_size: tuple[int, int] = tuple(
            cfg.get("data", {}).get("image_size", [256, 256])
        )  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SegmentationEngine":
        """Build an engine directly from a config file path."""
        from models.factory import build_model, get_device, load_checkpoint
        from data.preprocessing import Preprocessor

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        device = get_device(cfg.get("inference", {}).get("device", "auto"))

        cfg["model"]["num_classes"] = cfg["data"].get("num_classes", 1)
        cfg["model"]["in_channels"] = cfg["data"].get("num_channels", 1)

        model = build_model(cfg["model"]).to(device)

        model_path = Path(cfg["inference"].get("model_path", "outputs/checkpoints/best_model.pth"))
        if model_path.exists():
            load_checkpoint(model, str(model_path), device)
            logger.info("Loaded checkpoint: %s", model_path)
        else:
            logger.warning("No checkpoint found at %s – using random weights.", model_path)

        prep_cfg = {
            **cfg.get("data", {}),
            **cfg.get("preprocessing", {}),
        }
        preprocessor = Preprocessor(prep_cfg)

        return cls(model=model, preprocessor=preprocessor, cfg=cfg, device=device)

    # ------------------------------------------------------------------
    # Public predict API
    # ------------------------------------------------------------------

    def predict_file(
        self,
        image_path: str | Path,
        gt_mask_path: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        """
        Run inference on a single image file.

        Parameters
        ----------
        image_path    : path to MRI image file
        gt_mask_path  : optional path to ground-truth mask for metric computation

        Returns
        -------
        dict with all inference results (see _build_result_dict for keys)
        """
        image_path = Path(image_path)
        image = self.preprocessor.load_image(image_path)

        gt_mask: Optional[np.ndarray] = None
        if gt_mask_path is not None:
            raw_mask = self.preprocessor.load_mask(gt_mask_path)
            gt_mask  = self.preprocessor.process_mask(raw_mask)

        return self.predict_array(image, gt_mask=gt_mask, filename=image_path.name)

    def predict_array(
        self,
        image: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        filename: str = "input",
    ) -> dict[str, Any]:
        """
        Run inference on a raw NumPy image array.

        Parameters
        ----------
        image    : raw MRI array (H, W) or (H, W, C) or (C, H, W)
        gt_mask  : optional ground-truth binary mask (H, W)
        filename : identifier used in result metadata

        Returns
        -------
        dict with all inference results
        """
        t0 = time.perf_counter()

        # ── Preprocess ───────────────────────────────────────────────
        proc_image = self.preprocessor.process_image(image)   # (C, H, W) float32
        tensor = torch.from_numpy(proc_image).unsqueeze(0).to(self.device)  # (1, C, H, W)

        # ── Forward pass(es) ─────────────────────────────────────────
        if self.mc_dropout and self._has_dropout():
            prob_map, uncertainty_map = self._mc_dropout_predict(tensor)
        else:
            prob_map      = self._deterministic_predict(tensor)
            uncertainty_map = np.zeros_like(prob_map)

        # ── TTA ──────────────────────────────────────────────────────
        if self.tta_enabled and self.tta_flips:
            prob_map = self._tta_predict(tensor, base_prob=prob_map)
            # Re-compute uncertainty if TTA applied (approximate as std of TTA probs)
            if not (self.mc_dropout and self._has_dropout()):
                uncertainty_map = self._tta_uncertainty(tensor)

        # ── Threshold -> binary mask ───────────────────────────────────
        pred_mask = (prob_map >= self.threshold).astype(np.uint8)

        # ── Overlay ──────────────────────────────────────────────────
        overlay_rgb = self._make_overlay(proc_image, pred_mask)

        # ── Metrics (if GT available) ─────────────────────────────────
        from training.metrics import SegmentationMetrics
        metrics: Optional[dict] = None
        if gt_mask is not None:
            metrics = SegmentationMetrics.compute_from_arrays(
                prob_map, gt_mask.astype(np.float32), threshold=self.threshold
            )

        # ── Confidence score ──────────────────────────────────────────
        confidence = self._compute_confidence(prob_map, uncertainty_map)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return self._build_result_dict(
            filename       = filename,
            proc_image     = proc_image,
            prob_map       = prob_map,
            pred_mask      = pred_mask,
            uncertainty_map= uncertainty_map,
            overlay_rgb    = overlay_rgb,
            metrics        = metrics,
            confidence     = confidence,
            elapsed_ms     = elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Prediction strategies
    # ------------------------------------------------------------------

    def _deterministic_predict(self, tensor: torch.Tensor) -> np.ndarray:
        """Single deterministic forward pass -> probability map."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
        return self._logits_to_prob(logits)

    def _mc_dropout_predict(
        self, tensor: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout: run N forward passes with dropout active.
        Returns (mean_prob, std_prob) as uncertainty estimate.
        """
        self._enable_dropout()
        probs: list[np.ndarray] = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.model(tensor)
                probs.append(self._logits_to_prob(logits))

        self.model.eval()   # disable dropout again
        prob_stack    = np.stack(probs, axis=0)   # (N, H, W)
        mean_prob     = prob_stack.mean(axis=0)
        uncertainty   = prob_stack.std(axis=0)
        return mean_prob, uncertainty

    def _tta_predict(
        self, tensor: torch.Tensor, base_prob: np.ndarray
    ) -> np.ndarray:
        """
        Test-Time Augmentation: average flipped predictions with base prediction.
        """
        self.model.eval()
        probs = [base_prob]

        with torch.no_grad():
            for flip in self.tta_flips:
                aug_tensor, inverse_fn = self._flip_tensor(tensor, flip)
                logits   = self.model(aug_tensor)
                aug_prob = self._logits_to_prob(logits)
                # Reverse the flip on the probability map
                probs.append(self._flip_numpy(aug_prob, flip))

        return np.mean(probs, axis=0)

    def _tta_uncertainty(self, tensor: torch.Tensor) -> np.ndarray:
        """Compute TTA-based uncertainty as std of TTA probability maps."""
        self.model.eval()
        probs = []

        with torch.no_grad():
            for flip in self.tta_flips:
                aug_tensor, _ = self._flip_tensor(tensor, flip)
                logits  = self.model(aug_tensor)
                aug_prob = self._logits_to_prob(logits)
                probs.append(self._flip_numpy(aug_prob, flip))

        base_prob = self._deterministic_predict(tensor)
        probs.insert(0, base_prob)
        return np.stack(probs).std(axis=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _logits_to_prob(self, logits: torch.Tensor) -> np.ndarray:
        """Convert model logits to a 2D probability map."""
        if self.num_classes == 1:
            prob = torch.sigmoid(logits.squeeze()).cpu().numpy()
        else:
            prob = torch.softmax(logits, dim=1)[:, 1].squeeze().cpu().numpy()
        return prob.astype(np.float32)

    def _has_dropout(self) -> bool:
        return any(isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in self.model.modules())

    def _enable_dropout(self) -> None:
        """Set model to train mode only for Dropout layers (keeps BN in eval)."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    @staticmethod
    def _flip_tensor(
        tensor: torch.Tensor, flip: str
    ) -> tuple[torch.Tensor, Any]:
        if flip == "horizontal":
            return torch.flip(tensor, dims=[-1]), flip
        elif flip == "vertical":
            return torch.flip(tensor, dims=[-2]), flip
        raise ValueError(f"Unknown flip: {flip!r}")

    @staticmethod
    def _flip_numpy(arr: np.ndarray, flip: str) -> np.ndarray:
        if flip == "horizontal":
            return np.flip(arr, axis=-1).copy()
        elif flip == "vertical":
            return np.flip(arr, axis=-2).copy()
        raise ValueError(f"Unknown flip: {flip!r}")

    def _make_overlay(
        self, proc_image: np.ndarray, pred_mask: np.ndarray
    ) -> np.ndarray:
        """Return (H, W, 3) uint8 RGB overlay of prediction on MRI."""
        from training.visualization import create_overlay_array
        return create_overlay_array(
            proc_image, pred_mask,
            colormap=self.colormap,
            alpha=self.overlay_alpha,
        )

    def _compute_confidence(
        self, prob_map: np.ndarray, uncertainty_map: np.ndarray
    ) -> float:
        """
        Confidence score in [0, 1].
        High confidence = predictions are clearly above/below threshold
        with low uncertainty.
        """
        # Distance from decision boundary (0.5)
        margin = np.abs(prob_map - 0.5).mean()

        # Penalise high uncertainty (if available)
        if uncertainty_map.max() > 0:
            mean_unc = uncertainty_map.mean()
            confidence = float(margin * (1.0 - 2.0 * mean_unc))
        else:
            confidence = float(margin * 2.0)

        return float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _build_result_dict(
        filename:        str,
        proc_image:      np.ndarray,
        prob_map:        np.ndarray,
        pred_mask:       np.ndarray,
        uncertainty_map: np.ndarray,
        overlay_rgb:     np.ndarray,
        metrics:         Optional[dict],
        confidence:      float,
        elapsed_ms:      float,
    ) -> dict[str, Any]:
        tumor_pixels = int(pred_mask.sum())
        total_pixels = int(pred_mask.size)
        tumor_pct    = round(100.0 * tumor_pixels / max(total_pixels, 1), 4)

        return {
            # Metadata
            "filename":         filename,
            "elapsed_ms":       round(elapsed_ms, 2),
            "image_size":       list(prob_map.shape),

            # Arrays (accessible by backend/CLI)
            "processed_image":  proc_image,
            "prob_map":         prob_map,
            "pred_mask":        pred_mask,
            "uncertainty_map":  uncertainty_map,
            "overlay_rgb":      overlay_rgb,

            # Quantitative
            "tumor_pixels":     tumor_pixels,
            "total_pixels":     total_pixels,
            "tumor_percentage": tumor_pct,
            "confidence":       round(confidence, 4),
            "is_confident":     confidence >= 0.6,

            # Metrics (populated when GT is available)
            "metrics":          metrics,

            # Disclaimer
            "disclaimer": (
                "AI-assisted research tool only. "
                "NOT a medical diagnosis. "
                "Always consult a qualified radiologist."
            ),
        }

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def predict_batch(
        self, image_paths: list[str | Path]
    ) -> list[dict[str, Any]]:
        """Run inference on a list of image files."""
        results = []
        for p in image_paths:
            try:
                r = self.predict_file(p)
                results.append(r)
            except Exception as exc:
                logger.error("Failed to process %s: %s", p, exc)
                results.append({"filename": str(p), "error": str(exc)})
        return results
