"""Segmentation evaluation metrics."""
import numpy as np
from typing import Optional


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b > 1e-8 else default


class SegmentationMetrics:
    """Pixel-level segmentation metrics."""

    @staticmethod
    def compute_from_arrays(
        prob_map: np.ndarray,
        gt_mask: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """
        Compute segmentation metrics from probability map and ground-truth mask.

        Parameters
        ----------
        prob_map  : (H, W) float32 predicted probability map  [0, 1]
        gt_mask   : (H, W) float32 / uint8 binary ground-truth mask  {0, 1}
        threshold : binarisation threshold for prob_map

        Returns
        -------
        dict with keys: dice, iou, precision, recall, f1,
                        sensitivity, specificity, hausdorff
        """
        pred = (prob_map >= threshold).astype(np.float32).ravel()
        true = (gt_mask  >= 0.5      ).astype(np.float32).ravel()

        tp = float(np.sum(pred * true))
        fp = float(np.sum(pred * (1.0 - true)))
        fn = float(np.sum((1.0 - pred) * true))
        tn = float(np.sum((1.0 - pred) * (1.0 - true)))

        dice        = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
        iou         = _safe_div(tp, tp + fp + fn)
        precision   = _safe_div(tp, tp + fp)
        recall      = _safe_div(tp, tp + fn)
        f1          = dice  # same as dice for binary
        sensitivity = recall
        specificity = _safe_div(tn, tn + fp)

        # Hausdorff distance (95th percentile, approximate)
        hausdorff = _hausdorff_95(pred.reshape(prob_map.shape), true.reshape(gt_mask.shape))

        return {
            "dice":        round(dice,        4),
            "iou":         round(iou,         4),
            "precision":   round(precision,   4),
            "recall":      round(recall,      4),
            "f1":          round(f1,          4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "hausdorff":   round(hausdorff,   4) if hausdorff is not None else None,
        }


def _hausdorff_95(pred: np.ndarray, true: np.ndarray) -> Optional[float]:
    """95th-percentile Hausdorff distance (returns None if either mask empty)."""
    try:
        from scipy.ndimage import distance_transform_edt
        pred_b = pred.astype(bool)
        true_b = true.astype(bool)
        if not pred_b.any() or not true_b.any():
            return None
        dt_pred = distance_transform_edt(~pred_b)
        dt_true = distance_transform_edt(~true_b)
        hd_pt = np.percentile(dt_true[pred_b], 95)
        hd_tp = np.percentile(dt_pred[true_b], 95)
        return float(max(hd_pt, hd_tp))
    except Exception:
        return None
