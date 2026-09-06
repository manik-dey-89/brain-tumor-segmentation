#!/usr/bin/env python3
"""
scripts/generate_samples.py
============================
Generate synthetic MRI-like sample images with artificial tumour masks
for testing the pipeline without real patient data.

Usage
-----
    python scripts/generate_samples.py --n 50 --out data/raw
    python scripts/generate_samples.py --n 20 --size 256 --out data/samples
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("generate_samples")


def make_mri_like(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a synthetic brain-MRI-like grayscale image.

    Uses concentric ellipses to simulate brain tissue layers
    with realistic intensity gradients and Gaussian noise.
    """
    img = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2

    # Outer skull ring
    cv2.ellipse(img, (cx, cy), (w // 2 - 10, h // 2 - 10), 0, 0, 360, 0.15, -1)
    # Brain parenchyma
    cv2.ellipse(img, (cx, cy), (w // 2 - 20, h // 2 - 20), 0, 0, 360, 0.45, -1)
    # White matter
    cv2.ellipse(img, (cx, cy), (int(w * 0.3), int(h * 0.3)), 0, 0, 360, 0.65, -1)
    # Ventricles (darker)
    cv2.ellipse(img, (cx - 15, cy), (int(w * 0.07), int(h * 0.12)), 0, 0, 360, 0.2, -1)
    cv2.ellipse(img, (cx + 15, cy), (int(w * 0.07), int(h * 0.12)), 0, 0, 360, 0.2, -1)

    # Mild random intensity variation
    noise = rng.normal(0, 0.04, (h, w)).astype(np.float32)
    img   = np.clip(img + noise, 0, 1)

    # Gaussian blur for tissue smoothing
    img = cv2.GaussianBlur(img, (7, 7), 2.0)

    return (img * 255).astype(np.uint8)


def make_tumour_mask(h: int, w: int, rng: np.random.Generator, tumour_prob: float = 0.65) -> np.ndarray:
    """
    Generate an artificial tumour mask.
    tumour_prob fraction of masks will contain a tumour; rest are empty.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if rng.random() > tumour_prob:
        return mask   # no tumour

    # Random position inside brain region
    margin = h // 5
    cy = rng.integers(margin, h - margin)
    cx = rng.integers(margin, w - margin)

    # Random ellipse size (5–15% of image dimension)
    ry = rng.integers(int(h * 0.05), int(h * 0.15))
    rx = rng.integers(int(w * 0.05), int(w * 0.15))
    angle = rng.integers(0, 180)

    cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), int(angle), 0, 360, 255, -1)

    # Optional: irregular edges via distortion
    blur = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.5)
    mask = (blur > 100).astype(np.uint8) * 255

    return mask


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic MRI samples")
    p.add_argument("--n",      type=int, default=50,       help="Number of samples")
    p.add_argument("--size",   type=int, default=256,      help="Image size (square)")
    p.add_argument("--out",    type=str, default="data/raw",help="Output root directory")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--prefix", type=str, default="SYNTH",  help="Filename prefix")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir  = ROOT / args.out
    img_dir  = out_dir / "images"
    msk_dir  = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    tumor_count = 0
    for i in range(args.n):
        # Use zero-padded patient-style naming: SYNTH_001_slice000.png
        pid   = f"{i:04d}"
        fname = f"{args.prefix}_{pid}_slice000.png"

        img  = make_mri_like(args.size, args.size, rng)
        mask = make_tumour_mask(args.size, args.size, rng)

        cv2.imwrite(str(img_dir / fname), img)
        cv2.imwrite(str(msk_dir / fname), mask)

        if mask.max() > 0:
            tumor_count += 1

    logger.info(
        "Generated %d synthetic samples -> %s "
        "(%d with tumour, %d without)",
        args.n, out_dir, tumor_count, args.n - tumor_count,
    )
    logger.info("Images -> %s", img_dir)
    logger.info("Masks  -> %s", msk_dir)


if __name__ == "__main__":
    main()
