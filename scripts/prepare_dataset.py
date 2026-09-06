#!/usr/bin/env python3
"""
scripts/prepare_dataset.py
===========================
Prepare and validate a raw dataset for training.

Tasks
-----
1. Discover all image–mask pairs in data/raw/images + data/raw/masks
2. Validate every mask (correct labels, matching spatial size)
3. Resize + normalise images and save to data/processed/
4. Generate a dataset manifest CSV
5. Print a summary report

Usage
-----
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --raw-dir data/raw --out-dir data/processed
    python scripts/prepare_dataset.py --validate-only
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.preprocessing import Preprocessor
from data.splitter import discover_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prepare_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset preparation script")
    p.add_argument("--config",        default="configs/config.yaml")
    p.add_argument("--raw-dir",       default=None, help="Override raw data directory")
    p.add_argument("--out-dir",       default=None, help="Override processed output directory")
    p.add_argument("--validate-only", action="store_true", help="Only validate, do not process")
    p.add_argument("--max-samples",   type=int, default=None, help="Limit number of samples")
    return p.parse_args()


def validate_pair(img_path: Path, msk_path: Path, preprocessor: Preprocessor) -> dict:
    """Validate one image–mask pair. Returns a report dict."""
    issues = []

    try:
        img = preprocessor.load_image(img_path)
        msk = preprocessor.load_mask(msk_path)
    except Exception as e:
        return {"status": "error", "issue": str(e)}

    # Spatial size match
    img_hw = img.shape[:2]
    msk_hw = msk.shape[:2]
    if img_hw != msk_hw:
        issues.append(f"Size mismatch: image={img_hw}, mask={msk_hw}")

    # Mask value check
    unique = np.unique(msk)
    unexpected = [v for v in unique if v < 0]
    if unexpected:
        issues.append(f"Negative mask values: {unexpected}")

    # Empty image check
    if img.max() == img.min():
        issues.append("Image appears blank (uniform intensity)")

    status = "ok" if not issues else "warning"
    return {
        "status":      status,
        "image_shape": img.shape,
        "mask_shape":  msk.shape,
        "mask_unique": unique.tolist(),
        "tumor_present": bool(msk.max() > 0),
        "issues":      issues,
    }


def process_pair(
    img_path: Path,
    msk_path: Path,
    preprocessor: Preprocessor,
    out_img_dir: Path,
    out_msk_dir: Path,
) -> None:
    """Preprocess and save one image–mask pair."""
    img = preprocessor.load_image(img_path)
    msk = preprocessor.load_mask(msk_path)

    proc_img = preprocessor.process_image(img)   # (C, H, W) float32
    proc_msk = preprocessor.process_mask(msk)    # (H, W) int64

    # Save as PNG
    out_img = out_img_dir / img_path.name
    out_msk = out_msk_dir / msk_path.name

    # Image: take first channel, scale to uint8
    img_save = proc_img[0] if proc_img.shape[0] == 1 else proc_img.mean(axis=0)
    img_save = ((img_save - img_save.min()) /
                (img_save.max() - img_save.min() + 1e-8) * 255).astype(np.uint8)
    cv2.imwrite(str(out_img), img_save)

    # Mask: 0/255
    msk_save = (proc_msk * 255).astype(np.uint8)
    cv2.imwrite(str(out_msk), msk_save)


def main() -> None:
    args = parse_args()

    config_path = ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    prep_cfg = {**data_cfg, **cfg.get("preprocessing", {})}

    raw_dir = Path(args.raw_dir or (ROOT / data_cfg["raw_dir"]))
    out_dir = Path(args.out_dir or (ROOT / data_cfg["processed_dir"]))

    images_dir = raw_dir / data_cfg.get("images_subdir", "images")
    masks_dir  = raw_dir / data_cfg.get("masks_subdir",  "masks")

    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    preprocessor = Preprocessor(prep_cfg)
    img_paths, msk_paths = discover_pairs(images_dir, masks_dir)

    if not img_paths:
        logger.error("No image–mask pairs found.")
        sys.exit(1)

    if args.max_samples:
        img_paths = img_paths[:args.max_samples]
        msk_paths = msk_paths[:args.max_samples]

    logger.info("Found %d pairs. Validating…", len(img_paths))

    # Validation pass
    manifest = []
    errors, warnings, ok_count = 0, 0, 0
    tumor_count = 0

    for img_p, msk_p in zip(img_paths, msk_paths):
        report = validate_pair(img_p, msk_p, preprocessor)
        manifest.append({
            "image":         img_p.name,
            "mask":          msk_p.name,
            "status":        report["status"],
            "tumor_present": report.get("tumor_present", False),
            "issues":        "; ".join(report.get("issues", [])),
        })
        if report["status"] == "error":
            errors += 1
            logger.warning("  ERROR  %s: %s", img_p.name, report.get("issue", ""))
        elif report["status"] == "warning":
            warnings += 1
            for iss in report.get("issues", []):
                logger.warning("  WARN   %s: %s", img_p.name, iss)
        else:
            ok_count += 1
        if report.get("tumor_present"):
            tumor_count += 1

    logger.info("Validation: %d OK | %d warnings | %d errors", ok_count, warnings, errors)
    logger.info("Tumour present in %d / %d images (%.1f%%)",
                tumor_count, len(img_paths), 100 * tumor_count / len(img_paths))

    # Save manifest CSV
    manifest_path = ROOT / "data" / "dataset_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    logger.info("Manifest saved -> %s", manifest_path)

    if args.validate_only:
        return

    # Processing pass
    out_img_dir = out_dir / "images"
    out_msk_dir = out_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing and saving to %s …", out_dir)
    processed = 0
    for img_p, msk_p in zip(img_paths, msk_paths):
        try:
            process_pair(img_p, msk_p, preprocessor, out_img_dir, out_msk_dir)
            processed += 1
            if processed % 50 == 0:
                logger.info("  %d / %d processed …", processed, len(img_paths))
        except Exception as exc:
            logger.error("  Failed %s: %s", img_p.name, exc)

    logger.info("Done. Processed %d images -> %s", processed, out_dir)


if __name__ == "__main__":
    main()
