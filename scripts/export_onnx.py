#!/usr/bin/env python3
"""
scripts/export_onnx.py
=======================
Export a trained PyTorch model to ONNX format for deployment.

Usage
-----
    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint outputs/checkpoints/best_model.pth
    python scripts/export_onnx.py --output outputs/model.onnx --opset 17
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.factory import build_model, get_device, load_checkpoint

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("export_onnx")


def parse_args():
    p = argparse.ArgumentParser(description="Export model to ONNX")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output",     default="outputs/model.onnx")
    p.add_argument("--opset",      type=int, default=17)
    p.add_argument("--dynamic",    action="store_true",
                   help="Use dynamic batch/spatial axes")
    return p.parse_args()


def main():
    args = parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["num_classes"] = cfg["data"].get("num_classes", 1)
    cfg["model"]["in_channels"] = cfg["data"].get("num_channels", 1)
    h, w = cfg["data"].get("image_size", [256, 256])

    device = get_device("cpu")   # ONNX export on CPU
    model  = build_model(cfg["model"]).to(device)
    model.eval()

    ckpt = args.checkpoint or cfg["inference"].get("model_path", "outputs/checkpoints/best_model.pth")
    ckpt_path = ROOT / ckpt
    if ckpt_path.exists():
        load_checkpoint(model, str(ckpt_path), device)
        logger.info("Loaded checkpoint: %s", ckpt_path)
    else:
        logger.warning("Checkpoint not found: %s — exporting with random weights", ckpt_path)

    # Dummy input
    C      = cfg["model"]["in_channels"]
    dummy  = torch.randn(1, C, h, w)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        model, dummy, str(out_path),
        opset_version   = args.opset,
        input_names     = ["input"],
        output_names    = ["output"],
        dynamic_axes    = dynamic_axes,
        do_constant_folding=True,
    )
    logger.info("ONNX model exported -> %s", out_path)

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed.")
    except ImportError:
        logger.info("Install onnx to verify: pip install onnx")


if __name__ == "__main__":
    main()
