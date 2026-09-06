"""CLI prediction entry point."""
import argparse, sys, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("predict")


def parse_args():
    p = argparse.ArgumentParser(description="Run inference on a single MRI image")
    p.add_argument("image",         help="Path to input MRI image")
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--gt-mask",     default=None, help="Optional ground-truth mask")
    p.add_argument("--output-dir",  default="outputs/predictions")
    return p.parse_args()


def main():
    args = parse_args()
    from inference.engine import SegmentationEngine
    engine = SegmentationEngine.from_config(args.config)
    result = engine.predict_file(args.image, gt_mask_path=args.gt_mask)
    logger.info("Prediction complete: tumour_detected=%s confidence=%.3f",
                result.get("tumor_detected"), result.get("confidence", 0))


if __name__ == "__main__":
    main()
