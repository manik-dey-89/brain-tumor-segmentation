"""Evaluation entry point."""
import argparse, sys, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("evaluate")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate segmentation model")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--image-dir",  default="data/raw/images")
    p.add_argument("--mask-dir",   default="data/raw/masks")
    p.add_argument("--output-dir", default="outputs/predictions")
    return p.parse_args()


def main():
    args = parse_args()
    from inference.engine import SegmentationEngine
    engine = SegmentationEngine.from_config(args.config)
    logger.info("Engine loaded. Point --image-dir and --mask-dir to your data.")


if __name__ == "__main__":
    main()
