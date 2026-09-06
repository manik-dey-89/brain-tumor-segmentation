"""Training entry point."""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description="Train brain tumour segmentation model")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    import yaml, torch
    from models.factory import build_model, get_device
    from training.losses import build_loss
    from training.trainer import Trainer

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("inference", {}).get("device", "auto"))
    cfg["model"]["num_classes"] = cfg["data"].get("num_classes", 1)
    cfg["model"]["in_channels"] = cfg["data"].get("num_channels", 1)
    model = build_model(cfg["model"]).to(device)

    criterion = build_loss(cfg.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"].get("learning_rate", 1e-4),
                                  weight_decay=cfg["training"].get("weight_decay", 1e-5))
    trainer = Trainer(model, criterion, optimizer, device, cfg)
    logger.info("Trainer ready — implement data loading to begin training.")


if __name__ == "__main__":
    main()
