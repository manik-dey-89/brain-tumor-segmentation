"""
Model factory – instantiates the correct architecture from a config dict.

Usage
-----
    from models.factory import build_model
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg["model"])
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.resunet import ResUNet

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type] = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "resunet": ResUNet,
}


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    """
    Build a segmentation model from the ``model`` section of config.yaml.

    Parameters
    ----------
    model_cfg : dict
        Contents of the ``model:`` block from config.yaml.

    Returns
    -------
    nn.Module
        Instantiated (and optionally GPU-placed) model.
    """
    arch = model_cfg.get("architecture", "attention_unet").lower()
    if arch not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture: {arch!r}. "
            f"Choose from {list(_REGISTRY.keys())}"
        )

    cls = _REGISTRY[arch]
    kwargs: dict[str, Any] = {
        "in_channels":  model_cfg.get("in_channels", 1),
        "num_classes":  model_cfg.get("num_classes", 1),
        "base_filters": model_cfg.get("base_filters", 64),
        "depth":        model_cfg.get("depth", 4),
        "bilinear":     model_cfg.get("bilinear", False),
        "norm_type":    model_cfg.get("norm_type", "batch"),
        "num_groups":   model_cfg.get("num_groups", 8),
        "dropout_rate": model_cfg.get("dropout_rate", 0.3),
    }

    # ResUNet-specific pretrained encoder args
    if arch == "resunet":
        kwargs["use_pretrained_encoder"] = model_cfg.get("use_pretrained_encoder", False)
        kwargs["pretrained_encoder"]     = model_cfg.get("pretrained_encoder", "resnet34")
        kwargs["freeze_encoder"]         = model_cfg.get("freeze_encoder", False)

    model = cls(**kwargs)

    logger.info(
        "Built %s | params=%s | architecture=%s",
        cls.__name__,
        f"{model.count_parameters():,}",
        arch,
    )
    return model


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve the compute device from a string.

    Parameters
    ----------
    device_str : "auto" | "cuda" | "cpu" | "cuda:0" etc.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info("Using device: %s", device)
    return device


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load model weights from a checkpoint file.

    Parameters
    ----------
    model           : the model to load weights into
    checkpoint_path : path to the .pth checkpoint file
    device          : target device
    strict          : strict key matching (True by default)

    Returns
    -------
    dict containing epoch, best_metric, and any other saved metadata.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both bare state-dicts and wrapped checkpoints
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        meta = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        state_dict = checkpoint
        meta = {}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    logger.info("Loaded checkpoint from %s", checkpoint_path)
    return meta


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_value: float,
    path: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint."""
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_metric": metric_value,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info("Checkpoint saved -> %s  (metric=%.4f)", path, metric_value)
