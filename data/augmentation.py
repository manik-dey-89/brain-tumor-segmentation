"""Data augmentation transforms — used during training only."""
import numpy as np


def get_train_transforms(cfg: dict = None):
    """Return training augmentation pipeline (identity if albumentations unavailable)."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        aug = cfg.get("augmentation", {}) if cfg else {}
        transforms = []
        if aug.get("random_horizontal_flip", 0) > 0:
            transforms.append(A.HorizontalFlip(p=aug["random_horizontal_flip"]))
        if aug.get("random_vertical_flip", 0) > 0:
            transforms.append(A.VerticalFlip(p=aug["random_vertical_flip"]))
        if aug.get("random_rotation_degrees", 0) > 0:
            transforms.append(A.Rotate(limit=aug["random_rotation_degrees"], p=0.5))
        transforms.append(ToTensorV2())
        return A.Compose(transforms)
    except ImportError:
        return _IdentityTransform()


def get_val_transforms(cfg: dict = None):
    """Return validation transforms (no augmentation)."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        return A.Compose([ToTensorV2()])
    except ImportError:
        return _IdentityTransform()


class _IdentityTransform:
    def __call__(self, image=None, mask=None, **kw):
        import torch
        result = {}
        if image is not None:
            arr = np.array(image, dtype=np.float32)
            result["image"] = torch.from_numpy(arr)
        if mask is not None:
            arr = np.array(mask, dtype=np.float32)
            result["mask"] = torch.from_numpy(arr)
        return result
