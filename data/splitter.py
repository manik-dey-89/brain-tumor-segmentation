"""Dataset splitting utilities."""
import random
from pathlib import Path
from typing import Tuple, List


def split_dataset(
    image_dir: str,
    mask_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path], List[Path], List[Path], List[Path]]:
    """
    Split image/mask pairs into train/val/test sets.

    Returns
    -------
    train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks
    """
    img_dir = Path(image_dir)
    msk_dir = Path(mask_dir)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    pairs = [(p, msk_dir / p.name) for p in imgs if (msk_dir / p.name).exists()]

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]

    def unzip(lst):
        if not lst:
            return [], []
        a, b = zip(*lst)
        return list(a), list(b)

    ti, tm = unzip(train)
    vi, vm = unzip(val)
    sti, stm = unzip(test)
    return ti, tm, vi, vm, sti, stm
