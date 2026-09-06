"""Brain tumour segmentation dataset — used during training only."""
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class BrainTumorDataset(Dataset):
    def __init__(self, image_paths: list, mask_paths: list,
                 transform: Optional[Callable] = None,
                 preprocessor=None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import cv2
        img = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            result = self.transform(image=img, mask=mask)
            img = result["image"]
            mask = result["mask"]
        else:
            img = torch.from_numpy(img).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


def create_dataloaders(train_dataset: Dataset, val_dataset: Dataset,
                       batch_size: int = 8, num_workers: int = 0,
                       pin_memory: bool = False) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)
    return train_loader, val_loader
