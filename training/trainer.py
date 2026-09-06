"""Training loop — not used at serve time."""
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module,
                 optimizer: torch.optim.Optimizer, device: torch.device,
                 cfg: dict = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg or {}

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for imgs, masks in loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, masks)
                total_loss += loss.item()
        return total_loss / max(len(loader), 1)
