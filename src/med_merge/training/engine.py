"""Pure training loop: epoch iteration, gradient stepping, validation."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class TrainingEngine:
    """Minimal training engine — owns no data or model, just runs loops."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        scaler: GradScaler,
        *,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

    def train_epoch(self, loader: DataLoader, epoch: int = 0) -> float:
        """Train one epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
        metric_fn: Optional[Callable] = None,
        task_type: str = "multiclass",
        class_names: Optional[list[str]] = None,
        epoch: int = 0,
    ) -> dict[str, float]:
        """Run validation. Returns metrics dict (always includes 'val_loss')."""
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        for images, labels in tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            n_batches += 1

        metrics: dict[str, float] = {"val_loss": total_loss / max(n_batches, 1)}

        if metric_fn is not None:
            logits_np = torch.cat(all_logits).numpy()
            labels_np = torch.cat(all_labels).numpy()
            metrics.update(
                metric_fn(logits_np, labels_np, task_type=task_type, class_names=class_names or [])
            )

        return metrics
