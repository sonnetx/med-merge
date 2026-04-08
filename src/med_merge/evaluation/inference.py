"""Shared inference runner — used by Evaluator and hyperopt."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


@dataclass
class InferenceResult:
    """Raw model outputs from an inference pass."""

    logits: np.ndarray
    labels: np.ndarray
    dataset_name: str = ""
    task_type: str = ""


class InferenceRunner:
    """Run model inference on a DataLoader, returning raw logits and labels."""

    def __init__(
        self,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.device = device
        self.use_amp = use_amp

    @torch.no_grad()
    def run(self, model: nn.Module, loader: DataLoader) -> InferenceResult:
        """Forward-pass through the entire loader."""
        model = model.to(self.device).eval()
        all_logits = []
        all_labels = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            with autocast(enabled=self.use_amp):
                logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        return InferenceResult(
            logits=torch.cat(all_logits).numpy(),
            labels=torch.cat(all_labels).numpy(),
        )
