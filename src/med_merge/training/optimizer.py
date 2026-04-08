"""Optimizer and scheduler factories."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from med_merge.config.schema import TrainingConfig


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    name = config.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    total_steps: int,
) -> LRScheduler:
    """Build LR scheduler from config."""
    warmup_steps = int(total_steps * config.warmup_fraction)

    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / max(total_steps, 1),
            anneal_strategy="cos",
        )
    elif config.scheduler == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )
    elif config.scheduler == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
