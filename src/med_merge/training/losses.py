"""Loss functions for different task types."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def get_loss_function(
    task_type: str,
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> nn.Module:
    """Return appropriate loss function for the task type.

    Args:
        task_type: One of 'multiclass', 'multilabel', 'binary', 'ordinal'.
        class_weights: Optional inverse-frequency class weights.
        device: Device for weight tensors.

    Returns:
        Loss function module.
    """
    if task_type in ("multiclass", "ordinal"):
        if class_weights is not None:
            class_weights = class_weights.to(device)
        return nn.CrossEntropyLoss(weight=class_weights)

    elif task_type in ("multilabel", "binary"):
        return nn.BCEWithLogitsLoss()

    else:
        raise ValueError(f"Unknown task type: {task_type}")
