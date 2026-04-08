"""Classification head variants."""

from __future__ import annotations

import torch.nn as nn


class LinearHead(nn.Module):
    """Simple linear classification head."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


class MLPHead(nn.Module):
    """Two-layer MLP classification head with LayerNorm and GELU."""

    def __init__(self, in_features: int, hidden_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def create_head(
    head_type: str,
    in_features: int,
    num_classes: int,
    *,
    hidden_features: int | None = None,
) -> nn.Module:
    """Factory for classification heads."""
    if head_type == "linear":
        return LinearHead(in_features, num_classes)
    elif head_type == "mlp":
        hf = hidden_features or in_features
        return MLPHead(in_features, hf, num_classes)
    else:
        raise ValueError(f"Unknown head type: {head_type!r}")
