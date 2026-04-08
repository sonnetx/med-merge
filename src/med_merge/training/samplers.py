"""Balanced sampling for imbalanced datasets."""

from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler


def get_balanced_sampler(dataset) -> WeightedRandomSampler:
    """Create a weighted random sampler using dataset's sample weights.

    Each epoch sees approximately equal representation per class.
    """
    weights = dataset.get_sample_weights()
    if weights is None:
        raise ValueError("Dataset does not provide sample weights for balanced sampling")

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
