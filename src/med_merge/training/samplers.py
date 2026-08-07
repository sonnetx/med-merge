"""Balanced sampling for imbalanced datasets."""

from __future__ import annotations

import torch
from torch.utils.data import Subset, WeightedRandomSampler


def get_balanced_sampler(dataset) -> WeightedRandomSampler:
    """Create a weighted random sampler using dataset's sample weights.

    Each epoch sees approximately equal representation per class. A ``Subset`` (as
    produced by a ``max_train_samples`` cap) is unwrapped so the weights are taken from
    the underlying dataset and restricted to the retained indices.
    """
    indices = None
    while isinstance(dataset, Subset):
        indices = [dataset.indices[i] for i in indices] if indices is not None else list(dataset.indices)
        dataset = dataset.dataset

    weights = dataset.get_sample_weights()
    if weights is None:
        raise ValueError("Dataset does not provide sample weights for balanced sampling")
    if indices is not None:
        weights = weights[torch.as_tensor(indices, dtype=torch.long)]

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
