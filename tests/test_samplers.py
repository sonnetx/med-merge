"""Tests for balanced sampling, including the ``Subset`` unwrapping path.

Regression coverage for a bug where a ``max_train_samples`` cap wrapped the dataset in a
``torch.utils.data.Subset``, which does not forward ``get_sample_weights``. Balanced
sampling then raised ``AttributeError`` mid-run, and callers that only logged the failure
silently trained on fewer tasks than requested.
"""

import pytest
import torch
from torch.utils.data import Subset

from med_merge.training.samplers import get_balanced_sampler


class _WeightedDataset:
    """Minimal dataset exposing per-sample weights, weight i == i."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def get_sample_weights(self) -> torch.Tensor:
        return torch.arange(self.n, dtype=torch.float)


class _UnweightedDataset:
    def __len__(self) -> int:
        return 4

    def get_sample_weights(self):
        return None


def test_balanced_sampler_uses_dataset_weights():
    sampler = get_balanced_sampler(_WeightedDataset(5))
    assert sampler.weights.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert sampler.num_samples == 5


def test_balanced_sampler_unwraps_subset():
    """A capped dataset must draw the weights of its retained indices, in order."""
    sampler = get_balanced_sampler(Subset(_WeightedDataset(10), [9, 3, 5]))
    assert sampler.weights.tolist() == [9.0, 3.0, 5.0]
    assert sampler.num_samples == 3


def test_balanced_sampler_unwraps_nested_subset():
    """Indices compose through nesting: outer [2, 0] selects inner [5, 9]."""
    nested = Subset(Subset(_WeightedDataset(10), [9, 3, 5]), [2, 0])
    assert get_balanced_sampler(nested).weights.tolist() == [5.0, 9.0]


def test_balanced_sampler_rejects_dataset_without_weights():
    with pytest.raises(ValueError, match="sample weights"):
        get_balanced_sampler(_UnweightedDataset())

    with pytest.raises(ValueError, match="sample weights"):
        get_balanced_sampler(Subset(_UnweightedDataset(), [0, 1]))
