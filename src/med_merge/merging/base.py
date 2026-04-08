"""Abstract base class for all merging methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.task_vector import TaskVector


class BaseMerger(ABC):
    """Interface for all merging methods."""

    def __init__(
        self,
        pretrained_state_dict: dict[str, torch.Tensor],
        config: MergingConfig,
    ):
        self.pretrained = pretrained_state_dict
        self.config = config

    @abstractmethod
    def merge(self, task_vectors: dict[str, TaskVector], **kwargs) -> dict[str, torch.Tensor]:
        """Merge task vectors into a single encoder state dict.

        Args:
            task_vectors: Mapping of dataset_name -> TaskVector.

        Returns:
            Merged encoder state dict.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def hyperparameters(self) -> dict:
        """Return method-specific hyperparameters for logging."""
        return {}
