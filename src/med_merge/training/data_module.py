"""DataModule: encapsulates dataset creation, transforms, and DataLoaders."""

from __future__ import annotations

from functools import cached_property
from typing import Optional

import torch
from torch.utils.data import DataLoader

from med_merge.config.schema import DatasetConfig, ModelConfig, TrainingConfig
from med_merge.data.registry import build_dataset
from med_merge.data.transforms import (
    get_aggressive_train_transform,
    get_eval_transform,
    get_train_transform,
    norm_key_for_backbone,
)
from med_merge.training.samplers import get_balanced_sampler


class DataModule:
    """Wraps dataset creation, transforms, and DataLoaders."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        model_config: Optional[ModelConfig] = None,
    ):
        self.dataset_config = dataset_config
        self.training_config = training_config
        self._norm_key = norm_key_for_backbone(
            model_config.backbone if model_config else "openai/clip-vit-base-patch16"
        )
        self._image_size = dataset_config.image_size

    def _loader_kwargs(self) -> dict:
        """Extra kwargs forwarded to the dataset loader (csv_path, etc.)."""
        kw: dict = {}
        if self.dataset_config.csv_path:
            kw["csv_path"] = self.dataset_config.csv_path
        return kw

    @cached_property
    def train_dataset(self):
        if self.dataset_config.augmentation == "aggressive":
            transform = get_aggressive_train_transform(self._image_size, self._norm_key)
        else:
            transform = get_train_transform(self._image_size, self._norm_key)
        return build_dataset(
            self.dataset_config.name,
            self.dataset_config.data_dir,
            split="train",
            transform=transform,
            **self._loader_kwargs(),
        )

    @cached_property
    def val_dataset(self):
        transform = get_eval_transform(self._image_size, self._norm_key)
        return build_dataset(
            self.dataset_config.name,
            self.dataset_config.data_dir,
            split="validation",
            transform=transform,
            **self._loader_kwargs(),
        )

    @cached_property
    def train_loader(self) -> DataLoader:
        tc = self.training_config
        sampler = None
        shuffle = True
        if tc.use_balanced_sampler:
            sampler = get_balanced_sampler(self.train_dataset)
            shuffle = False
        return DataLoader(
            self.train_dataset,
            batch_size=tc.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            drop_last=True,
        )

    @cached_property
    def val_loader(self) -> DataLoader:
        tc = self.training_config
        return DataLoader(
            self.val_dataset,
            batch_size=tc.batch_size * 2,
            shuffle=False,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
        )

    @property
    def class_weights(self) -> Optional[torch.Tensor]:
        if self.training_config.use_class_weights and hasattr(self.train_dataset, "get_class_weights"):
            return self.train_dataset.get_class_weights()
        return None
