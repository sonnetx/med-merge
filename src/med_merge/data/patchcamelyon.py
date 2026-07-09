"""PatchCamelyon histopathology dataset (binary breast cancer metastasis)."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PatchCamelyonDataset(Dataset):
    """PatchCamelyon binary classification dataset.

    Loaded from HuggingFace `1aurent/PatchCamelyon`.
    96x96 patches, pre-split into train/validation/test.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        **kwargs,
    ):
        self.split = split
        self.transform = transform
        self._dataset = None
        self._load_data()

    def _load_data(self) -> None:
        from datasets import load_dataset

        split_name = self.split
        if split_name == "validation":
            split_name = "valid"
        self._dataset = load_dataset("1aurent/PatchCamelyon", split=split_name)
        logger.info(f"PatchCamelyon {self.split}: {len(self._dataset)} images")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._dataset[idx]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.float).unsqueeze(0)
        return image, label

    @property
    def num_classes(self) -> int:
        return 1  # single-logit binary head (BCEWithLogits), matches config convention

    @property
    def task_type(self) -> str:
        return "binary"

    @property
    def class_names(self) -> list[str]:
        return ["negative", "positive"]
