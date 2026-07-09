"""Fitzpatrick17k fairness evaluation dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Fitzpatrick skin type groups for fairness analysis
SKIN_TYPE_GROUPS = {
    "light": [1, 2],      # Fitzpatrick I-II
    "medium": [3, 4],     # Fitzpatrick III-IV
    "dark": [5, 6],       # Fitzpatrick V-VI
}


class Fitzpatrick17kDataset(Dataset):
    """Fitzpatrick17k dataset for out-of-distribution fairness evaluation.

    16,577 clinical images with Fitzpatrick skin type annotations (I-VI).
    Used to evaluate fairness of merged models across skin tone groups.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "test",
        transform=None,
        **kwargs,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self._image_paths: list[Path] = []
        self._labels: list[int] = []
        self._skin_types: list[int] = []
        self._load_data()

    def _load_data(self) -> None:
        csv_path = self.data_dir / "fitzpatrick17k.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Fitzpatrick17k CSV not found at {csv_path}. "
                "Clone from: https://github.com/mattgroh/fitzpatrick17k"
            )

        df = pd.read_csv(csv_path)
        img_dir = self.data_dir / "images"

        for _, row in df.iterrows():
            # Check image exists
            img_name = row.get("hasher", row.get("md5hash", ""))
            img_path = img_dir / f"{img_name}.jpg"
            if not img_path.exists():
                continue

            skin_type = row.get("fitzpatrick_skin_type", row.get("fitzpatrick", -1))
            if not isinstance(skin_type, (int, float)) or np.isnan(skin_type):
                continue

            self._image_paths.append(img_path)
            self._skin_types.append(int(skin_type))
            # Use three_partition_label or label as the classification target
            label = row.get("three_partition_label", row.get("label", 0))
            if isinstance(label, str):
                label = hash(label) % 3  # coarse grouping
            self._labels.append(int(label))

        logger.info(
            f"Fitzpatrick17k: {len(self._image_paths)} images loaded, "
            f"skin type distribution: {np.bincount(self._skin_types, minlength=7)[1:].tolist()}"
        )

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self._image_paths[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self._labels[idx], dtype=torch.long)
        return image, label

    def get_skin_types(self) -> list[int]:
        """Return Fitzpatrick skin types for all samples."""
        return self._skin_types

    def get_skin_type_groups(self) -> list[str]:
        """Return skin type group ('light', 'medium', 'dark') for each sample."""
        groups = []
        for st in self._skin_types:
            for group_name, types in SKIN_TYPE_GROUPS.items():
                if st in types:
                    groups.append(group_name)
                    break
            else:
                groups.append("unknown")
        return groups

    @property
    def num_classes(self) -> int:
        return len(set(self._labels)) if self._labels else 0

    @property
    def task_type(self) -> str:
        return "multiclass"

    @property
    def class_names(self) -> list[str]:
        return []  # varies by label scheme
