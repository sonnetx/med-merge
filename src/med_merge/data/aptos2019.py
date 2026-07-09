"""APTOS 2019 diabetic retinopathy dataset (5-class ordinal)."""

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

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


class APTOS2019Dataset(Dataset):
    """APTOS 2019 Blindness Detection (diabetic retinopathy grading).

    5-class ordinal scale. Fixed 80/10/10 split from training set
    (test set has no labels in competition format).
    Kaggle source: aptos2019-blindness-detection.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        split_seed: int = 42,
        split_ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
        **kwargs,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self._image_paths: list[Path] = []
        self._labels: list[int] = []
        self._load_data(split_seed, split_ratios)

    def _load_data(self, seed: int, ratios: tuple[float, ...]) -> None:
        csv_path = self.data_dir / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"APTOS 2019 train.csv not found at {csv_path}. "
                "Download from Kaggle: kaggle competitions download -c aptos2019-blindness-detection"
            )

        df = pd.read_csv(csv_path)
        img_dir = self.data_dir / "train_images"

        # Fixed split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(df))
        n = len(df)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        split_map = {
            "train": indices[:train_end],
            "validation": indices[train_end:val_end],
            "test": indices[val_end:],
        }
        selected = split_map.get(self.split, split_map["train"])

        for idx in selected:
            row = df.iloc[idx]
            self._image_paths.append(img_dir / f"{row['id_code']}.png")
            self._labels.append(int(row["diagnosis"]))

        logger.info(
            f"APTOS 2019 {self.split}: {len(self._image_paths)} images, "
            f"class distribution: {np.bincount(self._labels, minlength=5).tolist()}"
        )

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self._image_paths[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self._labels[idx], dtype=torch.long)
        return image, label

    @property
    def num_classes(self) -> int:
        return 5

    @property
    def task_type(self) -> str:
        return "ordinal"

    @property
    def class_names(self) -> list[str]:
        return CLASS_NAMES
