"""NCT-CRC-HE100K colon histopathology dataset (9-class tissue classification)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class NCTCRCDataset(Dataset):
    """NCT-CRC-HE100K colon histopathology tissue classification.

    Train: NCT-CRC-HE-100K (100K patches, 224x224)
    Test: CRC-VAL-HE-7K (7,180 patches)
    Images organized in class-named subdirectories.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        val_fraction: float = 0.1,
        split_seed: int = 42,
        **kwargs,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self._image_paths: list[Path] = []
        self._labels: list[int] = []
        self._load_data(val_fraction, split_seed)

    def _load_data(self, val_fraction: float, seed: int) -> None:
        if self.split == "test":
            # CRC-VAL-HE-7K is the test set
            base = self.data_dir / "CRC-VAL-HE-7K"
        else:
            # NCT-CRC-HE-100K for train/validation
            base = self.data_dir / "NCT-CRC-HE-100K"

        if not base.exists():
            # Try alternate names
            for alt in ["NCT-CRC-HE-100K-NONORM", "NCT-CRC-HE100K"]:
                alt_path = self.data_dir / alt
                if alt_path.exists():
                    base = alt_path
                    break
            else:
                if self.split != "test":
                    raise FileNotFoundError(
                        f"NCT-CRC training data not found at {base}. "
                        "Download from Zenodo: https://zenodo.org/record/1214456"
                    )

        # Load images from class subdirectories
        all_paths = []
        all_labels = []
        for class_name in CLASS_NAMES:
            class_dir = base / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.tif")):
                all_paths.append(img_path)
                all_labels.append(CLASS_TO_IDX[class_name])
            # Also check for .png
            for img_path in sorted(class_dir.glob("*.png")):
                all_paths.append(img_path)
                all_labels.append(CLASS_TO_IDX[class_name])

        if self.split == "test":
            self._image_paths = all_paths
            self._labels = all_labels
        else:
            # Split NCT-CRC-HE-100K into train/validation
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(all_paths))
            val_size = int(len(all_paths) * val_fraction)

            if self.split == "validation":
                selected = indices[:val_size]
            else:
                selected = indices[val_size:]

            self._image_paths = [all_paths[i] for i in selected]
            self._labels = [all_labels[i] for i in selected]

        logger.info(
            f"NCT-CRC {self.split}: {len(self._image_paths)} images, "
            f"classes: {np.bincount(self._labels, minlength=9).tolist()}"
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
        return 9

    @property
    def task_type(self) -> str:
        return "multiclass"

    @property
    def class_names(self) -> list[str]:
        return CLASS_NAMES
