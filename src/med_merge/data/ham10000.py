"""HAM10000 dermoscopy dataset (7-class skin lesion classification)."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# The HuggingFace copy (marmal88/skin_cancer) spells diagnoses out rather than using the
# HAM10000 abbreviations, so an unmapped lookup silently collapses every image onto class 0.
DX_ALIASES = {
    "actinic_keratoses": "akiec",
    "basal_cell_carcinoma": "bcc",
    "benign_keratosis-like_lesions": "bkl",
    "dermatofibroma": "df",
    "melanoma": "mel",
    "melanocytic_Nevi": "nv",
    "vascular_lesions": "vasc",
}


def _dx_to_index(dx: str) -> int:
    """Map a diagnosis string to its class index, accepting either spelling."""
    key = DX_ALIASES.get(dx, dx)
    if key not in CLASS_TO_IDX:
        raise ValueError(
            f"unrecognized HAM10000 diagnosis {dx!r}; known values are "
            f"{sorted(CLASS_TO_IDX)} or {sorted(DX_ALIASES)}"
        )
    return CLASS_TO_IDX[key]


class HAM10000Dataset(Dataset):
    """HAM10000 skin lesion classification dataset.

    Loaded from HuggingFace `marmal88/skin_cancer`.
    Splits by lesion_id to prevent data leakage.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        split_seed: int = 42,
        split_ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self._images: list = []
        self._labels: list[int] = []
        self._load_data(split_seed, split_ratios)

    def _load_data(self, seed: int, ratios: tuple[float, ...]) -> None:
        """Load from HuggingFace datasets and create lesion-aware splits."""
        from datasets import load_dataset

        ds = load_dataset("marmal88/skin_cancer", split="train")

        # Build lesion_id -> indices mapping for no-leakage splits
        # The HF dataset has 'dx' (diagnosis) and 'lesion_id' columns
        lesion_to_indices: dict[str, list[int]] = {}
        for i, row in enumerate(ds):
            lid = row.get("lesion_id", str(i))
            lesion_to_indices.setdefault(lid, []).append(i)

        # Stratified split by lesion
        rng = np.random.RandomState(seed)
        lesion_ids = list(lesion_to_indices.keys())
        rng.shuffle(lesion_ids)

        n = len(lesion_ids)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        split_map = {"train": lesion_ids[:train_end],
                     "validation": lesion_ids[train_end:val_end],
                     "test": lesion_ids[val_end:]}

        selected_lesions = set(split_map.get(self.split, split_map["train"]))
        indices = []
        for lid in selected_lesions:
            indices.extend(lesion_to_indices[lid])

        for idx in indices:
            row = ds[idx]
            self._images.append(row["image"])
            dx = row["dx"]
            self._labels.append(_dx_to_index(dx))

        logger.info(
            f"HAM10000 {self.split}: {len(self._images)} images, "
            f"class distribution: {Counter(self._labels)}"
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._images[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self._labels[idx], dtype=torch.long)
        return image, label

    @property
    def num_classes(self) -> int:
        return 7

    @property
    def task_type(self) -> str:
        return "multiclass"

    @property
    def class_names(self) -> list[str]:
        return CLASS_NAMES

    def get_sample_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for balanced sampling."""
        counts = Counter(self._labels)
        total = len(self._labels)
        weights = []
        for label in self._labels:
            weights.append(total / (len(counts) * counts[label]))
        return torch.tensor(weights, dtype=torch.float)

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for loss function."""
        counts = Counter(self._labels)
        total = len(self._labels)
        weights = []
        for i in range(self.num_classes):
            c = counts.get(i, 1)
            weights.append(total / (self.num_classes * c))
        return torch.tensor(weights, dtype=torch.float)
