"""Base classes for all med-merge datasets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MedMergeDataset(ABC, Dataset):
    """Abstract base for all medical imaging datasets."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @property
    @abstractmethod
    def task_type(self) -> str:
        ...

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        ...

    def get_sample_weights(self) -> Optional[torch.Tensor]:
        return None

    def get_class_weights(self) -> Optional[torch.Tensor]:
        return None


class ImageListDataset(MedMergeDataset):
    """Concrete dataset backed by a list of (image_source, label) samples.

    ``image_source`` can be a ``Path``, a ``PIL.Image``, or a numpy array.
    This class handles the common ``__getitem__`` logic shared by all
    med-merge datasets — individual loaders only need to produce the
    sample list and metadata.
    """

    def __init__(
        self,
        samples: list[tuple[Any, Any]],
        transform,
        *,
        dataset_name: str,
        dataset_task_type: str,
        dataset_num_classes: int,
        dataset_class_names: list[str],
    ):
        self._samples = samples
        self.transform = transform
        self._name = dataset_name
        self._task_type = dataset_task_type
        self._num_classes = dataset_num_classes
        self._class_names = dataset_class_names

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self._samples)
        for offset in range(n):
            i = (idx + offset) % n
            source, label = self._samples[i]
            try:
                image = self._load_image(source)
                break
            except (FileNotFoundError, OSError) as e:
                if offset == 0:
                    logger.warning(f"Skipping missing image {source!r}: {e}")
                continue
        else:
            raise FileNotFoundError("No readable images left in dataset")

        if self.transform is not None:
            image = self.transform(image)
        label_tensor = self._encode_label(label)
        return image, label_tensor

    # -- property overrides --------------------------------------------------

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _load_image(source: Any) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")
        raise TypeError(f"Cannot load image from {type(source)}")

    def _encode_label(self, label: Any) -> torch.Tensor:
        if self._task_type == "multilabel":
            return torch.tensor(label, dtype=torch.float)
        elif self._task_type == "binary":
            return torch.tensor(label, dtype=torch.float).unsqueeze(0)
        else:  # multiclass, ordinal
            return torch.tensor(label, dtype=torch.long)

    def get_sample_weights(self) -> Optional[torch.Tensor]:
        if self._task_type in ("multilabel",):
            return None
        labels = [s[1] for s in self._samples]
        counts = Counter(labels)
        total = len(labels)
        n_cls = len(counts)
        return torch.tensor(
            [total / (n_cls * counts[l]) for l in labels], dtype=torch.float
        )

    def get_class_weights(self) -> Optional[torch.Tensor]:
        if self._task_type in ("multilabel",):
            return None
        labels = [s[1] for s in self._samples]
        counts = Counter(labels)
        total = len(labels)
        return torch.tensor(
            [total / (self._num_classes * counts.get(i, 1)) for i in range(self._num_classes)],
            dtype=torch.float,
        )
