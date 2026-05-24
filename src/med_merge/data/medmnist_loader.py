"""Generic loader for MedMNIST-v2 subsets.

Wraps the ``medmnist`` Python package
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

import torch

from med_merge.data.base import ImageListDataset, MedMergeDataset

logger = logging.getLogger(__name__)


MEDMNIST_SUBSETS: dict[str, dict] = {
    "pathmnist": {
        "py_class": "PathMNIST",
        "num_classes": 9,
        "task_type": "multiclass",
        # Standard MedMNIST class names for pathmnist.
        "class_names": [
            "ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM",
        ],
    },
    "retinamnist": {
        "py_class": "RetinaMNIST",
        "num_classes": 5,
        "task_type": "multiclass",  # ordinal-regression underlying; we use CE
        "class_names": ["0", "1", "2", "3", "4"],
    },
}


def _split_arg(split: str) -> str:
    """Map our split names to MedMNIST's."""
    if split == "validation":
        return "val"
    return split  # 'train' / 'val' / 'test' (medmnist accepts 'val')


def _import_subset(name: str):
    import medmnist
    cls_name = MEDMNIST_SUBSETS[name]["py_class"]
    return getattr(medmnist, cls_name)


class _MedMNISTLazyDataset(MedMergeDataset):
    """Lazy adapter over a medmnist dataset object.

    """

    def __init__(self, mm_ds, transform, meta: dict, name: str):
        self._mm = mm_ds
        self.transform = transform
        self._name = name
        self._task_type = meta["task_type"]
        self._num_classes = meta["num_classes"]
        self._class_names = meta["class_names"]

    def __len__(self) -> int:
        return len(self._mm)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, label = self._mm[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if hasattr(label, "shape") and label.shape == (1,):
            label_val = int(label[0])
        else:
            label_val = label

        if self._task_type == "multilabel":
            label_t = torch.tensor(label_val, dtype=torch.float)
        elif self._task_type == "binary":
            label_t = torch.tensor(label_val, dtype=torch.float).unsqueeze(0)
        else:
            label_t = torch.tensor(label_val, dtype=torch.long)
        return img, label_t

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    def get_class_weights(self):
        if self._task_type == "multilabel":
            return None
        from collections import Counter
        labels = [int(self._mm[i][1][0]) for i in range(len(self._mm))]
        counts = Counter(labels)
        total = len(labels)
        return torch.tensor(
            [total / (self._num_classes * counts.get(i, 1))
             for i in range(self._num_classes)],
            dtype=torch.float,
        )


def load_medmnist_subset(
    subset_name: str,
    data_dir: str,
    split: str = "train",
    transform=None,
    image_size: int = 224,
    **kwargs,
) -> MedMergeDataset:
    """Lazy MedMNIST loader. Returns the medmnist dataset wrapped to match
    our MedMergeDataset interface, without pre-materializing all PIL images.
    """
    if subset_name not in MEDMNIST_SUBSETS:
        raise ValueError(
            f"Unknown MedMNIST subset: {subset_name}. "
            f"Available: {list(MEDMNIST_SUBSETS)}"
        )

    meta = MEDMNIST_SUBSETS[subset_name]
    cls = _import_subset(subset_name)

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"MedMNIST {subset_name} split={split} size={image_size} root={root}"
    )
    ds = cls(
        split=_split_arg(split),
        download=True,
        size=image_size,
        root=str(root),
    )
    logger.info(f"MedMNIST {subset_name} {split}: {len(ds)} samples")
    return _MedMNISTLazyDataset(ds, transform, meta, name=subset_name)


# Thin per-subset wrappers so the registry can address them by name without
# having to pass subset_name through the kwargs chain.
def load_pathmnist(data_dir: str, split: str = "train", transform=None, **kwargs):
    return load_medmnist_subset("pathmnist", data_dir, split, transform, **kwargs)


def load_retinamnist(data_dir: str, split: str = "train", transform=None, **kwargs):
    return load_medmnist_subset("retinamnist", data_dir, split, transform, **kwargs)
