"""Recast a multiclass dataset as one-vs-rest binary detection.

The controlled benchmark fixes task type, so extending it to more tasks means expressing
each new domain as a binary detection problem on the same images rather than swapping in a
natively binary dataset. ``BinaryView`` wraps any dataset returning integer class labels and
maps a chosen positive class to 1 and everything else to 0, matching the label shape and
dtype the binary training path expects.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from med_merge.data.base import MedMergeDataset

logger = logging.getLogger(__name__)


class BinaryView(MedMergeDataset):
    """One-vs-rest view of a multiclass dataset. Images are untouched."""

    def __init__(self, base, positive_index: int, positive_name: Optional[str] = None):
        self._base = base
        self._positive_index = int(positive_index)
        names = getattr(base, "class_names", None)
        if positive_name is None and names:
            positive_name = names[self._positive_index]
        self._positive_name = positive_name or f"class{self._positive_index}"

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._base[idx]
        if hasattr(label, "shape") and getattr(label, "numel", lambda: 1)() == 1:
            label = int(label.reshape(-1)[0])
        else:
            label = int(label)
        y = float(label == self._positive_index)
        return image, torch.tensor(y, dtype=torch.float).unsqueeze(0)

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def task_type(self) -> str:
        return "binary"

    @property
    def class_names(self) -> list[str]:
        return [f"not_{self._positive_name}", self._positive_name]

    def get_sample_weights(self) -> Optional[torch.Tensor]:
        # Binary heads use pos_weight rather than per-sample weights.
        return None

    def get_class_weights(self) -> Optional[torch.Tensor]:
        return None


def binary_view(base, positive_name: str, split: str = "") -> BinaryView:
    """Wrap ``base``, taking the positive class by name from its ``class_names``."""
    names = list(base.class_names)
    if positive_name not in names:
        raise ValueError(f"positive class {positive_name!r} not in {names}")
    view = BinaryView(base, names.index(positive_name), positive_name)
    pos = sum(1 for i in range(len(view)) if view[i][1].item() > 0.5) if len(view) < 5000 else None
    if pos is not None:
        logger.info(f"{type(base).__name__} {split} (binary {positive_name}): "
                    f"{len(view)} images, pos={pos}")
    else:
        logger.info(f"{type(base).__name__} {split} (binary {positive_name}): {len(view)} images")
    return view


def load_ham10000_mel(data_dir: str, split: str = "train", transform=None, **kwargs):
    """HAM10000 as binary melanoma detection, a second dermoscopy task."""
    from med_merge.data.ham10000 import HAM10000Dataset
    return binary_view(HAM10000Dataset(data_dir, split, transform, **kwargs), "mel", split)


def load_nct_crc_tum(data_dir: str, split: str = "train", transform=None, **kwargs):
    """NCT-CRC as binary tumor detection, a second histopathology task."""
    from med_merge.data.nct_crc import NCTCRCDataset
    return binary_view(NCTCRCDataset(data_dir, split, transform, **kwargs), "TUM", split)
