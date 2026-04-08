"""CheXpert chest X-ray dataset (5-label multi-label classification)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageFile

from med_merge.data.base import ImageListDataset
from med_merge.data.splits import SplitManager

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

LABEL_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def _construct_filename(csv_path_cell: str) -> str:
    """Convert CSV path column to flat-file name used on disk."""
    parts = csv_path_cell.replace("\\", "/").split("/")
    filtered = [p for p in parts[1:] if p not in ("train", "valid")]
    return "_".join(filtered)


def load_chexpert(
    data_dir: str,
    split: str = "train",
    transform=None,
    csv_path: str | None = None,
    split_seed: int = 42,
    split_ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
    splits_dir: str = "./outputs/splits",
    **kwargs,
) -> ImageListDataset:
    """Load CheXpert with persistent splits via SplitManager.

    ``data_dir`` must point to the image directory.
    ``csv_path`` must point to the labels CSV.  Both are required — there
    are no hardcoded Sherlock paths.
    """
    if csv_path is None:
        csv_path = str(Path(data_dir) / "train_valid_combined.csv")

    df = pd.read_csv(csv_path)
    labels = df[LABEL_COLUMNS].fillna(0.0).values.astype(np.float32)

    mgr = SplitManager("chexpert", output_dir=splits_dir)
    splits = mgr.get_or_create_split(
        n_samples=len(df), seed=split_seed, ratios=split_ratios,
    )

    indices = splits.get(split, splits["train"])
    img_dir = Path(data_dir)

    samples = []
    for idx in indices:
        idx = int(idx)
        filename = _construct_filename(df.iloc[idx]["Path"])
        samples.append((img_dir / filename, labels[idx].tolist()))

    logger.info(f"CheXpert {split}: {len(samples)} images")

    return ImageListDataset(
        samples,
        transform,
        dataset_name="chexpert",
        dataset_task_type="multilabel",
        dataset_num_classes=5,
        dataset_class_names=LABEL_COLUMNS,
    )


# Backward-compatible alias
class CheXpertDataset:
    """Deprecated: use ``load_chexpert()`` instead."""

    def __new__(cls, data_dir="./data/chexpert", split="train", transform=None, **kwargs):
        return load_chexpert(data_dir, split, transform, **kwargs)
