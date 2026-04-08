"""ISIC 2017 dermoscopy dataset (3-class skin lesion classification).

Matches compressed-perception's dermatology pipeline:
- 3 classes: nevus (0), melanoma (1), seborrheic keratosis (2)
- Labels derived from multi-column CSV (melanoma, seborrheic_keratosis)
- Images loaded from local directory as .jpg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from med_merge.data.base import ImageListDataset
from med_merge.data.splits import SplitManager

logger = logging.getLogger(__name__)

CLASS_NAMES = ["nevus", "melanoma", "seborrheic_keratosis"]
LABEL_COLUMNS = ["melanoma", "seborrheic_keratosis"]


def _encode_label(row: dict) -> int:
    """Convert multi-column binary labels to single class index.

    If both melanoma and seborrheic_keratosis are 0 → nevus (0).
    Otherwise argmax + 1.
    """
    mel = float(row.get("melanoma", 0))
    seb = float(row.get("seborrheic_keratosis", 0))
    if max(mel, seb) == 0:
        return 0  # nevus
    return int(np.argmax([mel, seb])) + 1


def load_isic2017(
    data_dir: str,
    split: str = "train",
    transform=None,
    csv_path: Optional[str] = None,
    image_id_column: str = "image_id",
    image_extension: str = ".jpg",
    split_seed: int = 42,
    split_ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
    splits_dir: str = "./outputs/splits",
    **kwargs,
) -> ImageListDataset:
    """Load ISIC 2017 with persistent splits via SplitManager.

    Args:
        data_dir: Path to image directory containing .jpg files.
        csv_path: Path to labels CSV. Defaults to ``data_dir/../merged_ground_truth_part3.csv``
            or ``data_dir/labels.csv``.
        image_id_column: Column name for image identifiers.
        image_extension: File extension for images.
    """
    data_dir_p = Path(data_dir)

    # Resolve CSV path
    if csv_path is None:
        candidates = [
            data_dir_p.parent / "merged_ground_truth_part3.csv",
            data_dir_p / "labels.csv",
            data_dir_p / "merged_ground_truth_part3.csv",
        ]
        for c in candidates:
            if c.exists():
                csv_path = str(c)
                break
        if csv_path is None:
            raise FileNotFoundError(
                f"No labels CSV found. Tried: {[str(c) for c in candidates]}. "
                "Pass csv_path explicitly."
            )

    df = pd.read_csv(csv_path)

    # Encode labels
    labels = []
    for _, row in df.iterrows():
        labels.append(_encode_label(row))
    labels_arr = np.array(labels)

    mgr = SplitManager("isic2017", output_dir=splits_dir)
    splits = mgr.get_or_create_split(
        n_samples=len(df),
        seed=split_seed,
        ratios=split_ratios,
        stratify_labels=labels_arr,
    )

    indices = splits.get(split, splits["train"])

    # Build samples: (image_path, label)
    img_dir = data_dir_p
    samples = []
    for idx in indices:
        idx = int(idx)
        image_id = df.iloc[idx][image_id_column]
        img_path = img_dir / f"{image_id}{image_extension}"
        samples.append((img_path, labels[idx]))

    logger.info(
        f"ISIC 2017 {split}: {len(samples)} images, "
        f"nevus={sum(1 for _, l in samples if l == 0)}, "
        f"melanoma={sum(1 for _, l in samples if l == 1)}, "
        f"seb_k={sum(1 for _, l in samples if l == 2)}"
    )

    return ImageListDataset(
        samples,
        transform,
        dataset_name="isic2017",
        dataset_task_type="multiclass",
        dataset_num_classes=3,
        dataset_class_names=CLASS_NAMES,
    )
