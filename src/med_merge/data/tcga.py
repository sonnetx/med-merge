"""TCGA lung histology subtype dataset (LUAD vs LUSC binary classification)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from med_merge.data.base import ImageListDataset
from med_merge.data.splits import SplitManager

logger = logging.getLogger(__name__)


def load_tcga(
    data_dir: str,
    split: str = "train",
    transform=None,
    csv_path: Optional[str] = None,
    split_seed: int = 42,
    split_ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
    splits_dir: str = "./outputs/splits",
    **kwargs,
) -> ImageListDataset:
    """Load TCGA LUAD-vs-LUSC with persistent splits."""
    data_dir_p = Path(data_dir)
    if csv_path is None:
        csv_path = str(data_dir_p / "tables" / "dataset.csv")

    df = pd.read_csv(csv_path)
    df = df[df["project_id"].isin(["TCGA-LUAD", "TCGA-LUSC"])].copy()
    df["label"] = df["project_id"].map({"TCGA-LUAD": 0, "TCGA-LUSC": 1})

    # Filter to slides with thumbnails
    thumb_dir = data_dir_p / "thumbnails"
    valid_mask = [
        (thumb_dir / f"{row['slide_id']}.jpg").exists()
        for _, row in df.iterrows()
    ]
    df = df[valid_mask].reset_index(drop=True)

    mgr = SplitManager("tcga", output_dir=splits_dir)
    splits = mgr.get_or_create_split(
        n_samples=len(df), seed=split_seed, ratios=split_ratios,
    )

    indices = splits.get(split, splits["train"])
    samples = []
    for idx in indices:
        idx = int(idx)
        row = df.iloc[idx]
        samples.append((thumb_dir / f"{row['slide_id']}.jpg", int(row["label"])))

    logger.info(
        f"TCGA {split}: {len(samples)} images, "
        f"LUAD={sum(1 for _, l in samples if l == 0)}, "
        f"LUSC={sum(1 for _, l in samples if l == 1)}"
    )

    return ImageListDataset(
        samples,
        transform,
        dataset_name="tcga",
        dataset_task_type="binary",
        dataset_num_classes=1,
        dataset_class_names=["LUAD", "LUSC"],
    )


# Backward-compatible alias
class TCGADataset:
    """Deprecated: use ``load_tcga()`` instead."""

    def __new__(cls, data_dir="./data/tcga", split="train", transform=None, **kwargs):
        return load_tcga(data_dir, split, transform, **kwargs)
