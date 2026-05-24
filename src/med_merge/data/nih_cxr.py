"""NIH ChestX-ray14 dataset (5-label multi-label, near-domain pair to CheXpert).
"""

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

# CheXpert label set (target). Map NIH names -> CheXpert names.
TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
NIH_TO_TARGET = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Pleural Effusion",
}


def _resolve_image_path(image_id: str, img_dir: Path) -> Path:
    """Find an NIH image. Tries flat layout first, then images_XXX/images/."""
    flat = img_dir / image_id
    if flat.exists():
        return flat
    nested = img_dir / "images" / image_id
    if nested.exists():
        return nested
    # Fall back to flat path; ImageListDataset will surface the actual error.
    return flat


def load_nih_cxr(
    data_dir: str,
    split: str = "train",
    transform=None,
    csv_path: str | None = None,
    split_seed: int = 42,
    split_ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
    splits_dir: str = "./outputs/splits",
    subsample: int | None = None,
    **kwargs,
) -> ImageListDataset:
    """Load NIH ChestX-ray14 with persistent splits.

    ``data_dir`` should contain the images (flat ``{data_dir}/{ImageIndex}.png``
    or nested ``{data_dir}/images/{ImageIndex}.png``).
    ``csv_path`` defaults to ``{data_dir}/Data_Entry_2017.csv``.
    ``subsample`` optionally caps the dataset size for fast smoke tests.
    """
    if csv_path is None:
        csv_path = str(Path(data_dir) / "Data_Entry_2017.csv")

    df = pd.read_csv(csv_path).reset_index(drop=True)
    img_dir = Path(data_dir)
    image_col = "Image Index" if "Image Index" in df.columns else "ImageIndex"

    # Optional subsampling for smoke tests — done before splitting so train/val/test
    # all come from the same subsample (and the split is deterministic in size).
    if subsample is not None and subsample < len(df):
        df = df.sample(n=subsample, random_state=split_seed).reset_index(drop=True)

    # Filter to rows whose image file actually exists on disk.
    image_ids = [str(x) for x in df[image_col].tolist()]
    valid_mask = [_resolve_image_path(iid, img_dir).exists() for iid in image_ids]
    n_missing = len(valid_mask) - sum(valid_mask)
    if n_missing > 0:
        logger.warning(
            f"NIH-CXR: {n_missing}/{len(df)} images missing on disk; filtering them out."
        )
    df = df[valid_mask].reset_index(drop=True)

    # Build the multi-hot label matrix from the pipe-separated "Finding Labels" column.
    n = len(df)
    labels = np.zeros((n, len(TARGET_LABELS)), dtype=np.float32)
    target_idx = {name: i for i, name in enumerate(TARGET_LABELS)}

    for row_i, finding_str in enumerate(df["Finding Labels"].fillna("").astype(str)):
        if not finding_str or finding_str == "No Finding":
            continue
        for raw in finding_str.split("|"):
            raw = raw.strip()
            target = NIH_TO_TARGET.get(raw)
            if target is not None:
                labels[row_i, target_idx[target]] = 1.0

    mgr = SplitManager("nih_cxr", output_dir=splits_dir)
    splits = mgr.get_or_create_split(
        n_samples=n, seed=split_seed, ratios=split_ratios,
    )

    indices = splits.get(split, splits["train"])

    samples = []
    for idx in indices:
        idx = int(idx)
        image_id = str(df.iloc[idx][image_col])
        samples.append((_resolve_image_path(image_id, img_dir), labels[idx].tolist()))

    logger.info(f"NIH-CXR {split}: {len(samples)} images")

    return ImageListDataset(
        samples,
        transform,
        dataset_name="nih_cxr",
        dataset_task_type="multilabel",
        dataset_num_classes=5,
        dataset_class_names=TARGET_LABELS,
    )
