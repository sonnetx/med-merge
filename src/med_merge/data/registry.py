"""Dataset registry — maps dataset names to loader functions."""

from __future__ import annotations

from typing import Any, Optional

from med_merge.data.base import ImageListDataset


# Lazy registry: maps name -> (module_path, function_name)
_LOADER_REGISTRY: dict[str, tuple[str, str]] = {
    "isic2017": ("med_merge.data.isic2017", "load_isic2017"),
    "chexpert": ("med_merge.data.chexpert", "load_chexpert"),
    "tcga": ("med_merge.data.tcga", "load_tcga"),
    "nih_cxr": ("med_merge.data.nih_cxr", "load_nih_cxr"),
    "pathmnist": ("med_merge.data.medmnist_loader", "load_pathmnist"),
    "pathmnist_bin": ("med_merge.data.medmnist_loader", "load_pathmnist_bin"),
    "retinamnist": ("med_merge.data.medmnist_loader", "load_retinamnist"),
    # Real full-resolution loaders pulled from the Sherlock working tree. The
    # values are Dataset classes whose __init__(data_dir, split, transform, **kwargs)
    # matches the loader-callable convention used by build_dataset().
    "nct_crc": ("med_merge.data.nct_crc", "NCTCRCDataset"),
    "patchcamelyon": ("med_merge.data.patchcamelyon", "PatchCamelyonDataset"),
    # Binary controlled trio (one task type across three disjoint domains).
    "isic_mel": ("med_merge.data.isic2017", "load_isic_mel"),
    # Second task per domain, for the task-count extension (all binary, AUROC).
    "ham10000_mel": ("med_merge.data.binarize", "load_ham10000_mel"),
    "nct_crc_tum": ("med_merge.data.binarize", "load_nct_crc_tum"),
    "chexpert_cm": ("med_merge.data.chexpert", "load_chexpert_cm"),
    "chexpert_pe": ("med_merge.data.chexpert", "load_chexpert_pe"),
    "aptos2019": ("med_merge.data.aptos2019", "APTOS2019Dataset"),
    "ham10000": ("med_merge.data.ham10000", "HAM10000Dataset"),
    "fitzpatrick17k": ("med_merge.data.fitzpatrick", "Fitzpatrick17kDataset"),
}


def build_dataset(
    name: str,
    data_dir: str,
    split: str = "train",
    transform: Optional[Any] = None,
    **kwargs,
) -> ImageListDataset:
    """Build a dataset by name.

    Args:
        name: Dataset name (e.g., 'isic2017', 'chexpert', 'tcga').
        data_dir: Path to data directory.
        split: One of 'train', 'validation', 'test'.
        transform: Image transforms to apply.
        **kwargs: Additional dataset-specific arguments.

    Returns:
        ImageListDataset instance.
    """
    if name not in _LOADER_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(_LOADER_REGISTRY.keys())}"
        )

    module_path, func_name = _LOADER_REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    loader_fn = getattr(mod, func_name)
    return loader_fn(data_dir=data_dir, split=split, transform=transform, **kwargs)
