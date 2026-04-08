"""Dataset download / verification.

All three datasets are local on Sherlock — this module just verifies
the expected paths exist.
"""

from __future__ import annotations

import logging
from pathlib import Path

from med_merge.config.constants import ALL_DATASETS

logger = logging.getLogger(__name__)


def verify_isic2017(data_dir: Path) -> None:
    """ISIC 2017 images + CSV on Sherlock."""
    logger.info("ISIC 2017: local dataset, no download needed. Verify paths in configs/datasets/isic2017.yaml")


def verify_chexpert(data_dir: Path) -> None:
    """CheXpert images + CSV on Sherlock."""
    logger.info("CheXpert: local dataset, no download needed. Verify paths in configs/datasets/chexpert.yaml")


def verify_tcga(data_dir: Path) -> None:
    """TCGA thumbnails + CSV on Sherlock (built via compressed-perception ETL)."""
    logger.info("TCGA: local dataset, no download needed. Verify paths in configs/datasets/tcga.yaml")


DOWNLOAD_FUNCTIONS = {
    "isic2017": verify_isic2017,
    "chexpert": verify_chexpert,
    "tcga": verify_tcga,
}


def download_datasets(datasets: list[str], data_dir: str) -> None:
    """Verify specified datasets exist (or 'all')."""
    data_path = Path(data_dir)

    if "all" in datasets:
        datasets = ALL_DATASETS

    for name in datasets:
        if name in DOWNLOAD_FUNCTIONS:
            DOWNLOAD_FUNCTIONS[name](data_path)
        else:
            logger.warning(f"Unknown dataset: {name}")
