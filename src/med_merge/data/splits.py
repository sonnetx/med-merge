"""Persistent, reproducible train/val/test splits.

Saves split indices to disk so the same split is reused across experiments,
matching the pattern from compressed-perception's ``SplitManager``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SplitManager:
    """Create and cache stratified (optionally group-aware) splits."""

    def __init__(
        self,
        dataset_name: str,
        output_dir: str = "./outputs/splits",
    ):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)

    def get_or_create_split(
        self,
        n_samples: int,
        seed: int,
        ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
        group_keys: Optional[np.ndarray] = None,
        stratify_labels: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Return ``{"train": indices, "validation": indices, "test": indices}``.

        If a cached split exists for this dataset/seed, load it.
        Otherwise create one and save to disk.

        Args:
            n_samples: Total number of samples.
            seed: Random seed.
            ratios: (train, val, test) fractions summing to 1.
            group_keys: If provided, split by unique groups (e.g. lesion_id)
                so no group straddles splits.
            stratify_labels: If provided (and ``group_keys`` is None),
                use stratified splitting.
        """
        split_dir = self.output_dir / self.dataset_name / f"seed_{seed}"
        meta_path = split_dir / "metadata.json"

        if meta_path.exists():
            # Validate cached n_samples matches current. A `max_idx < n_samples`
            # check is not enough: if the dataset grew, old indices stay in-range
            # but no longer correspond to the same rows. TCGA hit exactly this:
            # 797 cached indices ended up entirely in the LUAD region of a
            # later 3220-row table, collapsing val to one class.
            try:
                cached_meta = json.loads(meta_path.read_text())
                cached_n = int(cached_meta.get("n_samples", -1))
            except Exception:
                cached_n = -1
            if cached_n == n_samples:
                return self._load(split_dir)
            logger.warning(
                f"[{self.dataset_name}] Cached split n_samples={cached_n} "
                f"!= current {n_samples}. Regenerating split."
            )
            import shutil
            shutil.rmtree(split_dir)

        if group_keys is not None:
            splits = self._split_by_groups(n_samples, group_keys, seed, ratios)
        elif stratify_labels is not None:
            splits = self._stratified_split(n_samples, stratify_labels, seed, ratios)
        else:
            splits = self._simple_split(n_samples, seed, ratios)

        self._save(split_dir, splits, n_samples, seed, ratios)
        return splits

    # ------------------------------------------------------------------

    def _simple_split(
        self, n: int, seed: int, ratios: tuple[float, ...]
    ) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])
        return {
            "train": indices[:train_end],
            "validation": indices[train_end:val_end],
            "test": indices[val_end:],
        }

    def _stratified_split(
        self,
        n: int,
        labels: np.ndarray,
        seed: int,
        ratios: tuple[float, ...],
    ) -> dict[str, np.ndarray]:
        """Per-class shuffle + ratio cut. Each class is split independently
        into train/val/test slices, then concatenated. Guarantees both val
        and test contain every class with at least floor(class_count * ratio)
        samples.
        """
        rng = np.random.RandomState(seed)
        labels = np.asarray(labels)
        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] = []
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            nc = len(cls_idx)
            tr_end = int(nc * ratios[0])
            va_end = tr_end + int(nc * ratios[1])
            train_idx.extend(cls_idx[:tr_end].tolist())
            val_idx.extend(cls_idx[tr_end:va_end].tolist())
            test_idx.extend(cls_idx[va_end:].tolist())
        return {
            "train": np.array(train_idx),
            "validation": np.array(val_idx),
            "test": np.array(test_idx),
        }

    def _split_by_groups(
        self,
        n_samples: int,
        group_keys: np.ndarray,
        seed: int,
        ratios: tuple[float, ...],
    ) -> dict[str, np.ndarray]:
        """Split by unique groups so no group spans splits."""
        rng = np.random.RandomState(seed)
        unique_groups = list(set(group_keys.tolist()))
        rng.shuffle(unique_groups)

        ng = len(unique_groups)
        train_end = int(ng * ratios[0])
        val_end = train_end + int(ng * ratios[1])

        group_to_split = {}
        for g in unique_groups[:train_end]:
            group_to_split[g] = "train"
        for g in unique_groups[train_end:val_end]:
            group_to_split[g] = "validation"
        for g in unique_groups[val_end:]:
            group_to_split[g] = "test"

        splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
        for i, g in enumerate(group_keys):
            splits[group_to_split[g]].append(i)

        return {k: np.array(v) for k, v in splits.items()}

    # ------------------------------------------------------------------

    def _save(
        self,
        split_dir: Path,
        splits: dict[str, np.ndarray],
        n_samples: int,
        seed: int,
        ratios: tuple[float, ...],
    ) -> None:
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, indices in splits.items():
            np.save(split_dir / f"{name}_indices.npy", indices)
        meta = {
            "dataset": self.dataset_name,
            "seed": seed,
            "n_samples": n_samples,
            "ratios": list(ratios),
            "split_sizes": {k: len(v) for k, v in splits.items()},
        }
        (split_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Saved split to {split_dir}")

    def _load(self, split_dir: Path) -> dict[str, np.ndarray]:
        result = {}
        for name in ("train", "validation", "test"):
            path = split_dir / f"{name}_indices.npy"
            if path.exists():
                result[name] = np.load(path)
        logger.info(
            f"Loaded cached split from {split_dir}: "
            + ", ".join(f"{k}={len(v)}" for k, v in result.items())
        )
        return result
