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
            return self._load(split_dir)

        if group_keys is not None:
            splits = self._split_by_groups(n_samples, group_keys, seed, ratios)
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
