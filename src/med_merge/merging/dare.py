"""DARE: Drop And Rescale merging method."""

from __future__ import annotations

from typing import Optional

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_arithmetic import TaskArithmeticMerger
from med_merge.merging.task_vector import TaskVector


class DAREMerger(BaseMerger):
    """Drop And Rescale delta parameters before merging.

    For each task vector:
    1. DROP: Randomly set proportion p of parameters to zero (Bernoulli mask).
    2. RESCALE: Multiply remaining by 1/(1-p) to preserve expected magnitude.
    3. Merge sparsified task vectors using inner method (Task Arithmetic).
    """

    def __init__(
        self,
        pretrained_state_dict: dict[str, torch.Tensor],
        config: MergingConfig,
    ):
        super().__init__(pretrained_state_dict, config)
        inner = config.inner_method or "task_arithmetic"
        if inner == "ties":
            from med_merge.merging.ties import TIESMerger
            self.inner_merger = TIESMerger(pretrained_state_dict, config)
        else:
            self.inner_merger = TaskArithmeticMerger(pretrained_state_dict, config)

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        drop_rate: Optional[float] = None,
        alpha: Optional[float] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        drop_rate = drop_rate if drop_rate is not None else self.config.drop_rate
        seed = self.config.dare_seed

        # Apply drop-and-rescale to each task vector
        sparsified = {}
        for i, (name, tv) in enumerate(task_vectors.items()):
            sparsified[name] = self._drop_and_rescale(tv, drop_rate, seed + i)

        # Merge using inner method
        return self.inner_merger.merge(sparsified, alpha=alpha)

    def _drop_and_rescale(
        self, task_vector: TaskVector, drop_rate: float, seed: int
    ) -> TaskVector:
        """Apply DARE to a single task vector."""
        rng = torch.Generator().manual_seed(seed)
        new_vector = {}

        for key, tensor in task_vector.vector.items():
            # Bernoulli mask: keep with probability (1 - drop_rate)
            mask = torch.bernoulli(
                torch.full_like(tensor.float(), 1 - drop_rate),
                generator=rng,
            )
            # Rescale kept values
            rescaled = tensor * mask / max(1 - drop_rate, 1e-8)
            new_vector[key] = rescaled

        return TaskVector(vector=new_vector)

    @property
    def name(self) -> str:
        return "dare"

    @property
    def hyperparameters(self) -> dict:
        return {
            "drop_rate": self.config.drop_rate,
            "alpha": self.config.alpha,
            "dare_seed": self.config.dare_seed,
        }
