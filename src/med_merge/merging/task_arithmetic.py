"""Task Arithmetic merging method."""

from __future__ import annotations

from typing import Optional

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector


class TaskArithmeticMerger(BaseMerger):
    """Task Arithmetic: pretrained + alpha * sum(task_vectors).

    alpha is tuned via validation performance.
    """

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        alpha: Optional[float] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        alpha = alpha if alpha is not None else self.config.alpha
        if alpha is None:
            alpha = 0.3  # sensible default

        combined = None
        for tv in task_vectors.values():
            if combined is None:
                combined = tv
            else:
                combined = combined + tv

        return combined.apply_to(self.pretrained, scaling_coef=alpha)

    @property
    def name(self) -> str:
        return "task_arithmetic"

    @property
    def hyperparameters(self) -> dict:
        return {"alpha": self.config.alpha}
