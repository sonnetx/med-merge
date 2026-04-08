"""SLERP: Spherical Linear Interpolation (two-model only)."""

from __future__ import annotations

from typing import Optional

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector


class SLERPMerger(BaseMerger):
    """Spherical linear interpolation between two task vectors.

    SLERP interpolates on the unit-sphere surface, avoiding the magnitude
    dip of linear interpolation.  Operates per-parameter on flattened
    task-vector tensors.

    Requires exactly 2 task vectors.
    """

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        t: Optional[float] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if len(task_vectors) != 2:
            raise ValueError(
                f"SLERP requires exactly 2 task vectors, got {len(task_vectors)}"
            )

        t = t if t is not None else self.config.slerp_t
        names = list(task_vectors.keys())
        tv_a = task_vectors[names[0]]
        tv_b = task_vectors[names[1]]

        merged_vector = {}
        for key in tv_a.vector:
            v1 = tv_a.vector[key].float()
            v2 = tv_b.vector[key].float()
            merged_vector[key] = self._slerp_tensor(v1, v2, t)

        result = TaskVector(vector=merged_vector)
        return result.apply_to(self.pretrained, scaling_coef=1.0)

    @staticmethod
    def _slerp_tensor(v1: torch.Tensor, v2: torch.Tensor, t: float) -> torch.Tensor:
        """SLERP between two tensors (flattened, then reshaped back)."""
        shape = v1.shape
        v1_flat = v1.reshape(-1)
        v2_flat = v2.reshape(-1)

        # Normalise
        n1 = v1_flat.norm()
        n2 = v2_flat.norm()

        if n1 < 1e-8 or n2 < 1e-8:
            # Degenerate — fall back to lerp
            return ((1 - t) * v1 + t * v2).to(v1.dtype)

        u1 = v1_flat / n1
        u2 = v2_flat / n2
        cos_omega = torch.clamp(torch.dot(u1, u2), -1.0, 1.0)

        if cos_omega.abs() > 0.9999:
            # Nearly parallel — lerp for numerical stability
            return ((1 - t) * v1 + t * v2).to(v1.dtype)

        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)
        coeff_a = torch.sin((1 - t) * omega) / sin_omega
        coeff_b = torch.sin(t * omega) / sin_omega

        # Interpolate on unit sphere, then scale magnitude
        interp_mag = (1 - t) * n1 + t * n2
        result_flat = (coeff_a * u1 + coeff_b * u2) * interp_mag
        return result_flat.reshape(shape).to(v1.dtype)

    @property
    def name(self) -> str:
        return "slerp"

    @property
    def hyperparameters(self) -> dict:
        return {"slerp_t": self.config.slerp_t}
