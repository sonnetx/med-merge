"""Iso-C: isotropic model merging (Marczak et al., "No Task Left Behind", ICML 2025).

Included as a *current-SOTA* baseline so the benchmark compares against 2025 methods,
not only the 2022-2024 ones.

Mechanism
---------
1. Sum the task vectors into a single task-arithmetic update  Delta = sum_i tau_i.
2. For each 2D weight matrix, take the SVD  Delta = U S V^T  and replace every
   singular value with the mean singular value ``s_bar``, reconstructing
   ``s_bar * U V^T``.  Flattening the (normally skewed) spectrum to an isotropic one
   keeps the subspace directions but removes the dominance of a handful of directions,
   which is what starves the weaker / underrepresented tasks in a naive sum.
3. Non-matrix parameters (1D biases, LayerNorm scale/shift) are passed through
   unchanged.
4. Apply to the pretrained weights with an overall scale ``alpha``.

This is the ``Iso-C`` variant (common isotropic subspace).  ``Iso-CTS`` (common +
task-specific orthogonal subspaces) is a natural follow-up baseline but is left out
here to keep the baseline dependency-free and simple.
"""

from __future__ import annotations

from typing import Optional

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector


class IsoCMerger(BaseMerger):
    """Isotropic (flattened-spectrum) merging of the summed task vector."""

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        alpha: Optional[float] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        alpha = alpha if alpha is not None else self.config.alpha
        if alpha is None:
            alpha = 1.0

        combined = None
        for tv in task_vectors.values():
            combined = tv if combined is None else combined + tv

        new_vector: dict[str, torch.Tensor] = {}
        for key, tensor in combined.vector.items():
            if tensor.ndim == 2 and min(tensor.shape) > 1:
                mat = tensor.double()
                u, s, vh = torch.linalg.svd(mat, full_matrices=False)
                s_bar = s.mean()
                iso = (u * s_bar) @ vh  # s_bar * U V^T
                new_vector[key] = iso.reshape(tensor.shape).to(tensor.dtype)
            else:
                new_vector[key] = tensor.clone()

        tv = TaskVector(vector=new_vector)
        return tv.apply_to(self.pretrained, scaling_coef=alpha)

    @property
    def name(self) -> str:
        return "iso_c"

    @property
    def hyperparameters(self) -> dict:
        return {"alpha": self.config.alpha}
