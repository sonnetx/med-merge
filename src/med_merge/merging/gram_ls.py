"""Gram-LS Merge: task-vector Gram-matrix least-squares (projection-preserving) merging.

Novel method (this project's contribution). For each parameter tensor, it solves for
merge coefficients ``c`` such that the merged task vector ``m = sum_j c_j * tau_j``
reproduces each task's own signal under projection:

    <m, tau_i> = <tau_i, tau_i>   for every task i.

In matrix form this is the K x K linear system

    G c = diag(G),      where   G_ij = <tau_i, tau_j>

is the Gram matrix of the task vectors themselves, and ``diag(G)`` is the vector of
squared task-vector norms.  The solution is ``c = G^{-1} diag(G)``.

Why this fixes cross-domain merge failure
------------------------------------------
Simple averaging uses ``c_j = 1/K`` uniformly, which shrinks each task's own
self-projection to roughly ``1/K`` plus cross-task contamination.  When one task
vector is small in norm and near-orthogonal to the others (the common regime in
cross-domain medical merging), averaging dilutes it away.  The projection-preserving
solution instead recovers:

* orthogonal tasks  ->  G diagonal  ->  c -> 1  (a *sum*: each task preserved, zero
  interference), and
* correlated tasks  ->  G^{-1} down-weights the shared directions to avoid
  double-counting.

Coefficients are solved per parameter tensor, so the merge adapts to the fact that
the cross-task orthogonality structure varies by layer.

Relation to prior work
-----------------------
Distinct from RegMean and MaTS, which solve least-squares systems over *activation /
gradient* Gram matrices (X^T X); and from Superpose, which uses the Gram of *singular
vectors*.  Here the Gram is over the raw task vectors, and the objective is explicit
projection preservation (undoing the 1/K averaging shrinkage).
"""

from __future__ import annotations

from typing import Optional

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector


class GramLSMerger(BaseMerger):
    """Projection-preserving merge via the task-vector Gram matrix."""

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        alpha: Optional[float] = None,
        gram_lambda: Optional[float] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        alpha = alpha if alpha is not None else self.config.alpha
        if alpha is None:
            alpha = 1.0  # coefficients already set per-task scale; alpha is a global knob
        lam = gram_lambda if gram_lambda is not None else self.config.gram_lambda

        names = list(task_vectors.keys())
        keys = list(task_vectors[names[0]].vector.keys())
        merged_vector: dict[str, torch.Tensor] = {}

        for key in keys:
            tensors = []
            for name in names:
                if key in task_vectors[name].vector:
                    tensors.append(task_vectors[name].vector[key])
            if not tensors:
                continue

            ref = tensors[0]
            if len(tensors) == 1:
                merged_vector[key] = tensors[0].clone()
                continue

            # Solve the K x K Gram system in float64 for numerical stability.
            flat = torch.stack([t.reshape(-1).double() for t in tensors])  # (K, D)
            gram = flat @ flat.t()  # (K, K)
            diag = torch.diagonal(gram)  # (K,) = squared norms
            k = flat.shape[0]

            # Relative Tikhonov regularization: scales with the Gram magnitude so it
            # is norm-invariant, and keeps the system well-posed when two tasks are
            # near-collinear (e.g. a same-task/different-site pair).
            # Norm-aware mode scales the ridge by the *largest* task norm rather than
            # the mean, so a weak/noisy task vector (small diagonal) is damped much
            # harder relative to its own magnitude, pulling its coefficient toward the
            # stable sum solution instead of amplifying an ill-conditioned inverse.
            scale = (diag.max() if self.config.gram_norm_aware else diag.mean()).clamp(min=1e-12)
            gram_reg = gram + lam * scale * torch.eye(k, dtype=gram.dtype)

            try:
                coeffs = torch.linalg.solve(gram_reg, diag)
            except RuntimeError:
                coeffs = torch.linalg.lstsq(gram_reg, diag.unsqueeze(1)).solution.squeeze(1)

            merged_flat = (coeffs.unsqueeze(1) * flat).sum(dim=0)  # (D,)
            merged_vector[key] = merged_flat.reshape(ref.shape).to(ref.dtype)

        tv = TaskVector(vector=merged_vector)
        return tv.apply_to(self.pretrained, scaling_coef=alpha)

    @property
    def name(self) -> str:
        return "gram_ls"

    @property
    def hyperparameters(self) -> dict:
        return {"alpha": self.config.alpha, "gram_lambda": self.config.gram_lambda}
