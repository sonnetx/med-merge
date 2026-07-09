"""Iso-CTS: isotropic merging with Common and Task-Specific subspaces.

The stronger variant of Iso-C from Marczak et al. (ICML 2025), and the one they
report as state of the art. Unlike Iso-C, which isotropizes the *full* singular
spectrum of the summed task matrix (and therefore crushes the few dominant task
directions when the spectrum is skewed / task count is low), Iso-CTS builds a
*bounded, curated* basis and isotropizes only over it:

Per 2D weight matrix, with T tasks and retained rank r:
1. Common subspace: top-k singular directions of the summed update Delta = sum_t tau_t.
2. Task-specific subspaces: for each task t, project its own update onto the
   orthogonal complement of the common subspace, Delta_t^perp = Delta_t - U_c U_c^T Delta_t,
   and keep its top-s directions (s = (r-k)/T). This guarantees every task
   dedicated directions ("no task left behind").
3. Orthonormalize the concatenated common+task-specific bases (U*, V*).
4. Isotropic scale by the mean singular value over the *retained* (top-r)
   directions of Delta -- not the full spectrum -- so dominant signal is preserved.
5. Apply with an overall scale alpha.

Non-matrix params are merged by a plain sum. Faithful re-implementation for use as
a fair subspace baseline; verify against the authors' repo before quoting head-to-head.
"""

from __future__ import annotations

from typing import Optional

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector


def _robust_svd(mat: torch.Tensor):
    """SVD with jitter-retry for ill-conditioned / repeated-singular-value matrices
    (device-agnostic: the gesvd driver is CUDA-only, so we perturb and retry instead)."""
    try:
        return torch.linalg.svd(mat, full_matrices=False)
    except torch._C._LinAlgError:
        scale = mat.abs().max().clamp(min=1e-12)
        for mult in (1e-6, 1e-4, 1e-2):
            try:
                return torch.linalg.svd(mat + mult * scale * torch.randn_like(mat),
                                        full_matrices=False)
            except torch._C._LinAlgError:
                continue
        raise


def _orthonormal(mat: torch.Tensor) -> torch.Tensor:
    """Nearest orthonormal basis for the columns of mat (polar factor U V^T)."""
    u, _, vh = _robust_svd(mat)
    return u @ vh


class IsoCTSMerger(BaseMerger):
    """Isotropic merge over common + task-specific subspaces."""

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        alpha: Optional[float] = None,
        rank_fraction: Optional[float] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        alpha = alpha if alpha is not None else self.config.alpha
        if alpha is None:
            alpha = 1.0
        # fraction of min(m,n) to retain as the total curated rank r
        frac = rank_fraction if rank_fraction is not None else getattr(self.config, "isocts_rank_fraction", 0.5)

        names = list(task_vectors.keys())
        keys = list(task_vectors[names[0]].vector.keys())
        n_tasks = len(names)

        combined = None
        for tv in task_vectors.values():
            combined = tv if combined is None else combined + tv

        merged_vector: dict[str, torch.Tensor] = {}
        for key in keys:
            ref = combined.vector[key]
            if ref.ndim != 2 or min(ref.shape) <= 1 or n_tasks == 1:
                merged_vector[key] = combined.vector[key].clone()
                continue

            delta = combined.vector[key].double()
            full = min(ref.shape)
            r = max(n_tasks, int(round(frac * full)))
            r = min(r, full)
            k = max(1, r // 2)                       # common budget
            s = max(1, (r - k) // n_tasks)           # per-task budget

            u, sd, vh = _robust_svd(delta)
            u_c = u[:, :k]
            v_c = vh[:k, :].t()

            u_blocks = [u_c]
            v_blocks = [v_c]
            for name in names:
                d_t = task_vectors[name].vector[key].double()
                d_perp = d_t - u_c @ (u_c.t() @ d_t)     # remove common column space
                ut, _, vht = _robust_svd(d_perp)
                u_blocks.append(ut[:, :s])
                v_blocks.append(vht[:s, :].t())

            u_star = _orthonormal(torch.cat(u_blocks, dim=1))
            v_star = _orthonormal(torch.cat(v_blocks, dim=1))
            R = u_star.shape[1]
            sigma_bar = sd[:R].mean()                 # mean over RETAINED (top) directions
            merged = (u_star * sigma_bar) @ v_star.t()
            merged_vector[key] = merged.reshape(ref.shape).to(ref.dtype)

        tv = TaskVector(vector=merged_vector)
        return tv.apply_to(self.pretrained, scaling_coef=alpha)

    @property
    def name(self) -> str:
        return "iso_cts"

    @property
    def hyperparameters(self) -> dict:
        return {"alpha": self.config.alpha}
