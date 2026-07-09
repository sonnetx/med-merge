"""TSV-Merge: Task Singular Vectors (Gargiulo et al., CVPR 2025).

Second current-SOTA baseline (alongside iso_c) so the benchmark reflects the 2025
SVD/subspace line, which is where the field's state of the art sits.

Mechanism (per 2D weight matrix)
--------------------------------
1. Reshape each task vector into its 2D weight matrix ``Delta_t``.
2. SVD each and truncate to a low rank ``k = floor(r / T)`` (r = min dimension,
   T = number of tasks), so the concatenated rank across tasks stays <= r.  Task
   matrices are empirically low-rank, so this keeps ~all of each task's signal.
3. Concatenate the truncated left/right singular vectors across tasks and
   *whiten* them: replace each concatenated basis with its nearest orthonormal
   matrix (the polar factor U V^T of its own SVD).  This decorrelates the
   cross-task singular directions, which is where destructive interference lives.
4. Reconstruct the merged update from the whitened bases and the concatenated
   singular values, and apply with an overall scale ``alpha``.

Non-matrix parameters (1D biases / LayerNorm, and any non-2D tensor such as the
patch-embedding conv) are merged by a plain sum, matching the paper's focus on the
transformer's linear layers.

This is a faithful re-implementation of the TSV-M idea for use as a baseline; the
exact numbers should be checked against the authors' repo on the standard CLIP-ViT
benchmark before being quoted as head-to-head SOTA.
"""

from __future__ import annotations

from typing import Optional

import torch

from med_merge.config.schema import MergingConfig
from med_merge.merging.base import BaseMerger
from med_merge.merging.task_vector import TaskVector


def _nearest_orthonormal(mat: torch.Tensor) -> torch.Tensor:
    """Polar factor: the orthonormal matrix closest to ``mat`` (columns) in
    Frobenius norm.  For mat = U S V^T this is U V^T."""
    u, _, vh = torch.linalg.svd(mat, full_matrices=False)
    return u @ vh


class TSVMerger(BaseMerger):
    """Task Singular Vectors merge (whitened low-rank subspace recombination)."""

    def merge(
        self,
        task_vectors: dict[str, TaskVector],
        alpha: Optional[float] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        alpha = alpha if alpha is not None else self.config.alpha
        if alpha is None:
            alpha = 1.0

        names = list(task_vectors.keys())
        keys = list(task_vectors[names[0]].vector.keys())
        n_tasks = len(names)
        merged_vector: dict[str, torch.Tensor] = {}

        for key in keys:
            mats = [task_vectors[n].vector[key] for n in names if key in task_vectors[n].vector]
            if not mats:
                continue
            ref = mats[0]

            if ref.ndim != 2 or min(ref.shape) <= 1 or len(mats) == 1:
                # Non-matrix params (or single task): plain sum.
                summed = mats[0].clone().double()
                for m in mats[1:]:
                    summed = summed + m.double()
                merged_vector[key] = summed.reshape(ref.shape).to(ref.dtype)
                continue

            m_rows, n_cols = ref.shape
            full_rank = min(m_rows, n_cols)
            k = max(1, full_rank // n_tasks)

            u_blocks, v_blocks, s_blocks = [], [], []
            for mat in mats:
                u, s, vh = torch.linalg.svd(mat.double(), full_matrices=False)
                u_blocks.append(u[:, :k])           # (m, k)
                v_blocks.append(vh[:k, :].t())      # (n, k)
                s_blocks.append(s[:k])              # (k,)

            u_cat = torch.cat(u_blocks, dim=1)      # (m, T*k)
            v_cat = torch.cat(v_blocks, dim=1)      # (n, T*k)
            s_cat = torch.cat(s_blocks, dim=0)      # (T*k,)

            # Whiten: decorrelate the cross-task singular directions.
            u_white = _nearest_orthonormal(u_cat)   # (m, T*k)
            v_white = _nearest_orthonormal(v_cat)   # (n, T*k)

            merged = (u_white * s_cat) @ v_white.t()  # (m, n)
            merged_vector[key] = merged.reshape(ref.shape).to(ref.dtype)

        tv = TaskVector(vector=merged_vector)
        return tv.apply_to(self.pretrained, scaling_coef=alpha)

    @property
    def name(self) -> str:
        return "tsv_merge"

    @property
    def hyperparameters(self) -> dict:
        return {"alpha": self.config.alpha}
