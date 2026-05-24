"""Compute task-vector geometry features for all (backbone, seed) combos.

For each (backbone, seed), reads the saved per-dataset task vectors and writes:
  - analysis/task_vector_cosine.csv  (N x N cosine matrix; overwrites existing)
  - analysis/task_vector_l2.csv      (N x N L2-distance matrix)
  - analysis/task_vector_norms.csv   (per-dataset L2 norm)
  - analysis/geometry_features.csv   (one row per dataset with all scalars
                                       used by predict_forgetting_lobo.py)

Run from repo root:
    python scripts/compute_geometry.py [--seeds 42 123 456]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.config.constants import ALL_DATASETS, SEEDS
from med_merge.merging.task_vector import TaskVector

BACKBONES = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]


def flatten(tv: TaskVector) -> torch.Tensor:
    return torch.cat([v.detach().cpu().float().flatten() for v in tv.vector.values()])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm().item(), b.norm().item()
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((a @ b).item() / (na * nb))


def load_tvs(backbone: str, seed: int) -> dict[str, TaskVector]:
    out: dict[str, TaskVector] = {}
    for ds in ALL_DATASETS:
        p = Path(f"outputs/{backbone}/seed_{seed}/task_vectors/{ds}/task_vector.pt")
        if p.exists():
            out[ds] = TaskVector.load(p)
    return out


def compute_one(backbone: str, seed: int) -> None:
    tvs = load_tvs(backbone, seed)
    if len(tvs) < 2:
        print(f"[{backbone}/seed_{seed}] only {len(tvs)} task vectors, skipping")
        return
    out_dir = Path(f"outputs/{backbone}/seed_{seed}/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(tvs.keys())
    flats = {n: flatten(tvs[n]) for n in names}
    norms = {n: flats[n].norm().item() for n in names}

    # Cosine matrix
    cos = pd.DataFrame(index=names, columns=names, dtype=float)
    l2 = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                cos.loc[a, b] = 1.0
                l2.loc[a, b] = 0.0
            else:
                cos.loc[a, b] = cosine(flats[a], flats[b])
                l2.loc[a, b] = float((flats[a] - flats[b]).norm().item())

    cos.to_csv(out_dir / "task_vector_cosine.csv")
    l2.to_csv(out_dir / "task_vector_l2.csv")

    norms_df = pd.DataFrame({"dataset": names, "norm": [norms[n] for n in names]})
    norms_df.to_csv(out_dir / "task_vector_norms.csv", index=False)

    # Per-dataset scalar features for the predictor
    rows = []
    for d in names:
        others = [x for x in names if x != d]
        cos_to_others = [cos.loc[d, o] for o in others]
        l2_to_others = [l2.loc[d, o] for o in others]
        norms_others = [norms[o] for o in others]
        rows.append({
            "dataset": d,
            "norm": norms[d],
            "mean_cos": float(pd.Series(cos_to_others).mean()),
            "min_cos": float(pd.Series(cos_to_others).min()),
            "max_cos": float(pd.Series(cos_to_others).max()),
            "mean_l2": float(pd.Series(l2_to_others).mean()),
            "max_l2": float(pd.Series(l2_to_others).max()),
            "mean_norm_others": float(pd.Series(norms_others).mean()),
            # The theory-motivated cross-term: norm * mean(|τ_other| * cos)
            "interference": float(sum(norms[d] * no * c
                                       for no, c in zip(norms_others, cos_to_others))),
        })
    feats = pd.DataFrame(rows)
    feats.to_csv(out_dir / "geometry_features.csv", index=False)
    print(f"[{backbone}/seed_{seed}] wrote {len(names)} datasets, "
          f"mean_norm={feats['norm'].mean():.2f}, mean_cos={feats['mean_cos'].mean():.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--backbones", nargs="*", default=BACKBONES)
    args = p.parse_args()

    for bb in args.backbones:
        for s in args.seeds:
            compute_one(bb, s)


if __name__ == "__main__":
    main()
