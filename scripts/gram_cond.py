"""Measure conditioning of the task-vector Gram matrix G per weight matrix.

For each arm (native-task vs controlled binary) and each (backbone, seed) cell, load the
per-dataset task vectors, form G_ij = <tau_i, tau_j> per parameter key, and report the
distribution of condition numbers kappa(G) = sigma_max / sigma_min.

This is the direct measurement behind the paper's claim that Gram-LS fails when task
vectors are ill-conditioned: if the mechanism is real, kappa(G) should be markedly larger
in the arm where Gram-LS collapses.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

import torch


def load_tv(path: str) -> dict[str, torch.Tensor]:
    """Load a task vector, tolerating a few common on-disk layouts."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("vector", "state_dict", "task_vector"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    if hasattr(obj, "vector"):
        return obj.vector
    raise ValueError(f"unrecognized task-vector format: {path}")


def arm_condition_numbers(tv_dir: str, datasets: list[str]) -> dict:
    tvs = {}
    for ds in datasets:
        hits = sorted(glob.glob(os.path.join(tv_dir, ds, "*.pt")))
        if not hits:
            return {}
        tvs[ds] = load_tv(hits[0])

    names = list(tvs.keys())
    keys = [k for k in tvs[names[0]] if tvs[names[0]][k].ndim == 2]

    conds, offdiag_ratio = [], []
    for k in keys:
        mats = [tvs[n][k].double().flatten() for n in names]
        n = len(mats)
        G = torch.empty(n, n, dtype=torch.float64)
        for i in range(n):
            for j in range(n):
                G[i, j] = torch.dot(mats[i], mats[j])
        sv = torch.linalg.svdvals(G)
        if sv[-1] <= 0:
            continue
        conds.append((sv[0] / sv[-1]).item())
        d = torch.sqrt(torch.diag(G))
        C = G / torch.outer(d, d)  # normalized -> cosine similarity
        off = C[~torch.eye(n, dtype=bool)].abs()
        offdiag_ratio.append(off.mean().item())

    if not conds:
        return {}
    conds.sort()
    return {
        "n_matrices": len(conds),
        "kappa_median": st.median(conds),
        "kappa_p90": conds[int(0.9 * (len(conds) - 1))],
        "kappa_max": conds[-1],
        "mean_abs_cosine": st.mean(offdiag_ratio),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--backbones", nargs="+",
                    default=["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"])
    ap.add_argument("--seeds", nargs="+", default=["42", "123", "456"])
    ap.add_argument("--out", default="gram_conditioning.json")
    args = ap.parse_args()

    arms = {
        "native_task": {"suffix": "", "datasets": ["isic2017", "chexpert", "pathmnist"]},
        "binary": {"suffix": "_cap2k", "datasets": ["isic_mel", "chexpert_pe", "patchcamelyon"]},
    }

    results: dict[str, list] = {a: [] for a in arms}
    for arm, spec in arms.items():
        for bb in args.backbones:
            for seed in args.seeds:
                tv_dir = os.path.join(args.root, bb, f"seed_{seed}{spec['suffix']}", "task_vectors")
                if not os.path.isdir(tv_dir):
                    continue
                stats = arm_condition_numbers(tv_dir, spec["datasets"])
                if stats:
                    stats.update(backbone=bb, seed=seed)
                    results[arm].append(stats)

    print(f"{'arm':<12} {'cells':>5} {'median kappa':>14} {'p90 kappa':>12} {'mean|cos|':>10}")
    summary = {}
    for arm, rows in results.items():
        if not rows:
            print(f"{arm:<12} {'0':>5}   (no cells found)")
            continue
        med = st.median([r["kappa_median"] for r in rows])
        p90 = st.median([r["kappa_p90"] for r in rows])
        cos = st.mean([r["mean_abs_cosine"] for r in rows])
        summary[arm] = {"cells": len(rows), "kappa_median": med, "kappa_p90": p90,
                        "mean_abs_cosine": cos}
        print(f"{arm:<12} {len(rows):>5} {med:>14.1f} {p90:>12.1f} {cos:>10.3f}")

    # Per-backbone detail for the native arm, where the collapse is reported.
    print("\nper-cell median kappa")
    for arm, rows in results.items():
        for r in sorted(rows, key=lambda x: (x["backbone"], x["seed"])):
            print(f"  {arm:<12} {r['backbone']:<10} seed{r['seed']:<5} "
                  f"kappa_med={r['kappa_median']:>10.1f}  kappa_p90={r['kappa_p90']:>10.1f}")

    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "cells": results}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
