"""Final ablation statistics across every completed cell.

Reports, per arm, the Gram-LS minus TSV-Merge gap (mean, sd, range), the variance ratio
between arms, and the correlation between task-vector conditioning and the magnitude of
the gap.
"""

import glob
import json
import math
import os
import statistics as st

import torch

BB = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
SEEDS = ["42", "123", "456"]
TRIO = {
    "binary": ["isic_mel", "chexpert_pe", "pathmnist_bin"],
    "native": ["isic2017", "chexpert", "pathmnist"],
}


def load(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("vector", "state_dict", "task_vector"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    return obj.vector


def kappa(arm, bb, seed):
    tv = {}
    for ds in TRIO[arm]:
        p = f"outputs/{bb}/seed_{seed}_abl_{arm}/task_vectors/{ds}/task_vector.pt"
        if not os.path.exists(p):
            return None
        tv[ds] = load(p)
    first = tv[TRIO[arm][0]]
    conds = []
    for k in [k for k in first if first[k].ndim == 2]:
        vs = [tv[d][k].double().flatten() for d in TRIO[arm]]
        G = torch.tensor([[torch.dot(a, b) for b in vs] for a in vs], dtype=torch.float64)
        sv = torch.linalg.svdvals(G)
        if sv[-1] > 0:
            conds.append((sv[0] / sv[-1]).item())
    return st.median(conds) if conds else None


def score(arm, bb, seed, method):
    p = f"outputs/{bb}/seed_{seed}_abl_{arm}/results/{method}/results.json"
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    vals = [m.get("auroc") if arm == "binary" else m.get("balanced_accuracy", m.get("macro_auroc"))
            for m in r.values()]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if len(vals) == 3 else None


def pearson(x, y):
    n = len(x)
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    if den == 0:
        return float("nan"), float("nan")
    r = num / den
    t = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else float("inf")
    return r, t


rows, gaps = [], {"binary": [], "native": []}
for arm in ("binary", "native"):
    for bb in BB:
        for seed in SEEDS:
            g, t_ = score(arm, bb, seed, "gram_ls"), score(arm, bb, seed, "tsv_merge")
            a = score(arm, bb, seed, "task_arithmetic")
            if g is None or t_ is None:
                continue
            gaps[arm].append(g - t_)
            k = kappa(arm, bb, seed)
            if k is not None:
                rows.append((arm, bb, seed, k, g - t_, (g - a) if a is not None else None))

print("=== Gram-LS minus TSV-Merge, by arm ===")
for arm in ("binary", "native"):
    v = gaps[arm]
    if len(v) > 1:
        print(f"  {arm:<8} n={len(v):<3} mean={st.mean(v):+.4f}  sd={st.stdev(v):.4f}  "
              f"range=[{min(v):+.3f}, {max(v):+.3f}]")

if len(gaps["native"]) > 1 and len(gaps["binary"]) > 1:
    F = (st.stdev(gaps["native"]) / st.stdev(gaps["binary"])) ** 2
    print(f"\n  variance ratio F = {F:.1f}  "
          f"(sd ratio {st.stdev(gaps['native']) / st.stdev(gaps['binary']):.1f}x), "
          f"df=({len(gaps['native'])-1},{len(gaps['binary'])-1})")

print("\n=== kappa(G) by arm ===")
for arm in ("binary", "native"):
    ks = [r[3] for r in rows if r[0] == arm]
    if ks:
        print(f"  {arm:<8} n={len(ks):<3} median={st.median(ks):8.2f}  "
              f"min={min(ks):.2f}  max={max(ks):.2f}")

lk = [math.log10(r[3]) for r in rows]
ab = [abs(r[4]) for r in rows]
sg = [r[4] for r in rows]
r1, t1 = pearson(lk, ab)
r2, t2 = pearson(lk, sg)
print(f"\n=== conditioning vs gap (n={len(rows)}) ===")
print(f"  log10 kappa vs |gap| : r={r1:+.3f}  t={t1:+.2f}")
print(f"  log10 kappa vs  gap  : r={r2:+.3f}  t={t2:+.2f}")

ga = [r[5] for r in rows if r[5] is not None]
if ga:
    print(f"\n  Gram-LS minus task arithmetic, all cells: mean={st.mean(ga):+.4f} (n={len(ga)})")

with open("ablation_final.json", "w") as fh:
    json.dump({"cells": [{"arm": a, "backbone": b, "seed": s, "kappa": k,
                          "gap_tsv": g, "gap_ta": ta} for a, b, s, k, g, ta in rows]}, fh, indent=2)
print("\nwrote ablation_final.json")
