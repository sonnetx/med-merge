"""Does task-vector conditioning predict where Gram-LS loses to low-rank truncation?

For each ablation cell, pair the median per-layer condition number kappa(G) of the
task-vector Gram matrix with the Gram-LS minus TSV-Merge performance gap.
"""

import json
import math
import os
import statistics as st

import torch

BB = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
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


def kappa(arm, bb):
    tv = {}
    for ds in TRIO[arm]:
        p = f"outputs/{bb}/seed_42_abl_{arm}/task_vectors/{ds}/task_vector.pt"
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


def score(arm, bb, method):
    p = f"outputs/{bb}/seed_42_abl_{arm}/results/{method}/results.json"
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    key = "auroc" if arm == "binary" else None
    vals = []
    for mm in r.values():
        vals.append(mm.get("auroc") if key else mm.get("balanced_accuracy", mm.get("macro_auroc")))
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    rows = []
    print(f"{'arm/backbone':<20}{'kappa(G)':>12}{'gram-tsv':>11}")
    for arm in ("binary", "native"):
        for bb in BB:
            k = kappa(arm, bb)
            g, t = score(arm, bb, "gram_ls"), score(arm, bb, "tsv_merge")
            if k is None or g is None or t is None:
                continue
            rows.append((k, g - t))
            print(f"{arm + '/' + bb:<20}{k:>12.2f}{g - t:>+11.4f}")

    n = len(rows)
    if n < 3:
        print("\ntoo few cells for a correlation")
        return
    lk = [math.log10(r[0]) for r in rows]
    gp = [r[1] for r in rows]
    mx, my = st.mean(lk), st.mean(gp)
    num = sum((a - mx) * (b - my) for a, b in zip(lk, gp))
    den = math.sqrt(sum((a - mx) ** 2 for a in lk) * sum((b - my) ** 2 for b in gp))
    r = num / den if den else float("nan")
    t_stat = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else float("inf")
    print(f"\npearson r(log10 kappa, gram_ls - tsv) = {r:+.3f}   n={n}   t={t_stat:+.2f}")


if __name__ == "__main__":
    main()
