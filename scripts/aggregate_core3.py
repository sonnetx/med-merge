"""Aggregate 3-core merge results across backbones/seeds/methods.

Scans outputs/{backbone}/seed_{seed}/results_core3/{method}/results.json and prints
per-dataset primary metrics: a per-(backbone,method) table averaged over seeds, and
a per-method summary averaged over backbones+seeds. Works on partial results.

Run on Sherlock:  python3 scripts/aggregate_core3.py
"""
from __future__ import annotations
import json, statistics
from pathlib import Path

PD = Path(__file__).resolve().parent.parent
OUT = PD / "outputs"
BACKBONES = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
SEEDS = [42, 123, 456]
METHODS = ["simple_avg", "task_arithmetic", "iso_c", "tsv_merge", "gram_ls", "gram_ls_na"]
PRIMARY = {"isic2017": "balanced_accuracy", "chexpert": "macro_auroc", "pathmnist": "balanced_accuracy"}
DSES = list(PRIMARY)


def load(bb, seed, method):
    p = OUT / bb / f"seed_{seed}" / "results_core3" / method / "results.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    return {ds: d.get(ds, {}).get(PRIMARY[ds]) for ds in DSES}


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "


# coverage
done = sum(1 for bb in BACKBONES for s in SEEDS for m in METHODS if load(bb, s, m))
total = len(BACKBONES) * len(SEEDS) * len(METHODS)
print(f"coverage: {done}/{total} (backbone x seed x method) cells complete\n")

# per-method summary averaged over all backbones+seeds
print("=== per-method, averaged over backbones+seeds (mean +/- std, n) ===")
hdr = f"{'method':16}" + "".join(f"{ds:>22}" for ds in DSES)
print(hdr); print("-" * len(hdr))
for m in METHODS:
    cells = ""
    for ds in DSES:
        vals = [r[ds] for bb in BACKBONES for s in SEEDS
                if (r := load(bb, s, m)) and isinstance(r.get(ds), (int, float))]
        if vals:
            mean = statistics.mean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            cells += f"{mean:.3f}+/-{sd:.3f}(n={len(vals)})".rjust(22)
        else:
            cells += "-".rjust(22)
    print(f"{m:16}{cells}")

# per-backbone x method, seed-averaged
print("\n=== per-backbone x method (seed-averaged primary metric) ===")
for ds in DSES:
    print(f"\n-- {ds} ({PRIMARY[ds]}) --")
    print(f"{'backbone':10}" + "".join(f"{m:>16}" for m in METHODS))
    for bb in BACKBONES:
        row = f"{bb:10}"
        for m in METHODS:
            vals = [r[ds] for s in SEEDS if (r := load(bb, s, m)) and isinstance(r.get(ds), (int, float))]
            row += fmt(statistics.mean(vals) if vals else None).rjust(16)
        print(row)
