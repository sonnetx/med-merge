"""Aggregate the binary-2k study: outputs/{bb}/seed_{s}_cap2k/results_binary/{method}/results.json.

Binary trio (AUROC): isic_mel, chexpert_pe, patchcamelyon. Prints coverage, a
per-method summary (mean +/- std over backbones+seeds), and per-backbone tables.
vit/seed_42 lives at seed_42_cap2k too (the pilot). Works on partial results.
"""
from __future__ import annotations
import json, statistics
import os
from pathlib import Path

PD = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
BACKBONES = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
SEEDS = [42, 123, 456]
METHODS = ["simple_avg", "task_arithmetic", "ties", "dare", "dare_ties",
           "pcb_merging", "lines", "fisher", "iso_c", "tsv_merge", "gram_ls"]
DS = ["isic_mel", "chexpert_pe", "patchcamelyon"]


def load(bb, seed, m):
    p = PD / "outputs" / bb / f"seed_{seed}_cap2k" / "results_binary" / m / "results.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    return {ds: d.get(ds, {}).get("auroc") for ds in DS}


def f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "


done = sum(1 for bb in BACKBONES for s in SEEDS for m in METHODS if load(bb, s, m))
print(f"coverage: {done}/{len(BACKBONES)*len(SEEDS)*len(METHODS)} cells (binary-2k)\n")

print("=== per-method AUROC, mean +/- std over backbones+seeds (n) ===")
hdr = f"{'method':16}" + "".join(f"{d.split('_')[0][:8]:>20}" for d in DS) + f"{'MEAN':>8}"
print(hdr); print("-" * len(hdr))
for m in METHODS:
    cells, means = "", []
    for ds in DS:
        vals = [r[ds] for bb in BACKBONES for s in SEEDS
                if (r := load(bb, s, m)) and isinstance(r.get(ds), (int, float))]
        if vals:
            mean = statistics.mean(vals); sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            means.append(mean); cells += f"{mean:.3f}+/-{sd:.3f}(n={len(vals)})".rjust(20)
        else:
            cells += "-".rjust(20)
    overall = f"{statistics.mean(means):.3f}" if means else "-"
    print(f"{m:16}{cells}{overall:>8}")

for ds in DS:
    print(f"\n-- {ds} (auroc), per-backbone seed-avg --")
    print(f"{'backbone':10}" + "".join(f"{m[:12]:>13}" for m in METHODS))
    for bb in BACKBONES:
        row = f"{bb:10}"
        for m in METHODS:
            vals = [r[ds] for s in SEEDS if (r := load(bb, s, m)) and isinstance(r.get(ds), (int, float))]
            row += f(statistics.mean(vals) if vals else None).rjust(13)
        print(row)
