"""Aggregate tuned-MTL sweep results across (backbone, seed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY = {
    "isic2017": "balanced_accuracy",
    "chexpert": "macro_auroc",
    "tcga": "auroc",
    "nih_cxr": "macro_auroc",
}

BACKBONES = ["clip", "vit", "dinov3", "rad_dino"]
SEEDS = [42, 123, 456]
DATASETS = ["isic2017", "chexpert", "tcga", "nih_cxr"]


def load_one(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"WARN: failed to parse {path}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs_tuned")
    args = ap.parse_args()
    root = Path(args.output_dir)

    rows: list[dict] = []
    for bb in BACKBONES:
        for seed in SEEDS:
            path = root / bb / f"seed_{seed}" / "mtl" / "best_metrics.json"
            data = load_one(path)
            if data is None:
                rows.append({"backbone": bb, "seed": seed, "missing": True})
                continue
            per_ds = data.get("per_dataset", {})
            best_epoch = max(
                (m.get("epoch", -1) for m in per_ds.values()),
                default=-1,
            )
            row = {
                "backbone": bb,
                "seed": seed,
                "missing": False,
                "aggregate": data.get("best_aggregate"),
                "best_epoch": int(best_epoch),
            }
            for ds in DATASETS:
                row[ds] = per_ds.get(ds, {}).get(PRIMARY[ds])
            rows.append(row)

    print("\nPer-(backbone, seed) results")
    print("-" * 96)
    header = f"{'backbone':<10} {'seed':>5} {'best_ep':>8} {'aggregate':>10}  " + \
             "  ".join(f"{ds:>10}" for ds in DATASETS)
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["missing"]:
            print(f"{r['backbone']:<10} {r['seed']:>5} {'--':>8} {'MISSING':>10}")
            continue
        agg = r["aggregate"]
        agg_s = f"{agg:.4f}" if agg is not None else "n/a"
        cells = [
            f"{r[ds]:.4f}" if r[ds] is not None else "n/a"
            for ds in DATASETS
        ]
        print(
            f"{r['backbone']:<10} {r['seed']:>5} {r['best_epoch']:>8} "
            f"{agg_s:>10}  " + "  ".join(f"{c:>10}" for c in cells)
        )

    print("\n3-seed mean ± std per backbone (primary metric per dataset)")
    print("-" * 96)
    header = f"{'backbone':<10} {'n_seeds':>8} " + \
             "  ".join(f"{ds:>16}" for ds in DATASETS) + f"  {'aggregate':>16}"
    print(header)
    print("-" * len(header))
    for bb in BACKBONES:
        these = [r for r in rows if r["backbone"] == bb and not r["missing"]]
        n = len(these)
        if n == 0:
            print(f"{bb:<10} {n:>8}  (no completed runs)")
            continue
        cells = []
        for ds in DATASETS:
            vals = [r[ds] for r in these if r[ds] is not None]
            if not vals:
                cells.append("n/a")
            else:
                m, s = float(np.mean(vals)), float(np.std(vals))
                cells.append(f"{m:.3f}\xb1{s:.3f}")
        agg_vals = [r["aggregate"] for r in these if r["aggregate"] is not None]
        agg_cell = (
            f"{np.mean(agg_vals):.3f}\xb1{np.std(agg_vals):.3f}"
            if agg_vals else "n/a"
        )
        print(
            f"{bb:<10} {n:>8}  " +
            "  ".join(f"{c:>16}" for c in cells) +
            f"  {agg_cell:>16}"
        )

    print("\nLaTeX rows (per backbone, mean only; copy into ceiling table)")
    print("-" * 96)
    for bb in BACKBONES:
        these = [r for r in rows if r["backbone"] == bb and not r["missing"]]
        if not these:
            print(f"% {bb}: no completed runs")
            continue
        cells = []
        for ds in DATASETS:
            vals = [r[ds] for r in these if r[ds] is not None]
            cells.append(f"{np.mean(vals):.3f}" if vals else "--")
        bb_label = {"clip": "CLIP", "vit": "ViT",
                    "dinov3": "DINOv3", "rad_dino": "Rad-DINO"}[bb]
        print(f"{bb_label} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
