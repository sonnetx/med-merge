"""Compile ECE and Brier scores across all (backbone, method, dataset) combos.

Writes:
    outputs/_figures/calibration_metrics.csv
    outputs/_figures/calibration_table.md
"""

from __future__ import annotations

import json
from pathlib import Path

BACKBONES = ["clip", "vit", "dinov3", "rad_dino"]
METHODS = ["simple_avg", "task_arithmetic", "ties", "dare", "dare_ties",
           "pcb_merging", "lines", "fisher"]
DATASETS = ["isic2017", "chexpert", "tcga", "nih_cxr"]
SEED = 42

OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    for bb in BACKBONES:
        for method in METHODS:
            f = Path(f"outputs/{bb}/seed_{SEED}/results/{method}/results.json")
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            for ds in DATASETS:
                if ds not in r:
                    continue
                m = r[ds]
                rows.append({
                    "backbone": bb,
                    "method": method,
                    "dataset": ds,
                    "ece": m.get("ece"),
                    "brier": m.get("brier"),
                })

    # CSV
    import csv
    csv_path = OUT_DIR / "calibration_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "method", "dataset", "ece", "brier"])
        for r in rows:
            w.writerow([r["backbone"], r["method"], r["dataset"],
                        round(r["ece"], 4) if r["ece"] is not None else "",
                        round(r["brier"], 4) if r["brier"] is not None else ""])
    print(f"Wrote {csv_path}")

    # Markdown: one table per dataset, rows = backbones, cols = methods
    md_path = OUT_DIR / "calibration_table.md"
    with open(md_path, "w") as f:
        f.write("# Calibration metrics (ECE and Brier)\n\n")
        for metric in ["ece", "brier"]:
            f.write(f"\n## {metric.upper()}\n\n")
            for ds in DATASETS:
                f.write(f"\n### {ds}\n\n")
                f.write("| Backbone | " + " | ".join(METHODS) + " |\n")
                f.write("|---" * (len(METHODS) + 1) + "|\n")
                for bb in BACKBONES:
                    cells = [bb]
                    for m in METHODS:
                        row = next((r for r in rows if r["backbone"] == bb
                                    and r["method"] == m and r["dataset"] == ds), None)
                        if row and row[metric] is not None:
                            cells.append(f"{row[metric]:.3f}")
                        else:
                            cells.append("—")
                    f.write("| " + " | ".join(cells) + " |\n")
    print(f"Wrote {md_path}")

    # Print summary
    print("\n=== Calibration summary (mean ECE per dataset across methods/backbones) ===")
    for ds in DATASETS:
        eces = [r["ece"] for r in rows if r["dataset"] == ds and r["ece"] is not None]
        if eces:
            print(f"  {ds}: mean ECE = {sum(eces)/len(eces):.3f}, range [{min(eces):.3f}, {max(eces):.3f}]")


if __name__ == "__main__":
    main()
