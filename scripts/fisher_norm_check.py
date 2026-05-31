"""Per-dataset Fisher magnitude check (Appendix C of the paper).

Loads cached Fisher diagonals from outputs/{bb}/seed_{s}/fisher_cache/{ds}/
and tabulates sum and mean |F_i| per (backbone, dataset, seed), plus the
relative magnitude vs the per-backbone minimum. Writes
fisher_norm_table.csv and fisher_norm_bar.png under outputs/_figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.config.constants import ALL_DATASETS, SEEDS

BACKBONES = ["clip", "vit", "dinov3", "rad_dino"]
OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_fisher(bb: str, seed: int, ds: str) -> dict[str, torch.Tensor] | None:
    p = Path(f"outputs/{bb}/seed_{seed}/fisher_cache/{ds}/fisher.pt")
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=True)


def fisher_stats(f: dict[str, torch.Tensor]) -> tuple[float, float, int]:
    total_abs = 0.0
    total_elems = 0
    for v in f.values():
        v32 = v.detach().float().abs()
        total_abs += float(v32.sum().item())
        total_elems += int(v32.numel())
    return total_abs, total_abs / max(total_elems, 1), total_elems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--backbones", nargs="*", default=BACKBONES)
    args = p.parse_args()

    rows = []
    for bb in args.backbones:
        for seed in args.seeds:
            for ds in ALL_DATASETS:
                f = load_fisher(bb, seed, ds)
                if f is None:
                    continue
                total, mean, n = fisher_stats(f)
                rows.append({
                    "backbone": bb, "seed": seed, "dataset": ds,
                    "sum_abs_fisher": total,
                    "mean_abs_fisher": mean,
                    "n_params": n,
                })
    if not rows:
        print("No Fisher caches found. Did the Fisher merging jobs run?")
        return

    df = pd.DataFrame(rows)
    # Relative magnitude: divide by per-(backbone, seed) min so the ratio is
    # comparable across backbones with different param counts.
    df["rel_to_backbone_min"] = (
        df["sum_abs_fisher"] / df.groupby(["backbone", "seed"])["sum_abs_fisher"].transform("min")
    )

    print("\n=== Fisher magnitude per (backbone, seed, dataset) ===")
    print(df.round(2).to_string(index=False))
    df.to_csv(OUT_DIR / "fisher_norm_table.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'fisher_norm_table.csv'}")

    # Summary across seeds: per (backbone, dataset) mean relative magnitude
    summary = (
        df.groupby(["backbone", "dataset"])["rel_to_backbone_min"]
          .mean()
          .reset_index()
          .pivot(index="backbone", columns="dataset", values="rel_to_backbone_min")
    )
    print("\n=== Mean relative Fisher magnitude (vs per-backbone min) ===")
    print(summary.round(2).to_string())

    # Bar plot
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_ylabel("relative |Fisher| (per-backbone min = 1)")
    ax.set_title("Fisher magnitude per dataset, averaged across seeds")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fisher_norm_bar.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'fisher_norm_bar.png'}")

    # Verdict line for the paper:
    isic_mag = summary.loc[:, "isic2017"].mean() if "isic2017" in summary.columns else float("nan")
    other_mag = summary.drop(columns=[c for c in ["isic2017"] if c in summary.columns]).mean(axis=1).mean()
    print(f"\nISIC mean relative magnitude: {isic_mag:.2f}")
    print(f"Non-ISIC mean relative magnitude: {other_mag:.2f}")
    print(f"Ratio: ISIC dominates non-ISIC by {isic_mag / max(other_mag, 1e-12):.2f}x")
    print("\nGuidance:")
    print("  >= 10x  ->  keep the 'Fisher concentrates on small val sets' line in §5")
    print("  <  2x   ->  drop the explanation; Fisher's ISIC win has another cause")


if __name__ == "__main__":
    main()
