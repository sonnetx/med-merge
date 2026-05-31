"""Per-layer task-vector cosine similarity.

For each backbone, computes pairwise cosine similarity between task vectors
restricted to each transformer layer's parameters. Shows WHICH layers
drive the cross-task disagreement that predicts merge failure.

Outputs:
    outputs/_figures/per_layer_cosine_{backbone}.png  (heatmap of layer × task-pair)
    outputs/_figures/per_layer_cosine_summary.csv     (mean cosine sim per layer)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.merging.task_vector import TaskVector

BACKBONES = ["clip", "vit", "dinov3", "rad_dino"]
DATASETS = ["isic2017", "chexpert", "tcga", "nih_cxr"]
SEED = 42

LAYER_PATTERNS = {
    "clip": r"layers\.(\d+)\.",
    "vit": r"layer\.(\d+)\.",
    "dinov3": r"layer\.(\d+)\.",
    "rad_dino": r"layer\.(\d+)\.",
}

OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_task_vectors(backbone: str) -> dict[str, TaskVector]:
    tvs = {}
    for ds in DATASETS:
        path = Path(f"outputs/{backbone}/seed_{SEED}/task_vectors/{ds}/task_vector.pt")
        if path.exists():
            tvs[ds] = TaskVector.load(path)
    return tvs


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    na = torch.norm(a)
    nb = torch.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


def per_layer_cosine_matrices(tvs: dict[str, TaskVector], pattern: str) -> dict[int, np.ndarray]:
    """Return {layer_idx: NxN cosine matrix}."""
    layer_keys: dict[int, list[str]] = {}
    sample_keys = next(iter(tvs.values())).vector.keys()
    for k in sample_keys:
        m = re.search(pattern, k)
        if m:
            layer_keys.setdefault(int(m.group(1)), []).append(k)

    ds_names = list(tvs.keys())
    n = len(ds_names)
    out = {}
    for layer_idx, keys in sorted(layer_keys.items()):
        flats = {}
        for ds in ds_names:
            parts = [tvs[ds].vector[k].detach().cpu().flatten() for k in keys
                     if k in tvs[ds].vector]
            flats[ds] = torch.cat(parts) if parts else torch.zeros(0)
        mat = np.zeros((n, n))
        for i, a in enumerate(ds_names):
            for j, b in enumerate(ds_names):
                mat[i, j] = cosine(flats[a], flats[b])
        out[layer_idx] = mat
    return out


def main():
    summary_rows = []
    for bb in BACKBONES:
        tvs = load_task_vectors(bb)
        if len(tvs) < 2:
            print(f"[{bb}] need >=2 task vectors; have {len(tvs)}, skipping")
            continue
        ds_names = list(tvs.keys())
        pattern = LAYER_PATTERNS.get(bb, r"layer\.(\d+)\.")
        mats = per_layer_cosine_matrices(tvs, pattern)
        if not mats:
            print(f"[{bb}] no layer keys matched pattern {pattern!r}; skipping")
            continue

        layers = sorted(mats.keys())
        pair_labels = []
        pair_data = []
        for i in range(len(ds_names)):
            for j in range(i + 1, len(ds_names)):
                pair_labels.append(f"{ds_names[i][:4]}–{ds_names[j][:4]}")
                pair_data.append([mats[l][i, j] for l in layers])
        data = np.array(pair_data)

        fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.6), 4))
        sns.heatmap(
            data,
            xticklabels=layers,
            yticklabels=pair_labels,
            annot=True, fmt=".2f",
            cmap="viridis", vmin=-0.3, vmax=1.0,
            cbar_kws={"label": "cosine sim"},
            ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_xlabel("Transformer layer index (0 = shallowest)")
        ax.set_ylabel("Task pair")
        ax.set_title(f"Per-layer task-vector cosine similarity — {bb}")
        fig.tight_layout()
        out_path = OUT_DIR / f"per_layer_cosine_{bb}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")

        for li, layer in enumerate(layers):
            off_diag = []
            for i in range(len(ds_names)):
                for j in range(i + 1, len(ds_names)):
                    off_diag.append(mats[layer][i, j])
            mean_cos = float(np.mean(off_diag))
            min_idx = int(np.argmin(off_diag))
            i, j = next(((a, b) for cnt, (a, b) in enumerate(
                [(x, y) for x in range(len(ds_names)) for y in range(x + 1, len(ds_names))])
                if cnt == min_idx))
            summary_rows.append({
                "backbone": bb,
                "layer": layer,
                "mean_cos": round(mean_cos, 4),
                "min_cos": round(float(min(off_diag)), 4),
                "min_pair": f"{ds_names[i]}-{ds_names[j]}",
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(OUT_DIR / "per_layer_cosine_summary.csv", index=False)
        print(f"Wrote {OUT_DIR / 'per_layer_cosine_summary.csv'}")

        fig, ax = plt.subplots(figsize=(8, 5))
        for bb, sub in df.groupby("backbone"):
            ax.plot(sub["layer"], sub["mean_cos"], marker="o", label=bb)
        ax.set_xlabel("Transformer layer index")
        ax.set_ylabel("Mean off-diagonal cosine similarity")
        ax.set_title("Task-vector alignment vs layer depth")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "per_layer_cosine_depth.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {OUT_DIR / 'per_layer_cosine_depth.png'}")

        print("\n=== Per-layer cosine summary ===")
        print(df.groupby("backbone").agg(
            mean_cos_mean=("mean_cos", "mean"),
            mean_cos_min=("mean_cos", "min"),
            mean_cos_max=("mean_cos", "max"),
        ).round(3))


if __name__ == "__main__":
    main()
