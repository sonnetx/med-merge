"""Oracle router baseline.

For each (backbone, dataset), evaluates the SPECIALIST model (its own encoder
+ its own head) on the dataset's test split. 

Writes:
    outputs/_figures/oracle_router_metrics.csv
    outputs/_figures/oracle_router_table.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.config.constants import DATASET_DEFAULTS, PRIMARY_METRICS
from med_merge.config.schema import DatasetConfig, EvaluationConfig, ModelConfig
from med_merge.evaluation.evaluator import Evaluator
from med_merge.pipelines import _load_dataset_config
from med_merge.utils.io import load_state_dict

DEFAULT_BACKBONES = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
DEFAULT_DATASETS = ["isic2017", "chexpert", "tcga", "nih_cxr"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", nargs="+", default=["42"])
    ap.add_argument("--cell-suffix", default="",
                    help="suffix on the seed directory, e.g. _cap2k for the binary trio")
    ap.add_argument("--out-prefix", default="oracle_router",
                    help="basename for the CSV/markdown written under outputs/_figures")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    BACKBONES, DATASETS = args.backbones, args.datasets
    OUT_DIR = Path("outputs/_figures")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for bb in BACKBONES:
        # Load the backbone's saved model_config from any one checkpoint
        cfg_path = None
        for seed in args.seeds:
            for ds in DATASETS:
                p = Path(f"outputs/{bb}/seed_{seed}{args.cell_suffix}/checkpoints/{ds}/model_config.json")
                if p.exists():
                    cfg_path = p
                    break
            if cfg_path is not None:
                break
        if cfg_path is None:
            print(f"[{bb}] no model_config.json found; skipping")
            continue
        model_config = ModelConfig.model_validate(json.loads(cfg_path.read_text()))
        print(f"[{bb}] using backbone={model_config.backbone}")

        evaluator = Evaluator(EvaluationConfig(), model_config, device=args.device)

        for seed in args.seeds:
          for ds in DATASETS:
            ckpt_dir = Path(f"outputs/{bb}/seed_{seed}{args.cell_suffix}/checkpoints/{ds}")
            head_path = ckpt_dir / "head.pt"
            best_model_path = ckpt_dir / "best_model.pt"

            if not head_path.exists() or not best_model_path.exists():
                print(f"  [{bb}/{ds}] missing checkpoint; skipping")
                continue

            # Reconstruct the specialist encoder from the saved best_model state
            state = torch.load(best_model_path, map_location="cpu", weights_only=False)
            full_sd = state["model_state_dict"]
            encoder_sd = {k: v for k, v in full_sd.items() if k.startswith("encoder.")}
            head_sd = load_state_dict(head_path)

            ds_config = _load_dataset_config(ds, "./data")
            print(f"  [{bb}/{ds}] evaluating specialist on test set...")
            result = evaluator.evaluate_single(encoder_sd, ds_config, head_sd)
            metrics = result["metrics"]

            primary_key = PRIMARY_METRICS.get(ds, "accuracy")
            primary = metrics.get(primary_key)
            ece = metrics.get("ece")
            brier = metrics.get("brier")
            print(f"    {primary_key}={primary:.4f}  ece={ece:.4f}  brier={brier:.4f}")

            rows.append({
                "backbone": bb, "seed": seed, "dataset": ds,
                "primary_metric": primary_key,
                "primary": primary,
                "ece": ece, "brier": brier,
                "all_metrics": metrics,
            })

    # Save CSV
    import csv
    csv_path = OUT_DIR / f"{args.out_prefix}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "seed", "dataset", "primary_metric", "value", "ece", "brier"])
        for r in rows:
            w.writerow([r["backbone"], r["seed"], r["dataset"], r["primary_metric"],
                        round(r["primary"], 4), round(r["ece"], 4), round(r["brier"], 4)])
    print(f"\nWrote {csv_path}")

    # Markdown table (one row per backbone)
    md_path = OUT_DIR / f"{args.out_prefix}_table.md"
    with open(md_path, "w") as f:
        f.write("# Oracle Router (per-dataset specialist on test set)\n\n")
        f.write("Upper bound for merged-model performance: 'if we knew which dataset each test sample belonged to, route to that specialist'.\n\n")
        f.write("| Backbone | " + " | ".join(DATASETS) + " | Aggregate |\n")
        f.write("|---" * (len(DATASETS) + 2) + "|\n")
        for bb in BACKBONES:
            cells = [bb]
            vals = []
            for ds in DATASETS:
                # average over seeds for this (backbone, dataset)
                got = [r["primary"] for r in rows
                       if r["backbone"] == bb and r["dataset"] == ds and r["primary"] is not None]
                if got:
                    m = sum(got) / len(got)
                    cells.append(f"{m:.3f}")
                    vals.append(m)
                else:
                    cells.append("n/a")
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "n/a")
            f.write("| " + " | ".join(cells) + " |\n")

        # Column means across backbones, the number quoted as the oracle upper bound.
        f.write("| **mean** | ")
        col_means = []
        for ds in DATASETS:
            got = [r["primary"] for r in rows if r["dataset"] == ds and r["primary"] is not None]
            col_means.append(sum(got) / len(got) if got else None)
        f.write(" | ".join(f"{m:.3f}" if m is not None else "n/a" for m in col_means))
        finite = [m for m in col_means if m is not None]
        f.write(f" | {sum(finite) / len(finite):.3f} |\n" if finite else " | n/a |\n")
    print(f"Wrote {md_path}")
    print("\n=== Oracle router summary ===")
    with open(md_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
