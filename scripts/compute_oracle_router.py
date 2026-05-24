"""Oracle router baseline.

For each (backbone, dataset), evaluates the SPECIALIST model (its own encoder
+ its own head) on the dataset's test split. 

Writes:
    outputs/_figures/oracle_router_metrics.csv
    outputs/_figures/oracle_router_table.md
"""

from __future__ import annotations

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

BACKBONES = ["clip", "vit", "dinov3", "rad_dino"]
DATASETS = ["isic2017", "chexpert", "tcga", "nih_cxr"]
SEED = 42

OUT_DIR = Path("outputs/_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    for bb in BACKBONES:
        # Load the backbone's saved model_config from any one checkpoint
        cfg_path = None
        for ds in DATASETS:
            p = Path(f"outputs/{bb}/seed_{SEED}/checkpoints/{ds}/model_config.json")
            if p.exists():
                cfg_path = p
                break
        if cfg_path is None:
            print(f"[{bb}] no model_config.json found; skipping")
            continue
        model_config = ModelConfig.model_validate(json.loads(cfg_path.read_text()))
        print(f"[{bb}] using backbone={model_config.backbone}")

        evaluator = Evaluator(EvaluationConfig(), model_config, device="cuda")

        for ds in DATASETS:
            ckpt_dir = Path(f"outputs/{bb}/seed_{SEED}/checkpoints/{ds}")
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
                "backbone": bb, "dataset": ds,
                "primary_metric": primary_key,
                "primary": primary,
                "ece": ece, "brier": brier,
                "all_metrics": metrics,
            })

    # Save CSV
    import csv
    csv_path = OUT_DIR / "oracle_router_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "dataset", "primary_metric", "value", "ece", "brier"])
        for r in rows:
            w.writerow([r["backbone"], r["dataset"], r["primary_metric"],
                        round(r["primary"], 4), round(r["ece"], 4), round(r["brier"], 4)])
    print(f"\nWrote {csv_path}")

    # Markdown table (one row per backbone)
    md_path = OUT_DIR / "oracle_router_table.md"
    with open(md_path, "w") as f:
        f.write("# Oracle Router (per-dataset specialist on test set)\n\n")
        f.write("Upper bound for merged-model performance: 'if we knew which dataset each test sample belonged to, route to that specialist'.\n\n")
        f.write("| Backbone | ISIC bal_acc | CheXpert macro_auroc | TCGA auroc | NIH macro_auroc | Aggregate |\n")
        f.write("|---|---|---|---|---|---|\n")
        for bb in BACKBONES:
            cells = [bb]
            vals = []
            for ds in DATASETS:
                row = next((r for r in rows if r["backbone"] == bb and r["dataset"] == ds), None)
                if row:
                    cells.append(f"{row['primary']:.3f}")
                    vals.append(row["primary"])
                else:
                    cells.append("—")
            agg = sum(vals) / len(vals) if vals else 0.0
            cells.append(f"{agg:.3f}")
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"Wrote {md_path}")
    print("\n=== Oracle router summary ===")
    with open(md_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
