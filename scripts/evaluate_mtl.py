"""Recover MTL test-set metrics from saved encoders/heads.

Used when MTL training completed but failed to write best_metrics.json (an
earlier version crashed at the last LR-scheduler step after the best-epoch
encoder + heads were already saved). Loads encoder.pt + heads/ under
outputs/{alias}/seed_{seed}/mtl/, evaluates each backbone on every dataset's
test split, writes results.json in the same format as the merging results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.config.schema import DatasetConfig, EvaluationConfig, ModelConfig
from med_merge.evaluation.evaluator import Evaluator
from med_merge.models.factory import BACKBONE_ALIAS, BACKBONE_REGISTRY

ALIAS_TO_HF = {v: k for k, v in BACKBONE_ALIAS.items()}

DATASETS = ["isic2017", "chexpert", "tcga", "nih_cxr"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_dataset_config(name: str) -> DatasetConfig:
    yaml_path = CONFIGS_DIR / "datasets" / f"{name}.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    return DatasetConfig.model_validate(raw["dataset"])


def find_mtl_dir(alias: str, seed: int) -> Path | None:
    """Return the actual on-disk mtl dir (the buggy path), or None if absent."""
    hf_id = ALIAS_TO_HF[alias]
    bad_path = PROJECT_ROOT / "outputs" / hf_id.split("/")[-1] / f"seed_{seed}" / "mtl"
    if bad_path.exists():
        return bad_path
    good_path = PROJECT_ROOT / "outputs" / alias / f"seed_{seed}" / "mtl"
    if good_path.exists():
        return good_path
    return None


def evaluate_one_backbone(alias: str, seed: int, device: str) -> dict | None:
    src_dir = find_mtl_dir(alias, seed)
    if src_dir is None:
        print(f"[{alias}] no mtl/ dir found, skipping")
        return None

    encoder_path = src_dir / "encoder.pt"
    if not encoder_path.exists():
        print(f"[{alias}] no encoder.pt at {encoder_path}, skipping")
        return None

    hf_id = ALIAS_TO_HF[alias]
    _, hidden_size, num_layers = BACKBONE_REGISTRY[hf_id]
    model_config = ModelConfig(
        backbone=hf_id, hidden_size=hidden_size, num_layers=num_layers,
    )

    encoder_sd = torch.load(encoder_path, map_location="cpu", weights_only=True)

    heads: dict[str, dict[str, torch.Tensor]] = {}
    for ds in DATASETS:
        head_path = src_dir / "heads" / ds / "head.pt"
        if head_path.exists():
            heads[ds] = torch.load(head_path, map_location="cpu", weights_only=True)

    ds_configs = {ds: load_dataset_config(ds) for ds in DATASETS}

    eval_config = EvaluationConfig(
        batch_size=64,
        num_workers=4,
        num_calibration_bins=15,
    )
    evaluator = Evaluator(eval_config, model_config=model_config, device=device)

    print(f"[{alias}] evaluating on {list(heads.keys())} ...")
    raw_results = evaluator.evaluate_all(encoder_sd, ds_configs, heads)

    # Flatten to the same structure as merging results.json
    flat = {}
    for ds_name, r in raw_results.items():
        m = dict(r["metrics"])
        # Calibration is also included in metrics; results.json convention
        # keeps it at the same level
        flat[ds_name] = m

    out_dir = PROJECT_ROOT / "outputs" / alias / f"seed_{seed}" / "mtl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(flat, indent=2, default=float))
    print(f"[{alias}] wrote {out_path}")
    return flat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--backbones", nargs="*", default=list(ALIAS_TO_HF.keys()))
    args = p.parse_args()

    summary = {}
    for alias in args.backbones:
        if alias not in ALIAS_TO_HF:
            print(f"Unknown backbone alias {alias!r}, skipping")
            continue
        out = evaluate_one_backbone(alias, args.seed, args.device)
        if out is not None:
            summary[alias] = out

    if summary:
        # One combined snapshot for the paper table
        combined_path = PROJECT_ROOT / "outputs" / f"mtl_summary_seed{args.seed}.json"
        combined_path.write_text(json.dumps(summary, indent=2, default=float))
        print(f"\nWrote combined summary: {combined_path}")


if __name__ == "__main__":
    main()
