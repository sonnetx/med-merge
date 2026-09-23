"""Build the tuning-free tier of merged encoders for one (backbone, seed) cell.

Every method here consumes zero validation data and runs zero search. The point is to ask
whether any of them matches the tuned leading cluster, whose members each burned 5 to 99
validation trials, and which we showed can be reordered by the width of the search grid.

Methods:
  ta_sum_notune   plain sum, c = 1, alpha = 1
  gram_ls_notune  c = G^-1 diag(G) per parameter matrix, alpha = 1, lambda = 0 (the
                  projection-preserving objective pins the scale, so nothing is tuned)
  metagpt_norm    per-task scalar c_t = ||tau_t||^2 / sum_k ||tau_k||^2 (MetaGPT)
  nan_invnorm     per-task scalar c_t proportional to 1 / ||tau_t||, normalized (NAN)

simple_avg (c = 1/N) is already in the main table and needs no recomputation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.config.schema import DatasetConfig, MergingConfig, ModelConfig
from med_merge.merging.gram_ls import GramLSMerger
from med_merge.merging.task_arithmetic import TaskArithmeticMerger
from med_merge.merging.task_vector import TaskVector
from med_merge.models.factory import BACKBONE_REGISTRY, create_model

TASKS = ["isic_mel", "chexpert_pe", "patchcamelyon"]


def load_pretrained(bid: str) -> dict[str, torch.Tensor]:
    try:
        from med_merge.pipelines import load_pretrained_encoder
        mc = ModelConfig(backbone=bid, hidden_size=BACKBONE_REGISTRY[bid][1], num_layers=12)
        return load_pretrained_encoder(mc)
    except Exception:
        mc = ModelConfig(backbone=bid, hidden_size=BACKBONE_REGISTRY[bid][1], num_layers=12)
        dc = DatasetConfig(name="isic_mel", num_classes=1, task_type="binary")
        return create_model(dc, model_config=mc).get_encoder_state_dict()


def scalar_coef_merge(pretrained, tvs, coefs):
    """Merged encoder from per-task global scalars, applied at alpha = 1."""
    names = list(tvs)
    merged = {k: sum(coefs[n] * tvs[n].vector[k] for n in names)
              for k in tvs[names[0]].vector}
    return TaskVector(vector=merged).apply_to(pretrained, scaling_coef=1.0)


def global_gram_coefs(tvs):
    """Whole-vector Gram coefficients, for logging how far c sits from 1."""
    names = list(tvs)
    flat = {n: torch.cat([v.flatten() for v in tvs[n].vector.values()]).double() for n in names}
    G = torch.tensor([[float(flat[a] @ flat[b]) for b in names] for a in names],
                     dtype=torch.float64)
    c = torch.linalg.solve(G, torch.diag(G))
    return {n: float(c[i]) for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-id", required=True)
    ap.add_argument("--cell", required=True, help="outputs/<bb>/seed_<s>_cap2k")
    args = ap.parse_args()

    tvs = {}
    for t in TASKS:
        p = f"{args.cell}/task_vectors/{t}/task_vector.pt"
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        tvs[t] = TaskVector.load(p)

    pretrained = load_pretrained(args.backbone_id)

    norms2 = {t: float(sum((v.double() ** 2).sum() for v in tvs[t].vector.values()))
              for t in TASKS}
    tot2 = sum(norms2.values())
    inv = {t: 1.0 / (norms2[t] ** 0.5) for t in TASKS}
    tot_inv = sum(inv.values())

    outputs = {
        "ta_sum_notune": TaskArithmeticMerger(
            pretrained, MergingConfig(method="task_arithmetic")
        ).merge(tvs, alpha=1.0),
        "gram_ls_notune": GramLSMerger(
            pretrained, MergingConfig(method="gram_ls")
        ).merge(tvs, alpha=1.0, gram_lambda=0.0),
        "metagpt_norm": scalar_coef_merge(
            pretrained, tvs, {t: norms2[t] / tot2 for t in TASKS}),
        "nan_invnorm": scalar_coef_merge(
            pretrained, tvs, {t: inv[t] / tot_inv for t in TASKS}),
    }

    log = {"global_gram_c": global_gram_coefs(tvs),
           "norms": {t: norms2[t] ** 0.5 for t in TASKS},
           "metagpt_c": {t: norms2[t] / tot2 for t in TASKS},
           "nan_c": {t: inv[t] / tot_inv for t in TASKS}}

    for name, state in outputs.items():
        d = f"{args.cell}/merged_binary_tier/{name}"
        os.makedirs(d, exist_ok=True)
        torch.save(state, f"{d}/merged_encoder.pt")
        print(f"wrote {d}/merged_encoder.pt")
    with open(f"{args.cell}/merged_binary_tier/coefficients.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print("coefficients:", json.dumps(log["global_gram_c"]))


if __name__ == "__main__":
    main()
