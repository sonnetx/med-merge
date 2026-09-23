"""Does merging buy representation quality on a domain no specialist has seen?

Routing can only serve domains that have a specialist. The merging literature's strongest
claim against that is transfer to held-out tasks (TIES, RegMean, Model Soups). We test it
here on a fourth medical domain absent from the merge, fundus photography, by linear-probing
frozen encoders. Comparing the merged encoder against each individual specialist and against
the untouched pretrained backbone isolates what merging contributes beyond its own tasks.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from med_merge.config.schema import DatasetConfig, ModelConfig
from med_merge.data.registry import build_dataset
from med_merge.data.transforms import get_eval_transform
from med_merge.models.factory import BACKBONE_REGISTRY, create_model

BACKBONES = {
    "clip": "openai/clip-vit-base-patch16",
    "vit": "google/vit-base-patch16-224",
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "rad_dino": "microsoft/rad-dino",
    "dinov2": "facebook/dinov2-base",
    "mae": "facebook/vit-mae-base",
    "beit": "microsoft/beit-base-patch16-224-pt22k-ft22k",
}
SPECIALISTS = ["isic_mel", "chexpert_pe", "patchcamelyon"]
MERGED = ["task_arithmetic", "gram_ls", "tsv_merge", "iso_c"]


def features(model, loader, device):
    xs, ys = [], []
    model.eval()
    with torch.no_grad():
        for img, lab in loader:
            f = model.encoder(img.to(device, non_blocking=True))
            if isinstance(f, (tuple, list)):
                f = f[0]
            if f.ndim == 3:            # token sequence, take CLS
                f = f[:, 0]
            xs.append(f.float().cpu().numpy())
            ys.append(lab.reshape(len(lab), -1)[:, 0].cpu().numpy())
    return np.concatenate(xs), np.concatenate(ys)


def probe(tr_x, tr_y, te_x, te_y):
    """Multinomial logistic probe, standardized features, balanced accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(tr_x)
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
    clf.fit(sc.transform(tr_x), tr_y)
    return float(balanced_accuracy_score(te_y, clf.predict(sc.transform(te_x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default="retinamnist")
    ap.add_argument("--data-dir", default="/path/to/data/medmnist")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="heldout_probe.json")
    args = ap.parse_args()

    tf = get_eval_transform(224, "imagenet")
    ds_tr = build_dataset(args.heldout, args.data_dir, split="train", transform=tf)
    ds_te = build_dataset(args.heldout, args.data_dir, split="test", transform=tf)
    ld = lambda d: torch.utils.data.DataLoader(d, batch_size=64, shuffle=False, num_workers=4)
    print(f"held-out {args.heldout}: train={len(ds_tr)} test={len(ds_te)}")

    results = {}
    for bb, bid in BACKBONES.items():
        cell = f"outputs/{bb}/seed_{args.seed}_cap2k"
        if not os.path.isdir(cell):
            continue
        hidden = BACKBONE_REGISTRY[bid][1]
        mc = ModelConfig(backbone=bid, hidden_size=hidden, num_layers=12)
        dc = DatasetConfig(name=args.heldout, num_classes=int(ds_tr.num_classes),
                           task_type="multiclass")
        model = create_model(dc, model_config=mc).to(args.device)
        base = {k: v.clone() for k, v in model.state_dict().items()
                if k.startswith("encoder.")}

        def run(tag, enc_sd):
            try:
                model.load_encoder_state_dict(enc_sd)
            except Exception as e:
                print(f"  [{bb}/{tag}] load failed: {type(e).__name__}: {str(e)[:120]}")
                return
            tr = features(model, ld(ds_tr), args.device)
            te = features(model, ld(ds_te), args.device)
            acc = probe(tr[0], tr[1], te[0], te[1])
            results.setdefault(bb, {})[tag] = acc
            print(f"  [{bb}] {tag:<20} balanced acc = {acc:.4f}")

        run("pretrained", base)

        for ds in SPECIALISTS:
            p = f"{cell}/checkpoints/{ds}/best_model.pt"
            if not os.path.exists(p):
                continue
            st = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
            run(f"specialist:{ds}",
                {k: v for k, v in st.items() if k.startswith("encoder.")})

        for m in MERGED:
            p = f"{cell}/merged_binary/{m}/merged_encoder.pt"
            if not os.path.exists(p):
                continue
            sd = torch.load(p, map_location="cpu", weights_only=False)
            run(f"merged:{m}", sd)

        json.dump(results, open(args.out, "w"), indent=2)

    print("\n=== summary (balanced accuracy on held-out %s) ===" % args.heldout)
    for bb, r in results.items():
        spec = [v for k, v in r.items() if k.startswith("specialist:")]
        mrg = {k.split(":")[1]: v for k, v in r.items() if k.startswith("merged:")}
        best_spec = max(spec) if spec else float("nan")
        best_mrg = max(mrg.values()) if mrg else float("nan")
        print(f"  {bb:<10} pretrained={r.get('pretrained', float('nan')):.3f}  "
              f"best specialist={best_spec:.3f}  best merged={best_mrg:.3f}  "
              f"merged-minus-specialist={best_mrg - best_spec:+.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
