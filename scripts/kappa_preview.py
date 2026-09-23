"""How correlated are task vectors that share input images but differ in target?

Gram-LS reduces to a scaled sum when the task-vector Gram matrix is near-diagonal, so it can
only differ from task arithmetic when off-diagonal terms are large. This compares same-image
pairs (two CheXpert findings) against disjoint-domain pairs to see whether a
multi-finding suite would actually reach the regime the method needs.
"""
import glob, itertools, os, statistics as st
import torch

def load(p):
    o = torch.load(p, map_location="cpu", weights_only=False)
    return o.vector if hasattr(o, "vector") else o.get("vector", o)

for bb in ["dinov3", "dinov2", "beit"]:
    d = f"outputs/{bb}/seed_42_cap2k/task_vectors"
    names = [n for n in ["chexpert_pe", "chexpert_cm", "isic_mel", "ham10000_mel", "patchcamelyon"]
             if os.path.exists(f"{d}/{n}/task_vector.pt")]
    if len(names) < 2:
        continue
    tv = {n: load(f"{d}/{n}/task_vector.pt") for n in names}
    keys = [k for k in tv[names[0]] if tv[names[0]][k].ndim == 2]
    print(f"\n=== {bb} ===")
    for a, b in itertools.combinations(names, 2):
        cos = []
        for k in keys:
            x, y = tv[a][k].double().flatten(), tv[b][k].double().flatten()
            den = x.norm() * y.norm()
            if den > 0:
                cos.append(float((x @ y) / den))
        tag = "SAME IMAGES" if {a, b} == {"chexpert_pe", "chexpert_cm"} else \
              ("same modality" if {a, b} == {"isic_mel", "ham10000_mel"} else "disjoint domains")
        print(f"  {a:<14} vs {b:<14} median|cos| = {st.median([abs(c) for c in cos]):.4f}   [{tag}]")
    # kappa of the full Gram over whatever we have
    conds = []
    for k in keys:
        vs = [tv[n][k].double().flatten() for n in names]
        G = torch.tensor([[float(x @ y) for y in vs] for x in vs], dtype=torch.float64)
        sv = torch.linalg.svdvals(G)
        if sv[-1] > 0:
            conds.append(float(sv[0] / sv[-1]))
    print(f"  kappa(G) over {len(names)} tasks {names}: median {st.median(conds):.2f}")
