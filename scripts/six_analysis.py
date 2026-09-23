"""Six-task suite: does correlation finally separate Gram-LS from the sum?"""
import glob, itertools, json, os, statistics as st, math
import torch

SIX = ["isic_mel", "ham10000_mel", "chexpert_pe", "chexpert_cm", "patchcamelyon", "pathmnist_bin"]
BB = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]

def load(p):
    o = torch.load(p, map_location="cpu", weights_only=False)
    return o.vector if hasattr(o, "vector") else o.get("vector", o)

print("=== kappa(G) and within-domain cosines over 6 tasks (seed 42) ===")
for bb in BB:
    d = f"outputs/{bb}/seed_42_cap2k/task_vectors"
    if not all(os.path.exists(f"{d}/{t}/task_vector.pt") for t in SIX):
        continue
    tv = {t: load(f"{d}/{t}/task_vector.pt") for t in SIX}
    keys = [k for k in tv[SIX[0]] if tv[SIX[0]][k].ndim == 2]
    conds, pairs = [], {}
    for k in keys:
        vs = [tv[t][k].double().flatten() for t in SIX]
        G = torch.tensor([[float(a @ b) for b in vs] for a in vs], dtype=torch.float64)
        sv = torch.linalg.svdvals(G)
        if sv[-1] > 0:
            conds.append(float(sv[0] / sv[-1]))
    for a, b in [("isic_mel","ham10000_mel"),("chexpert_pe","chexpert_cm"),("patchcamelyon","pathmnist_bin")]:
        cs = []
        for k in keys:
            x, y = tv[a][k].double().flatten(), tv[b][k].double().flatten()
            den = x.norm()*y.norm()
            if den > 0: cs.append(abs(float(x@y)/float(den)))
        pairs[f"{a.split('_')[0]}|{b.split('_')[0]}"] = st.median(cs)
    print(f"  {bb:<10} kappa_med={st.median(conds):7.2f}   " +
          "  ".join(f"{k}={v:.3f}" for k, v in pairs.items()))

print()
print("=== six-task merge results (mean AUROC over 6 tasks, seed 42) ===")
M = ["gram_ls","task_arithmetic","tsv_merge","iso_c","iso_cts","simple_avg"]
rows = {}
for bb in BB:
    r = {}
    for m in M:
        p = f"outputs/{bb}/seed_42_cap2k/results_binary6/{m}/results.json"
        if not os.path.exists(p): continue
        d = json.load(open(p))
        v = [d[t]["auroc"] for t in SIX if t in d]
        if len(v) == 6: r[m] = sum(v)/6
    if r: rows[bb] = r
hdr = f"  {'backbone':<10}" + "".join(f"{m[:12]:>14}" for m in M)
print(hdr)
for bb, r in rows.items():
    print(f"  {bb:<10}" + "".join(f"{r.get(m, float('nan')):>14.3f}" for m in M))
if len(rows) >= 3:
    print()
    for m in M:
        v = [r[m] for r in rows.values() if m in r]
        if v: print(f"  {m:<16} mean {st.mean(v):.4f}  (n={len(v)})")
    d = [r["gram_ls"] - r["task_arithmetic"] for r in rows.values()
         if "gram_ls" in r and "task_arithmetic" in r]
    if len(d) > 2:
        t = st.mean(d)/(st.stdev(d)/math.sqrt(len(d)))
        print(f"\n  gram_ls minus task_arithmetic: {st.mean(d):+.4f}  t={t:.2f}  n={len(d)}")
