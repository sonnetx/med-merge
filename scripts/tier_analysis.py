"""Does any tuning-free merger match the tuned leading cluster?"""
import glob, json, os, math, statistics as st

BB = ["clip","vit","dinov3","rad_dino","dinov2","mae","beit"]; SE = ["42","123","456"]
T = ["isic_mel","chexpert_pe","patchcamelyon"]
TIER = ["ta_sum_notune","gram_ls_notune","metagpt_norm","nan_invnorm"]
TUNED = ["lines","gram_ls","task_arithmetic","dare","simple_avg"]

def cell(bb,s,m,root):
    p=f"outputs/{bb}/seed_{s}_cap2k/{root}/{m}/results.json"
    if not os.path.exists(p): return None
    r=json.load(open(p)); v=[r[t]["auroc"] for t in T if t in r]
    return sum(v)/len(v) if len(v)==3 else None

def series(m,root):
    return {(bb,s): cell(bb,s,m,root) for bb in BB for s in SE if cell(bb,s,m,root) is not None}

print("=== tuning-free tier, mean AUROC over 21 cells ===")
tier={m:series(m,"results_binary_tier") for m in TIER}
for m in TIER:
    v=list(tier[m].values())
    print(f"  {m:<16} {st.mean(v):.4f}  sd {st.pstdev(v):.4f}  n={len(v)}")
print()
print("=== reference, tuned ===")
tuned={m:series(m,"results_binary") for m in TUNED}
for m in TUNED:
    v=list(tuned[m].values())
    print(f"  {m:<16} {st.mean(v):.4f}  n={len(v)}")

def paired(A,B,la,lb):
    ks=[k for k in A if k in B]
    d=[A[k]-B[k] for k in ks]
    t=st.mean(d)/(st.stdev(d)/math.sqrt(len(d)))
    print(f"  {la:<16} - {lb:<18} {st.mean(d):+.4f}  t={t:6.2f}  n={len(d)}")

print()
print("=== key paired comparisons ===")
paired(tier["gram_ls_notune"],tier["ta_sum_notune"],"gramls@1","sum@1")
paired(tier["ta_sum_notune"],tuned["task_arithmetic"],"sum@1","TA tuned")
paired(tier["gram_ls_notune"],tuned["task_arithmetic"],"gramls@1","TA tuned")
paired(tier["gram_ls_notune"],tuned["lines"],"gramls@1","LiNeS tuned")
paired(tier["metagpt_norm"],tuned["simple_avg"],"metagpt","simple_avg")
paired(tier["nan_invnorm"],tuned["simple_avg"],"nan","simple_avg")

print()
print("=== per-backbone gramls@1 minus sum@1 (where do coefficients act?) ===")
for bb in BB:
    d=[tier["gram_ls_notune"][(bb,s)]-tier["ta_sum_notune"][(bb,s)]
       for s in SE if (bb,s) in tier["gram_ls_notune"] and (bb,s) in tier["ta_sum_notune"]]
    if d: print(f"  {bb:<10} {st.mean(d):+.4f}")

print()
print("=== logged Gram coefficients (deviation from 1, seed 42) ===")
for bb in BB:
    p=f"outputs/{bb}/seed_42_cap2k/merged_binary_tier/coefficients.json"
    if os.path.exists(p):
        c=json.load(open(p))["global_gram_c"]
        print(f"  {bb:<10} " + "  ".join(f"{k.split('_')[0]}={v:.3f}" for k,v in c.items()))
