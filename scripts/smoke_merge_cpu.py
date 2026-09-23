import torch, traceback
from med_merge.pipelines import run_merge
OUT="./outputs/vit/seed_42"
DS=["isic2017","chexpert","pathmnist"]
for m in ["simple_avg","iso_c","tsv_merge","gram_ls"]:
    print(f"=== merge {m} (no hyperopt, cpu) ===", flush=True)
    try:
        run_merge(method=m, task_vector_dir=f"{OUT}/task_vectors",
                  output_dir=f"{OUT}/merged_core3_cpu", datasets=DS,
                  run_hyperopt=False, device="cpu")
        sd=torch.load(f"{OUT}/merged_core3_cpu/{m}/merged_encoder.pt", map_location="cpu", weights_only=True)
        ff=[v for v in sd.values() if v.dtype.is_floating_point]
        finite=all(torch.isfinite(v).all() for v in ff)
        tot=(sum(float(v.float().norm())**2 for v in ff))**0.5
        print(f"  OK {m}: keys={len(sd)} all_finite={finite} total_norm={tot:.2f}", flush=True)
    except Exception as e:
        traceback.print_exc(); print(f"  FAILED {m}: {e}", flush=True)
print("SMOKE DONE")
