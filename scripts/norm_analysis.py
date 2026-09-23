"""Task-vector norms per (backbone, dataset) for the 3-core, seed 42, vs train size."""
from med_merge.merging.task_vector import TaskVector
from pathlib import Path
import statistics

PD = Path(".")
BBS = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
DS = ["isic2017", "chexpert", "pathmnist"]
TRAIN_N = {"isic2017": 2200, "chexpert": 51800, "pathmnist": 89996}

print(f"{'backbone':10}" + "".join(f"{d:>12}" for d in DS))
per_ds = {d: [] for d in DS}
for bb in BBS:
    row = f"{bb:10}"
    for d in DS:
        p = PD / "outputs" / bb / "seed_42" / "task_vectors" / d / "task_vector.pt"
        if p.exists():
            n = TaskVector.load(p).norm()
            per_ds[d].append(n)
            row += f"{n:12.2f}"
        else:
            row += f"{'MISS':>12}"
    print(row)
print("-" * 46)
print(f"{'MEAN':10}" + "".join(f"{statistics.mean(per_ds[d]):12.2f}" if per_ds[d] else f"{'-':>12}" for d in DS))
print(f"{'train N':10}" + "".join(f"{TRAIN_N[d]:12d}" for d in DS))
