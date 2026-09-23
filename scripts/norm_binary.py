from med_merge.merging.task_vector import TaskVector
from pathlib import Path
import statistics
PD = Path(".")
BBS = ["clip", "vit", "dinov3", "rad_dino", "dinov2", "mae", "beit"]
DS = ["isic_mel", "chexpert_pe", "patchcamelyon"]
print(f"{'backbone':10}" + "".join(f"{d:>15}" for d in DS))
per = {d: [] for d in DS}
for bb in BBS:
    row = f"{bb:10}"
    for d in DS:
        p = PD / "outputs" / bb / "seed_42_cap2k" / "task_vectors" / d / "task_vector.pt"
        if p.exists():
            n = TaskVector.load(p).norm(); per[d].append(n); row += f"{n:15.2f}"
        else:
            row += f"{'--':>15}"
    print(row)
print("-" * 55)
print(f"{'MEAN':10}" + "".join(f"{statistics.mean(per[d]):15.2f}" if per[d] else f"{'--':>15}" for d in DS))
