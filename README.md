# med-merge

A benchmark for systematically studying weight-space model merging on medical imaging vision transformers.

Fine-tune backbone per dataset → Compute task vectors → Merge → Evaluate → Analyze

## Datasets (3 domains)

| Dataset | Domain | Classes | Task Type | Source |
|---------|--------|---------|-----------|--------|
| ISIC 2017 | Dermoscopy | 3 | Multiclass | Local (CSV + images) |
| CheXpert | Chest X-ray | 5 | Multi-label | Local (CSV + images) |
| TCGA (LUAD vs LUSC) | Lung histology | 2 | Binary | Local (CSV + thumbnails) |

## Backbones

| Model | Type | Hidden Dim |
|-------|------|------------|
| CLIP ViT-B/16 | ImageNet + text-supervised | 768 |
| ViT-B/16 | ImageNet-supervised | 768 |
| DINOv3 ViT-S/16 | Self-supervised (medical) | 384 |

## Merging Methods (9 total)

| Method | Description |
|--------|-------------|
| Simple Averaging | `pretrained + (1/N) * Σ(task_vectors)` |
| Task Arithmetic | `pretrained + α * Σ(task_vectors)` with tuned α |
| TIES | Trim low-magnitude → elect dominant sign → disjoint merge |
| DARE | Drop and rescale delta params → Task Arithmetic |
| DARE-TIES | DARE preprocessing → TIES merge |
| PCB-Merging | Parameter Competition Balancing (NeurIPS 2024) |
| LiNeS | Layer-increasing scaling: shallow layers down, deep layers up |
| SLERP | Spherical linear interpolation (two-model only) |
| Fisher | Diagonal Fisher Information-weighted merge |

## SLURM

```bash
sbatch slurm/smoke_test.sh          # Quick 1-epoch test
sbatch slurm/full_benchmark.sh      # Full benchmark
```