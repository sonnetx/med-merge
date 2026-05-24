# med-merge

A controlled benchmark for weight-space model merging on medical imaging vision transformers. Companion code to a CS229 final paper studying what is predictable about merge forgetting under leave-one-backbone-out cross-validation.

Pipeline: fine-tune backbone-per-dataset specialists, compute task vectors, merge, evaluate, analyze.


## Grid

7 backbones x 6 datasets x 8 merging methods x 3 seeds = 1008 cells.

### Backbones

| Alias | HF id | Hidden | Pretraining |
|---|---|---|---|
| `clip` | `openai/clip-vit-base-patch16` | 768 | image-text contrastive |
| `vit` | `google/vit-base-patch16-224` | 768 | ImageNet-21k supervised |
| `dinov3` | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | self-supervised |
| `rad_dino` | `microsoft/rad-dino` | 768 | chest X-ray self-supervised |
| `dinov2` | `facebook/dinov2-base` | 768 | self-supervised |
| `mae` | `facebook/vit-mae-base` | 768 | masked image modeling |
| `beit` | `microsoft/beit-base-patch16-224-pt22k-ft22k` | 768 | BERT-style image pretraining |

### Datasets

| Alias | Domain | Classes | Task type |
|---|---|---|---|
| `isic2017` | dermoscopy | 3 | multiclass |
| `chexpert` | chest radiology | 5 | multilabel |
| `tcga` | lung histopathology | 2 | binary |
| `nih_cxr` | chest radiology | 5 | multilabel |
| `pathmnist` | colon histopathology | 9 | multiclass |
| `retinamnist` | fundus / DR | 5 | multiclass |

### Merging methods

Simple Averaging, Task Arithmetic, TIES, DARE, DARE-TIES, PCB-Merging, LiNeS, Fisher-Weighted.

### Baselines

Specialist oracle router (upper bound) + multi-task joint training (MTL).
