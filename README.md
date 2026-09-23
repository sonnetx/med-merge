# med-merge

A controlled benchmark for weight-space model merging on medical imaging vision transformers.

The pipeline fine-tunes one specialist per dataset from a shared pretrained backbone, computes task vectors, merges them with each method under a shared validation grid, and evaluates the merged encoder with the task-specific heads against routing to the individual specialists.

## Primary benchmark

The primary study fixes task type, metric, head architecture, and training budget across three disjoint medical domains, so that method comparisons are not confounded by protocol differences between domains.

| Alias | Domain | Task | Source |
|---|---|---|---|
| `isic_mel` | dermoscopy | melanoma detection | ISIC 2017 |
| `chexpert_pe` | chest radiography | pleural effusion detection | CheXpert |
| `patchcamelyon` | histopathology | tumor detection | PatchCamelyon |

Every task is binary, scored by AUROC through a single-logit head, and every specialist trains on a seed-deterministic subsample of 2,000 images. The six-task extension adds a second binary task per domain (`ham10000_mel`, `chexpert_cm`, `pathmnist_bin`), built with the one-vs-rest wrapper in `src/med_merge/data/binarize.py`.

## Backbones

| Alias | HF id | Hidden | Pretraining |
|---|---|---|---|
| `clip` | `openai/clip-vit-base-patch16` | 768 | image-text contrastive |
| `vit` | `google/vit-base-patch16-224` | 768 | ImageNet-21k supervised |
| `dinov3` | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | self-supervised |
| `rad_dino` | `microsoft/rad-dino` | 768 | chest X-ray self-supervised |
| `dinov2` | `facebook/dinov2-base` | 768 | self-supervised |
| `mae` | `facebook/vit-mae-base` | 768 | masked image modeling |
| `beit` | `microsoft/beit-base-patch16-224-pt22k-ft22k` | 768 | BERT-style image pretraining |

## Merging methods

Simple Averaging, Task Arithmetic, TIES, DARE, DARE-TIES, PCB-Merging, LiNeS, Fisher-Weighted, Iso-C, Iso-CTS, TSV-Merge, and Gram-LS. Gram-LS solves a per-layer linear system over the Gram matrix of the task vectors so that the merged update preserves each task's own projection. It is data-free and closed-form, and it reduces to a scaled sum when the task vectors are orthogonal.

All methods with a tunable scaling coefficient share one search grid (`alpha_search` in `src/med_merge/config/schema.py`), widened until no method selects its boundary.

## Baselines

Routing to the correct specialist (the reference every merger is measured against) and multi-task joint training.

## Other datasets

The repository also carries loaders and configs for the earlier multiclass study (`isic2017`, `chexpert`, `tcga`, `nih_cxr`, `pathmnist`, `retinamnist`) and for the held-out transfer probes (`ham10000`, `retinamnist`).

## Paper experiments

Each analysis in the paper maps to a job under `slurm/` and a script under `scripts/`. The job scripts write to `outputs/<backbone>/seed_<seed>_cap2k/` and expect to be submitted from the repository root.

| Paper section | Job | Script |
|---|---|---|
| Main benchmark (Table 1) | `bin2k_cell.sh`, `bin2k_isocts.sh` | `aggregate_binary.py` |
| Specialist routing reference | `oracle_job.sh` | `compute_oracle_router.py` |
| Widened scaling grid and LiNeS sensitivity | `alpha_rerun.sh` | `aggregate_binary.py` |
| Validation-free tier (Table 3) | `tier_job.sh` | `tier_merge.py`, `tier_analysis.py` |
| Held-out linear probes | `probe_job.sh`, `probe_ext.sh` | `heldout_probe.py` |
| Six-task extension | `six_cell.sh`, `six_an_job.sh` | `six_analysis.py` |
| Synthetic isotropization figure | none | `gram_ls_synthetic.py` |
| Binary versus native ablation and conditioning | `abl_cell.sh`, `abl_final_job.sh`, `abl_corr_job.sh`, `gram_cond_job.sh` | `abl_final.py`, `abl_corr.py`, `gram_cond.py` |
| Task-vector norms (appendix) | none | `norm_binary.py`, `norm_analysis.py` |

The paper compares 11 methods. Fisher-weighted merging, SLERP, and the norm-aware Gram-LS variant are implemented here but excluded from the paper's tables. Fisher's binary-loss and Fisher-estimation implementation still needs correction and a rerun.

The scripts and jobs listed above were restored from the anonymized paper supplement, so a few data paths in them read `/path/to/...` and the cluster credential-sourcing lines were removed. The other scripts in `scripts/` belong to an earlier multiclass study and are kept for reference.

## Running

Install in a Python 3.10+ environment with `pip install -e ".[dev]"` and run `pytest tests`. The CLI exposes the pipeline stages as `med-merge download`, `train`, `train-mtl`, `merge`, `evaluate`, `report`, and `run-all`.

The `slurm/` scripts run the full grid on a SLURM cluster. They resolve the project directory from `PROJECT_DIR`, then `SLURM_SUBMIT_DIR`, then the current directory, so submit them from the repository root or export `PROJECT_DIR`. Dataset locations live in `configs/datasets/*.yaml` and point at the cluster filesystem used for the paper. Set `data_dir` and `csv_path` there for your own copies of the data. Jobs that load HuggingFace weights run with `HF_HUB_OFFLINE=1`, so cache the backbones once before submitting.

`RESULTS.md` is the running lab log kept during the project. The paper's numbers come from the packaged supplement rather than from that log.
