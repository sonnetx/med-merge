"""Multi-task learning joint-training baseline.

Trains a single shared encoder with one task-specific head per dataset.
For each training step, samples a batch from each dataset's loader and
sums the per-task losses. 
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from med_merge.config.schema import DatasetConfig, ModelConfig, TrainingConfig
from med_merge.models.backbones import BackboneEncoder
from med_merge.models.factory import create_encoder
from med_merge.models.heads import create_head
from med_merge.training.data_module import DataModule
from med_merge.training.losses import get_loss_function
from med_merge.training.optimizer import create_optimizer, create_scheduler
from med_merge.utils.io import save_json, save_state_dict

logger = logging.getLogger(__name__)


def _primary_metric(task_type: str) -> tuple[str, str]:
    mapping = {
        "multiclass": ("balanced_accuracy", "max"),
        "multilabel": ("macro_auroc", "max"),
        "binary": ("auroc", "max"),
        "ordinal": ("qwk", "max"),
    }
    return mapping.get(task_type, ("accuracy", "max"))


class MTLModel(nn.Module):
    """Shared encoder + per-dataset heads."""

    def __init__(self, encoder: BackboneEncoder, heads: dict[str, nn.Module]):
        super().__init__()
        self.encoder: nn.Module = encoder  # type: ignore[assignment]
        self.heads = nn.ModuleDict(heads)

    def forward(self, pixel_values, dataset_name: str):
        features = self.encoder(pixel_values)
        return self.heads[dataset_name](features)


class MTLTrainer:
    """Joint-train an encoder on multiple datasets simultaneously."""

    def __init__(
        self,
        dataset_configs: dict[str, DatasetConfig],
        training_config: TrainingConfig,
        model_config: ModelConfig,
        output_dir: str,
        device: str = "cuda",
        exp_logger=None,
    ):
        self.dataset_configs = dataset_configs
        self.training_config = training_config
        self.model_config = model_config
        self.output_dir = Path(output_dir) / "mtl"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.exp_logger = exp_logger

        # One encoder, one head per dataset
        encoder = create_encoder(model_config)
        heads = {
            name: create_head(model_config.head_type, encoder.hidden_size, cfg.num_classes)
            for name, cfg in dataset_configs.items()
        }
        self.model = MTLModel(encoder, heads).to(device)
        if model_config.freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        # One DataModule per dataset for proper transforms
        self.data_modules: dict[str, DataModule] = {
            name: DataModule(cfg, training_config, model_config)
            for name, cfg in dataset_configs.items()
        }

        # Loss functions per dataset (handle multiclass / multilabel / binary)
        self.loss_fns: dict[str, nn.Module] = {
            name: get_loss_function(cfg.task_type,
                                    self.data_modules[name].class_weights,
                                    device)
            for name, cfg in dataset_configs.items()
        }

        # Single optimizer + scheduler over all parameters
        optimizer = create_optimizer(self.model, training_config)
        # _round_robin_batches yields max_len * n_datasets steps per epoch
        # (it cycles smaller loaders until the largest one is exhausted).
        # Buffer +n_datasets keeps OneCycleLR's strict step bound safe against
        # the very last-batch off-by-one we hit before.
        max_len = max(len(dm.train_loader) for dm in self.data_modules.values())
        total_steps = max_len * len(self.data_modules) * training_config.epochs + len(self.data_modules)
        scheduler = create_scheduler(optimizer, training_config, total_steps)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler("cuda", enabled=training_config.mixed_precision)
        self.use_amp = training_config.mixed_precision

    def _round_robin_batches(self):
        """Yield (dataset_name, images, labels) by interleaving the loaders.

        Uses ``itertools.zip_longest`` to cycle smaller loaders until the
        largest one is exhausted (so the bigger datasets see one full pass
        per epoch and smaller ones are repeated).
        """
        names = list(self.data_modules.keys())
        iters = {n: iter(self.data_modules[n].train_loader) for n in names}

        # Use the longest loader as the epoch length
        max_len = max(len(self.data_modules[n].train_loader) for n in names)

        for step in range(max_len):
            for n in names:
                try:
                    batch = next(iters[n])
                except StopIteration:
                    iters[n] = iter(self.data_modules[n].train_loader)
                    batch = next(iters[n])
                yield n, batch

    def train(self) -> dict[str, float]:
        """Run the MTL training loop. Returns best aggregate val metrics."""
        from med_merge.evaluation.metrics import compute_metrics

        best_aggregate = -float("inf")
        best_metrics_per_ds: dict[str, dict] = {}

        logger.info(
            f"MTL training: {len(self.dataset_configs)} datasets, "
            f"{self.training_config.epochs} epochs, "
            f"backbone={self.model_config.backbone}"
        )

        for epoch in range(self.training_config.epochs):
            self.model.train()
            total_loss_by_ds: dict[str, float] = {n: 0.0 for n in self.dataset_configs}
            count_by_ds: dict[str, int] = {n: 0 for n in self.dataset_configs}

            pbar = tqdm(self._round_robin_batches(), desc=f"MTL epoch {epoch}", leave=False)
            for name, (images, labels) in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                with autocast("cuda", enabled=self.use_amp):
                    logits = self.model(images, name)
                    loss = self.loss_fns[name](logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.training_config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                total_loss_by_ds[name] += loss.item()
                count_by_ds[name] += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", ds=name[:4])

            # Validation
            val_metrics = self._validate(compute_metrics)
            aggregate = sum(self._primary(ds_name, m) for ds_name, m in val_metrics.items())
            aggregate /= max(len(val_metrics), 1)

            log_payload = {
                f"train_loss_{n}": total_loss_by_ds[n] / max(count_by_ds[n], 1)
                for n in self.dataset_configs
            }
            for ds_name, m in val_metrics.items():
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        log_payload[f"val_{ds_name}_{k}"] = v
            log_payload["val_aggregate"] = aggregate
            log_payload["epoch"] = float(epoch)
            log_payload["lr"] = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch}: aggregate={aggregate:.4f}  "
                + "  ".join(
                    f"{n}={self._primary(n, val_metrics[n]):.3f}"
                    for n in val_metrics
                )
            )
            if self.exp_logger:
                self.exp_logger.log_metrics(log_payload, step=epoch)

            if aggregate > best_aggregate:
                best_aggregate = aggregate
                best_metrics_per_ds = {
                    ds: {**m, "epoch": epoch} for ds, m in val_metrics.items()
                }
                self._save_artifacts()
                # Persist incrementally — survives a crash in a later epoch's
                # scheduler.step (we hit a OneCycleLR off-by-one before).
                save_json(
                    {"best_aggregate": best_aggregate, "per_dataset": best_metrics_per_ds},
                    self.output_dir / "best_metrics.json",
                )
        logger.info(f"MTL done. Best aggregate={best_aggregate:.4f}")
        return {"aggregate": best_aggregate, **{ds: best_metrics_per_ds[ds] for ds in best_metrics_per_ds}}

    @torch.no_grad()
    def _validate(self, metric_fn) -> dict[str, dict]:
        self.model.eval()
        out: dict[str, dict] = {}
        for name, cfg in self.dataset_configs.items():
            loader = self.data_modules[name].val_loader
            all_logits = []
            all_labels = []
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                with autocast("cuda", enabled=self.use_amp):
                    logits = self.model(images, name)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
            logits = torch.cat(all_logits).numpy()
            labels = torch.cat(all_labels).numpy()
            out[name] = metric_fn(
                logits, labels,
                task_type=cfg.task_type, class_names=cfg.class_names,
            )
        return out

    def _primary(self, ds_name: str, metrics: dict) -> float:
        primary, _ = _primary_metric(self.dataset_configs[ds_name].task_type)
        return float(metrics.get(primary, 0.0))

    def _save_artifacts(self) -> None:
        # Save encoder
        encoder_sd = {f"encoder.{k}": v.clone() for k, v in self.model.encoder.state_dict().items()}
        save_state_dict(encoder_sd, self.output_dir / "encoder.pt")
        # Save each head separately
        heads_dir = self.output_dir / "heads"
        for ds_name, head in self.model.heads.items():
            head_sd = {f"head.{k}": v.clone() for k, v in head.state_dict().items()}
            save_state_dict(head_sd, heads_dir / ds_name / "head.pt")
        # Save model config so eval knows which backbone
        save_json(self.model_config.model_dump(), self.output_dir / "model_config.json")
