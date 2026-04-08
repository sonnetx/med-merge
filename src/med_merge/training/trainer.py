"""High-level training orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.amp import GradScaler

from med_merge.config.schema import DatasetConfig, ModelConfig, TrainingConfig
from med_merge.merging.task_vector import TaskVector
from med_merge.models.factory import create_model, load_pretrained_encoder
from med_merge.training.callbacks import CheckpointManager, EarlyStopping
from med_merge.training.data_module import DataModule
from med_merge.training.engine import TrainingEngine
from med_merge.training.losses import get_loss_function
from med_merge.training.optimizer import create_optimizer, create_scheduler
from med_merge.utils.io import save_json, save_state_dict
from med_merge.utils.logging import ExperimentLogger

logger = logging.getLogger(__name__)


def _get_primary_metric(task_type: str, explicit: Optional[str] = None) -> tuple[str, str]:
    if explicit:
        return explicit, "max"
    mapping = {
        "multiclass": ("balanced_accuracy", "max"),
        "multilabel": ("macro_auroc", "max"),
        "binary": ("auroc", "max"),
        "ordinal": ("qwk", "max"),
    }
    return mapping.get(task_type, ("accuracy", "max"))


class Trainer:
    """Compose DataModule, TrainingEngine, and callbacks to fine-tune a model."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        output_dir: str,
        model_config: Optional[ModelConfig] = None,
        device: str = "cuda",
        exp_logger: Optional[ExperimentLogger] = None,
    ):
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.model_config = model_config or ModelConfig()
        self.output_dir = Path(output_dir) / "checkpoints" / dataset_config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.exp_logger = exp_logger

        # Primary metric for early-stopping
        metric_name, mode = _get_primary_metric(
            dataset_config.task_type, training_config.early_stopping_metric,
        )
        self.primary_metric = metric_name
        self.metric_mode = mode

        # Components
        self.data_module = DataModule(dataset_config, training_config, self.model_config)
        self.model = create_model(dataset_config, self.model_config, training_config).to(device)
        self._pretrained_encoder = load_pretrained_encoder(self.model_config)

        # Loss
        self.loss_fn = get_loss_function(
            dataset_config.task_type, self.data_module.class_weights, device,
        )

        # Optimizer / scheduler
        optimizer = create_optimizer(self.model, training_config)
        total_steps = len(self.data_module.train_loader) * training_config.epochs
        scheduler = create_scheduler(optimizer, training_config, total_steps)
        scaler = GradScaler("cuda", enabled=training_config.mixed_precision)

        # Engine
        self.engine = TrainingEngine(
            self.model, optimizer, scheduler, self.loss_fn, scaler,
            device=device,
            max_grad_norm=training_config.max_grad_norm,
            use_amp=training_config.mixed_precision,
        )

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=training_config.early_stopping_patience, mode=self.metric_mode,
        )
        self.checkpoint_mgr = CheckpointManager(
            self.output_dir, save_best_only=training_config.save_best_only, mode=self.metric_mode,
        )

    def train(self) -> dict[str, float]:
        """Run full training loop. Returns best validation metrics."""
        from med_merge.evaluation.metrics import compute_metrics

        logger.info(
            f"Training {self.dataset_config.name}: "
            f"{self.training_config.epochs} epochs, "
            f"batch_size={self.training_config.batch_size}, "
            f"lr={self.training_config.learning_rate}"
        )

        best_metrics: dict[str, float] = {}
        for epoch in range(self.training_config.epochs):
            train_loss = self.engine.train_epoch(self.data_module.train_loader, epoch)
            val_metrics = self.engine.validate(
                self.data_module.val_loader,
                metric_fn=compute_metrics,
                task_type=self.dataset_config.task_type,
                class_names=self.dataset_config.class_names,
                epoch=epoch,
            )
            val_metrics["train_loss"] = train_loss
            val_metrics["epoch"] = float(epoch)
            val_metrics["lr"] = self.engine.optimizer.param_groups[0]["lr"]

            if self.exp_logger:
                self.exp_logger.log_metrics(val_metrics, step=epoch)

            score = val_metrics.get(self.primary_metric, val_metrics.get("val_loss", 0))
            saved = self.checkpoint_mgr.save_if_best(
                self.model, self.engine.optimizer, epoch, score, val_metrics,
            )
            if saved:
                best_metrics = val_metrics.copy()

            if self.early_stopping(score):
                break

        self._save_artifacts(best_metrics)
        return best_metrics

    def _save_artifacts(self, best_metrics: dict[str, float]) -> None:
        """Save task vector, head, and metrics after training."""
        best_path = self.output_dir / "best_model.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state_dict"])

        save_state_dict(self.model.get_head_state_dict(), self.output_dir / "head.pt")

        finetuned_encoder = self.model.get_encoder_state_dict()
        task_vector = TaskVector(
            pretrained_state_dict=self._pretrained_encoder,
            finetuned_state_dict=finetuned_encoder,
        )
        tv_dir = Path(str(self.output_dir).replace("checkpoints", "task_vectors"))
        task_vector.save(tv_dir / "task_vector.pt")

        save_json(best_metrics, self.output_dir / "best_metrics.json")
        save_json(self.model_config.model_dump(), self.output_dir / "model_config.json")
        logger.info(f"Saved artifacts to {self.output_dir}")
