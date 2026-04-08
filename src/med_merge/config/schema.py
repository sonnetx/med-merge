"""Typed configuration schema for med-merge experiments.

Uses Pydantic v2 BaseModel for validation with ``extra="forbid"`` so
typos in YAML keys are caught early.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Model / backbone
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Which vision encoder to use."""

    model_config = ConfigDict(extra="forbid")

    backbone: str = "openai/clip-vit-base-patch16"
    backend: Literal["huggingface", "timm"] = "huggingface"
    hidden_size: int = 768
    num_layers: int = 12
    head_type: Literal["linear", "mlp"] = "linear"
    freeze_encoder: bool = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source: str = ""  # HuggingFace ID, path, or URL
    data_dir: str = "./data"
    num_classes: int = 2
    task_type: Literal["multiclass", "multilabel", "binary", "ordinal"] = "multiclass"
    image_size: int = 224
    class_names: list[str] = []

    # Split configuration
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    split_seed: int = 42
    split_ratios: list[float] = [0.8, 0.1, 0.1]
    group_by: Optional[str] = None  # column for no-leakage splits (e.g. lesion_id)

    # Dataset-specific
    csv_path: Optional[str] = None  # path to labels CSV (CheXpert, TCGA, ISIC)
    label_columns: list[str] = []  # for multi-label
    uncertainty_strategy: Optional[str] = None  # "u_ones" for CheXpert
    view_filter: Optional[str] = None  # "frontal" for CheXpert
    class_weights: Optional[list[float]] = None

    # Augmentation hint (used by DataModule to pick transforms)
    augmentation: Literal["standard", "aggressive"] = "standard"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    scheduler: str = "cosine"
    optimizer: str = "adamw"
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    early_stopping_patience: int = 5
    early_stopping_metric: Optional[str] = None  # auto-determined if None
    save_best_only: bool = True
    finetune_strategy: Literal["full", "head_only"] = "full"
    use_balanced_sampler: bool = False
    use_class_weights: bool = False


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

class MergingConfig(BaseModel):
    """Configuration for a merging method."""

    model_config = ConfigDict(extra="forbid")

    method: str = "simple_avg"  # simple_avg, task_arithmetic, ties, dare, etc.
    datasets: list[str] = []
    checkpoint_dir: str = "./outputs/checkpoints"

    # Task Arithmetic / shared scaling coefficient
    alpha: Optional[float] = None
    alpha_search: list[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    # TIES
    trim_fraction: float = 0.2
    trim_fraction_search: list[float] = [0.05, 0.1, 0.2, 0.3]

    # DARE
    drop_rate: float = 0.5
    drop_rate_search: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
    dare_seed: int = 42

    # LiNeS
    lines_alpha: float = 0.1  # base scale for shallowest layer
    lines_beta: float = 0.9  # scale range (deepest = alpha + beta)
    lines_alpha_search: list[float] = [0.0, 0.05, 0.1, 0.2, 0.3]
    lines_beta_search: list[float] = [0.5, 0.7, 0.9, 1.0]

    # SLERP (two-model only)
    slerp_t: float = 0.5
    slerp_t_search: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Fisher
    fisher_n_samples: int = 1000
    fisher_alpha: Optional[float] = None  # falls back to `alpha` if None

    # Composite methods
    inner_method: Optional[str] = None  # for DARE, LiNeS

    # Hyperopt
    run_hyperopt: bool = True
    optimization_objective: Literal["aggregate", "ece", "combined"] = "aggregate"
    ece_weight: float = 0.1  # weight for ECE in combined objective


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = 64
    num_calibration_bins: int = 15
    bootstrap_n: int = 1000
    bootstrap_ci: float = 0.95
    run_fairness: bool = True
    fairness_dataset: str = "fitzpatrick17k"
    num_workers: int = 4


# ---------------------------------------------------------------------------
# Wandb
# ---------------------------------------------------------------------------

class WandbConfig(BaseModel):
    """Weights & Biases configuration."""

    model_config = ConfigDict(extra="forbid")

    project: str = "med-merge"
    entity: Optional[str] = None
    mode: str = "online"  # "online", "offline", "disabled"


# ---------------------------------------------------------------------------
# Top-level experiment
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    seed: int = 42
    output_dir: str = "./outputs"
    data_dir: str = "./data"
    device: str = "cuda"
    wandb: WandbConfig = WandbConfig()
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig(name="", source="")
    training: TrainingConfig = TrainingConfig()
    merging: MergingConfig = MergingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
