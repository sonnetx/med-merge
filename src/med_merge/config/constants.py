"""Canonical constants for the med-merge benchmark."""

from __future__ import annotations

ALL_DATASETS = ["isic2017", "chexpert", "tcga", "nih_cxr", "pathmnist", "retinamnist"]

ALL_METHODS = [
    "simple_avg",
    "task_arithmetic",
    "ties",
    "dare",
    "dare_ties",
    "pcb_merging",
    "lines",
    "slerp",
    "fisher",
]

SEEDS = [42, 123, 456]

# Primary evaluation metric per dataset (used by hyperopt, reporting, etc.)
PRIMARY_METRICS: dict[str, str] = {
    "isic2017": "balanced_accuracy",
    "chexpert": "macro_auroc",
    "tcga": "auroc",
    "nih_cxr": "macro_auroc",
    "pathmnist": "balanced_accuracy",
    "retinamnist": "balanced_accuracy",
}

# Dataset metadata: num_classes, task_type, class_names
DATASET_DEFAULTS: dict[str, dict] = {
    "isic2017": {
        "num_classes": 3,
        "task_type": "multiclass",
        "class_names": ["nevus", "melanoma", "seborrheic_keratosis"],
    },
    "chexpert": {
        "num_classes": 5,
        "task_type": "multilabel",
        "class_names": [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ],
    },
    "tcga": {
        "num_classes": 1,
        "task_type": "binary",
        "class_names": ["LUAD", "LUSC"],
    },
    "nih_cxr": {
        "num_classes": 5,
        "task_type": "multilabel",
        "class_names": [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ],
    },
    "pathmnist": {
        "num_classes": 9,
        "task_type": "multiclass",
        "class_names": [
            "ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM",
        ],
    },
    "retinamnist": {
        "num_classes": 5,
        "task_type": "multiclass",
        "class_names": ["0", "1", "2", "3", "4"],
    },
}
