"""Canonical constants for the med-merge benchmark."""

from __future__ import annotations

# All trainable/mergeable dataset names (used for CLI validation).
ALL_DATASETS = [
    "isic2017", "chexpert", "pathmnist", "tcga", "nih_cxr", "retinamnist",
    "isic_mel", "chexpert_pe", "patchcamelyon", "pathmnist_bin",
    "ham10000_mel", "nct_crc_tum", "chexpert_cm",
]

# Study cores.
# MULTICLASS_CORE: three disjoint domains at native task type (derm 3-class,
#   CXR 5-label multilabel, histo 9-class). Real-world / mixed-type comparison.
MULTICLASS_CORE = ["isic2017", "chexpert", "pathmnist"]
# BINARY_CORE (primary controlled study): one task type across the three domains,
#   melanoma detection / pleural effusion / tumor detection. Same metric + head.
BINARY_CORE = ["isic_mel", "chexpert_pe", "patchcamelyon"]
# BINARY_SIX: the task-count extension. Two tasks per domain, so task count doubles while
# the domains stay fixed, and the within-domain pairs raise task-vector correlation.
BINARY_SIX = [
    "isic_mel", "ham10000_mel",        # dermoscopy
    "chexpert_pe", "chexpert_cm",      # chest radiography
    "patchcamelyon", "nct_crc_tum",    # histopathology
]

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
    "iso_c",
    "iso_cts",
    "tsv_merge",
    "gram_ls",
    "gram_ls_na",
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
    # Binary controlled trio (one task type across three disjoint domains).
    "isic_mel": "auroc",
    "chexpert_pe": "auroc",
    "patchcamelyon": "auroc",
    "pathmnist_bin": "auroc",
    "ham10000_mel": "auroc",
    "nct_crc_tum": "auroc",
    "chexpert_cm": "auroc",
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
