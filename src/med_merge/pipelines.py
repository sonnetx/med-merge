"""Core pipeline functions — importable, testable, no subprocess.

Each function implements a stage of the benchmark pipeline.
The CLI wraps these with argument parsing and logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from med_merge.config.constants import ALL_DATASETS, ALL_METHODS, DATASET_DEFAULTS
from med_merge.config.schema import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    MergingConfig,
    ModelConfig,
)

logger = logging.getLogger(__name__)


def _load_dataset_config(ds_name: str, fallback_data_dir: str) -> DatasetConfig:
    """Load dataset config from YAML if available, else build from defaults."""
    from med_merge.config.loader import load_yaml

    configs_dir = Path(__file__).parent.parent.parent / "configs"
    yaml_path = configs_dir / "datasets" / f"{ds_name}.yaml"
    if yaml_path.exists():
        raw = load_yaml(yaml_path)
        return DatasetConfig.model_validate(raw["dataset"])

    defaults = DATASET_DEFAULTS.get(ds_name, {})
    return DatasetConfig(
        name=ds_name,
        data_dir=str(Path(fallback_data_dir) / ds_name),
        **defaults,
    )


def run_training(
    experiment: ExperimentConfig,
    output_dir: str = "./outputs",
    exp_logger=None,
) -> dict[str, float]:
    """Fine-tune a model on a single dataset. Returns best metrics."""
    from med_merge.training.trainer import Trainer
    from med_merge.utils.reproducibility import seed_everything

    seed_everything(experiment.seed)
    trainer = Trainer(
        dataset_config=experiment.dataset,
        training_config=experiment.training,
        model_config=experiment.model,
        output_dir=output_dir,
        device=experiment.device,
        exp_logger=exp_logger,
    )
    return trainer.train()


def run_merge(
    method: str,
    task_vector_dir: str,
    output_dir: str,
    datasets: list[str] | None = None,
    model_config: ModelConfig | None = None,
    merge_config: MergingConfig | None = None,
    run_hyperopt: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    """Merge task vectors using specified method. Returns merge metadata."""
    import torch

    from med_merge.merging.registry import build_merger
    from med_merge.merging.task_vector import TaskVector
    from med_merge.models.factory import load_pretrained_encoder
    from med_merge.utils.io import load_state_dict, save_json, save_state_dict

    import json

    datasets = datasets or ALL_DATASETS

    # Auto-detect model config from checkpoint dir if not provided
    if model_config is None:
        for ds in datasets:
            cfg_path = Path(task_vector_dir).parent / "checkpoints" / ds / "model_config.json"
            if cfg_path.exists():
                model_config = ModelConfig.model_validate(
                    json.loads(cfg_path.read_text())
                )
                logger.info(f"Loaded model config from {cfg_path}: backbone={model_config.backbone}")
                break

    model_config = model_config or ModelConfig()
    pretrained = load_pretrained_encoder(model_config)

    task_vectors = {}
    for ds in datasets:
        tv_path = Path(task_vector_dir) / ds / "task_vector.pt"
        if tv_path.exists():
            task_vectors[ds] = TaskVector.load(tv_path)
            logger.info(f"  Loaded task vector: {ds} (norm={task_vectors[ds].norm():.2f})")
        else:
            logger.warning(f"  Task vector not found for {ds} at {tv_path}")

    if not task_vectors:
        raise FileNotFoundError("No task vectors found. Run training first.")

    try:
        from med_merge.evaluation.reporting import generate_task_vector_similarity_table
        generate_task_vector_similarity_table(
            task_vectors, Path(output_dir).parent / "analysis"
        )
    except Exception as e:
        logger.warning(f"Could not write task-vector cosine table: {e}")

    config = merge_config or MergingConfig(method=method, datasets=list(task_vectors.keys()))
    if config.method != method:
        config.method = method
    config.datasets = list(task_vectors.keys())

    merger = build_merger(pretrained, config)

    # Hyperopt: sweep hyperparameters using validation data
    if run_hyperopt and method not in ("simple_avg", "pcb_merging"):
        logger.info(f"Running hyperopt for {method}...")
        from med_merge.data.registry import build_dataset
        from med_merge.data.transforms import get_eval_transform, norm_key_for_backbone
        from torch.utils.data import DataLoader

        nk = norm_key_for_backbone(model_config.backbone)
        val_loaders = {}
        heads = {}
        dataset_configs_map = {}
        for ds_name in task_vectors:
            ds_config = _load_dataset_config(ds_name, "./data")
            transform = get_eval_transform(ds_config.image_size, norm_key=nk)
            extra_kw = {}
            if ds_config.csv_path:
                extra_kw["csv_path"] = ds_config.csv_path
            val_ds = build_dataset(ds_name, ds_config.data_dir, split="validation",
                                   transform=transform, **extra_kw)
            val_loaders[ds_name] = DataLoader(val_ds, batch_size=64, shuffle=False,
                                              num_workers=4, pin_memory=True)
            head_path = Path(task_vector_dir).parent / "checkpoints" / ds_name / "head.pt"
            if head_path.exists():
                heads[ds_name] = load_state_dict(head_path)
            dataset_configs_map[ds_name] = ds_config

        if val_loaders and heads:
            from med_merge.merging.hyperopt import MergingHyperoptimizer
            # Checkpoint file: write to method-specific dir so it's resumable
            state_file = Path(output_dir) / method / "hyperopt_state.json"
            # Fisher cache: per-backbone, shared across methods + hyperopt trials
            fisher_cache_dir = Path(task_vector_dir).parent / "fisher_cache"
            hyperopt = MergingHyperoptimizer(type(merger), pretrained, config,
                                             model_config=model_config,
                                             state_file=state_file,
                                             fisher_cache_dir=fisher_cache_dir)
            result = hyperopt.search(task_vectors, val_loaders, heads,
                                     dataset_configs_map, device=device)
            # Re-merge with best params (Fisher also needs val data + cache)
            best_config = hyperopt._apply_params(result["best_params"])
            merger = build_merger(pretrained, best_config)
            extra_kwargs = {}
            if method == "fisher":
                extra_kwargs = {
                    "val_loaders": val_loaders,
                    "heads": heads,
                    "dataset_configs": dataset_configs_map,
                    "model_config": model_config,
                    "fisher_cache_dir": str(fisher_cache_dir),
                }
            merged_state_dict = merger.merge(task_vectors, **result["best_params"], **extra_kwargs)
            logger.info(f"Hyperopt best: {result['best_params']} -> {result['best_score']:.4f}")
        else:
            logger.warning("Hyperopt skipped: missing val loaders or heads")
            merged_state_dict = merger.merge(task_vectors)
    else:
        merged_state_dict = merger.merge(task_vectors)

    out_path = Path(output_dir) / method
    out_path.mkdir(parents=True, exist_ok=True)
    save_state_dict(merged_state_dict, out_path / "merged_encoder.pt")
    save_json(
        {"method": method, "datasets": list(task_vectors.keys()),
         "hyperparameters": merger.hyperparameters},
        out_path / "merge_config.json",
    )
    logger.info(f"Merged encoder saved to {out_path / 'merged_encoder.pt'}")
    return {"method": method, "datasets": list(task_vectors.keys())}


def run_evaluation(
    model_path: str,
    datasets: list[str],
    head_dir: str,
    data_dir: str,
    output_dir: str,
    model_config: ModelConfig | None = None,
    eval_config: EvaluationConfig | None = None,
    device: str = "cuda",
) -> dict[str, dict]:
    """Evaluate a merged model on benchmark datasets. Returns metrics per dataset."""
    from med_merge.evaluation.evaluator import Evaluator
    from med_merge.utils.io import load_state_dict, save_json

    import json

    encoder = load_state_dict(Path(model_path))

    # Auto-detect model config from checkpoint dir if not provided
    if model_config is None:
        for ds_name in datasets:
            cfg_path = Path(head_dir) / ds_name / "model_config.json"
            if cfg_path.exists():
                model_config = ModelConfig.model_validate(
                    json.loads(cfg_path.read_text())
                )
                logger.info(f"Loaded model config from {cfg_path}: backbone={model_config.backbone}")
                break

    evaluator = Evaluator(eval_config or EvaluationConfig(), model_config, device)

    all_metrics = {}
    for ds_name in datasets:
        head_path = Path(head_dir) / ds_name / "head.pt"
        if not head_path.exists():
            logger.warning(f"Skipping {ds_name}: head not found at {head_path}")
            continue

        head = load_state_dict(head_path)

        # Try loading dataset config from YAML (has real Sherlock paths),
        # fall back to defaults with data_dir-relative path
        ds_config = _load_dataset_config(ds_name, data_dir)

        result = evaluator.evaluate_single(encoder, ds_config, head)
        all_metrics[ds_name] = result["metrics"]
        logger.info(f"  {ds_name}: {result['metrics']}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_json(all_metrics, out_path / "results.json")
    return all_metrics


def run_full_benchmark(
    output_dir: str = "./outputs",
    data_dir: str = "./data",
    device: str = "cuda",
    wandb_mode: str = "disabled",
) -> None:
    """Orchestrate train-all -> merge-all -> eval-all -> report."""
    from med_merge.config.loader import load_config
    from med_merge.evaluation.forgetting import compute_forgetting
    from med_merge.utils.io import save_json

    import json

    oracle_dir = Path(output_dir) / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train all datasets
    oracle_results: dict[str, dict[str, float]] = {}
    for ds in ALL_DATASETS:
        logger.info(f"{'=' * 60}\nTraining on {ds}\n{'=' * 60}")
        ds_config = _load_dataset_config(ds, data_dir)
        experiment = ExperimentConfig(dataset=ds_config, device=device)
        metrics = run_training(experiment, output_dir=output_dir)
        oracle_results[ds] = dict(metrics or {})
        save_json(oracle_results[ds], oracle_dir / f"{ds}.json")

    # 2. Merge with all methods
    for method in ALL_METHODS:
        logger.info(f"{'=' * 60}\nMerging with {method}\n{'=' * 60}")
        try:
            run_merge(
                method=method,
                task_vector_dir=f"{output_dir}/task_vectors",
                output_dir=f"{output_dir}/merged",
                device=device,
            )
        except Exception as e:
            logger.error(f"Merging with {method} failed: {e}")

    # 3. Evaluate all merged models
    for method in ALL_METHODS:
        merged_path = Path(output_dir) / "merged" / method / "merged_encoder.pt"
        if merged_path.exists():
            logger.info(f"{'=' * 60}\nEvaluating {method}\n{'=' * 60}")
            run_evaluation(
                model_path=str(merged_path),
                datasets=ALL_DATASETS,
                head_dir=f"{output_dir}/checkpoints",
                data_dir=data_dir,
                output_dir=f"{output_dir}/results/{method}",
                device=device,
            )

    if not oracle_results:
        for ds in ALL_DATASETS:
            p = oracle_dir / f"{ds}.json"
            if p.exists():
                oracle_results[ds] = json.loads(p.read_text())

    forgetting_dir = Path(output_dir) / "forgetting"
    forgetting_dir.mkdir(parents=True, exist_ok=True)
    for method in ALL_METHODS:
        results_path = Path(output_dir) / "results" / method / "results.json"
        if not results_path.exists():
            continue
        merged_results = json.loads(results_path.read_text())
        forgetting = compute_forgetting(merged_results, oracle_results)
        save_json(forgetting, forgetting_dir / f"{method}.json")
        if "mean_forgetting" in forgetting:
            logger.info(f"  {method} mean forgetting: {forgetting['mean_forgetting']:.4f}")

    logger.info("Benchmark complete!")
