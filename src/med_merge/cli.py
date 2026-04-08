"""CLI entry point for med-merge benchmark."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from med_merge.config.constants import ALL_DATASETS, ALL_METHODS, DATASET_DEFAULTS

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """med-merge: Medical Imaging Model Merging Benchmark."""
    from med_merge.utils.logging import setup_logging

    setup_logging("DEBUG" if verbose else "INFO")


@cli.command()
@click.option("--datasets", "-d", multiple=True, default=["all"])
@click.option("--data-dir", type=click.Path(), default="./data")
def download(datasets: tuple[str, ...], data_dir: str) -> None:
    """Download and prepare datasets."""
    from med_merge.data.download import download_datasets

    download_datasets(list(datasets), data_dir)
    click.echo("Download complete.")


@cli.command()
@click.option("--dataset", "-d", required=True, type=click.Choice(ALL_DATASETS))
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--output-dir", "-o", type=click.Path(), default="./outputs")
@click.option("--data-dir", type=click.Path(), default="./data")
@click.option("--epochs", type=int, default=None, help="Override number of epochs")
@click.option("--batch-size", type=int, default=None, help="Override batch size")
@click.option("--lr", type=float, default=None, help="Override learning rate")
@click.option("--seed", type=int, default=42)
@click.option("--device", type=str, default="cuda")
@click.option("--wandb-mode", type=click.Choice(["online", "offline", "disabled"]), default="disabled")
def train(
    dataset: str,
    config: str | None,
    output_dir: str,
    data_dir: str,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    seed: int,
    device: str,
    wandb_mode: str,
) -> None:
    """Fine-tune a model on a medical imaging dataset."""
    from med_merge.config.loader import load_config, load_yaml, merge_configs
    from med_merge.config.schema import DatasetConfig, TrainingConfig
    from med_merge.pipelines import run_training
    from med_merge.utils.logging import ExperimentLogger, setup_wandb
    from med_merge.utils.reproducibility import seed_everything

    seed_everything(seed)

    # Load config
    configs_dir = Path(__file__).parent.parent.parent / "configs"
    ds_config_path = configs_dir / "datasets" / f"{dataset}.yaml"
    tr_config_path = configs_dir / "training" / f"{dataset}.yaml"

    if config:
        experiment = load_config(config)
    elif ds_config_path.exists() and tr_config_path.exists():
        ds_cfg = load_yaml(ds_config_path)
        tr_cfg = load_yaml(tr_config_path)
        merged = merge_configs(ds_cfg, tr_cfg)
        experiment = load_config()
        if "dataset" in merged:
            experiment.dataset = DatasetConfig.model_validate(merged["dataset"])
        if "training" in merged:
            experiment.training = TrainingConfig.model_validate(merged["training"])
    elif ds_config_path.exists():
        experiment = load_config(ds_config_path)
    else:
        experiment = load_config()

    # Apply dataset defaults — only override data_dir if no YAML path was loaded
    defaults = DATASET_DEFAULTS.get(dataset, {})
    if not experiment.dataset.name:
        experiment.dataset = DatasetConfig(name=dataset, data_dir=str(Path(data_dir) / dataset), **defaults)
    experiment.dataset.name = dataset
    # Only override data_dir if the YAML didn't set a real path (still the default "./data")
    if experiment.dataset.data_dir == "./data":
        experiment.dataset.data_dir = str(Path(data_dir) / dataset)
    if not experiment.dataset.class_names and "class_names" in defaults:
        experiment.dataset.class_names = defaults["class_names"]
    if experiment.dataset.num_classes == 2 and defaults.get("num_classes", 2) != 2:
        experiment.dataset.num_classes = defaults["num_classes"]
    if "task_type" in defaults:
        experiment.dataset.task_type = defaults["task_type"]
    experiment.seed = seed
    experiment.device = device

    # CLI overrides
    if epochs is not None:
        experiment.training.epochs = epochs
    if batch_size is not None:
        experiment.training.batch_size = batch_size
    if lr is not None:
        experiment.training.learning_rate = lr

    logger.info(
        f"Training {dataset}: {experiment.training.epochs} epochs, "
        f"batch_size={experiment.training.batch_size}, "
        f"lr={experiment.training.learning_rate}"
    )

    # Setup wandb
    wandb_run = setup_wandb(
        project="med-merge",
        mode=wandb_mode,
        config=experiment.model_dump(),
        name=f"train-{dataset}",
    )

    metrics = run_training(experiment, output_dir=output_dir)
    click.echo(f"Training complete. Best metrics: {metrics}")


@cli.command()
@click.option("--method", "-m", required=True, type=click.Choice(ALL_METHODS))
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--checkpoint-dir", type=click.Path(), default="./outputs/checkpoints")
@click.option("--task-vector-dir", type=click.Path(), default="./outputs/task_vectors")
@click.option("--output-dir", "-o", type=click.Path(), default="./outputs/merged")
@click.option("--datasets", "-d", multiple=True, default=ALL_DATASETS)
@click.option("--hyperopt/--no-hyperopt", default=False)
@click.option("--device", type=str, default="cuda")
def merge(
    method: str,
    config: str | None,
    checkpoint_dir: str,
    task_vector_dir: str,
    output_dir: str,
    datasets: tuple[str, ...],
    hyperopt: bool,
    device: str,
) -> None:
    """Merge task-specific models using specified method."""
    from med_merge.pipelines import run_merge

    result = run_merge(
        method=method,
        task_vector_dir=task_vector_dir,
        output_dir=output_dir,
        datasets=list(datasets),
        device=device,
    )
    click.echo(f"Merge complete: {result}")


@cli.command()
@click.option("--model-path", "-p", required=True, type=click.Path(exists=True))
@click.option("--datasets", "-d", multiple=True, default=ALL_DATASETS)
@click.option("--head-dir", type=click.Path(), default="./outputs/checkpoints")
@click.option("--data-dir", type=click.Path(), default="./data")
@click.option("--output-dir", "-o", type=click.Path(), default="./outputs/results")
@click.option("--device", type=str, default="cuda")
def evaluate(
    model_path: str,
    datasets: tuple[str, ...],
    head_dir: str,
    data_dir: str,
    output_dir: str,
    device: str,
) -> None:
    """Evaluate a merged model on benchmark datasets."""
    from med_merge.pipelines import run_evaluation

    all_metrics = run_evaluation(
        model_path=model_path,
        datasets=list(datasets),
        head_dir=head_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )
    click.echo(f"Results: {all_metrics}")


@cli.command()
@click.option("--results-dir", type=click.Path(exists=True), default="./outputs/results")
@click.option("--output-dir", "-o", type=click.Path(), default="./outputs/reports")
@click.option("--format", "fmt", type=click.Choice(["csv", "latex", "both"]), default="both")
def report(results_dir: str, output_dir: str, fmt: str) -> None:
    """Generate summary tables and visualizations."""
    import json

    from med_merge.config.constants import PRIMARY_METRICS
    from med_merge.evaluation.reporting import (
        generate_latex_table,
        generate_main_table,
        plot_results,
    )

    results_path = Path(results_dir)
    output_path = Path(output_dir)

    all_results = {}
    for method_dir in results_path.iterdir():
        if method_dir.is_dir():
            results_file = method_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    all_results[method_dir.name] = json.load(f)

    if not all_results:
        click.echo("No results found.")
        return

    if fmt in ("csv", "both"):
        generate_main_table(all_results, output_path, PRIMARY_METRICS)
    if fmt in ("latex", "both"):
        generate_latex_table(all_results, output_path, PRIMARY_METRICS)
    plot_results(all_results, output_path, PRIMARY_METRICS)
    click.echo(f"Reports saved to {output_path}")


@cli.command("run-all")
@click.option("--output-dir", "-o", type=click.Path(), default="./outputs")
@click.option("--data-dir", type=click.Path(), default="./data")
@click.option("--device", type=str, default="cuda")
@click.option("--wandb-mode", type=click.Choice(["online", "offline", "disabled"]), default="disabled")
def run_all(output_dir: str, data_dir: str, device: str, wandb_mode: str) -> None:
    """Run the complete benchmark: train all -> merge all -> evaluate all -> report."""
    from med_merge.pipelines import run_full_benchmark

    run_full_benchmark(
        output_dir=output_dir,
        data_dir=data_dir,
        device=device,
        wandb_mode=wandb_mode,
    )
    click.echo("Benchmark complete!")


if __name__ == "__main__":
    cli()
