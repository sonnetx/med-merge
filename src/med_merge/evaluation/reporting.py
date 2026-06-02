"""Report generation: tables, plots, and LaTeX output."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from med_merge.merging.task_vector import TaskVector

logger = logging.getLogger(__name__)


def generate_task_vector_similarity_table(
    task_vectors: dict[str, "TaskVector"],
    output_dir: Path,
) -> str:
    """Write a pairwise task-vector cosine similarity matrix as CSV.

    Reuses ``compute_pairwise_interference`` from ``analysis.conflict``;
    this function only formats and persists the output.

    Returns the CSV string.
    """
    from med_merge.analysis.conflict import compute_pairwise_interference

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise = compute_pairwise_interference(task_vectors)
    names = list(task_vectors.keys())

    rows = [[""] + names]
    for a in names:
        row = [a]
        for b in names:
            if a == b:
                row.append("1.0000")
                continue
            key = (a, b) if (a, b) in pairwise else (b, a)
            cos = pairwise[key]["cosine"]
            row.append(f"{cos:.4f}")
        rows.append(row)

    csv_path = output_dir / "task_vector_cosine.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    logger.info(f"Task vector cosine similarity saved to {csv_path}")
    return "\n".join(",".join(r) for r in rows)


def generate_main_table(
    all_results: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    primary_metrics: dict[str, str] | None = None,
) -> str:
    """Generate the main results table as CSV.

    Args:
        all_results: {method_name: {dataset_name: {metric: value}}}.
        output_dir: Where to save output files.
        primary_metrics: {dataset_name: metric_name} for the main column.

    Returns:
        CSV string.
    """
    if primary_metrics is None:
        primary_metrics = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = sorted(all_results.keys())
    datasets = set()
    for m in methods:
        datasets.update(all_results[m].keys())
    datasets = sorted(datasets)

    # CSV
    rows = []
    header = ["Method"] + [f"{ds}" for ds in datasets] + ["Mean"]
    rows.append(header)

    for method in methods:
        row = [method]
        scores = []
        for ds in datasets:
            metrics = all_results.get(method, {}).get(ds, {})
            pm = primary_metrics.get(ds, "balanced_accuracy")
            value = metrics.get(pm, metrics.get("accuracy", float("nan")))
            row.append(f"{value:.4f}" if not np.isnan(value) else "N/A")
            if not np.isnan(value):
                scores.append(value)
        mean = np.mean(scores) if scores else float("nan")
        row.append(f"{mean:.4f}" if not np.isnan(mean) else "N/A")
        rows.append(row)

    csv_path = output_dir / "main_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    logger.info(f"Results table saved to {csv_path}")
    return "\n".join(",".join(r) for r in rows)


def generate_latex_table(
    all_results: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    primary_metrics: dict[str, str] | None = None,
) -> str:
    """Generate LaTeX table for paper."""
    if primary_metrics is None:
        primary_metrics = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = sorted(all_results.keys())
    datasets = set()
    for m in methods:
        datasets.update(all_results[m].keys())
    datasets = sorted(datasets)

    n_cols = len(datasets) + 2  # method + datasets + mean
    col_spec = "l" + "c" * (n_cols - 1)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    header = "Method & " + " & ".join(ds.replace("_", "\\_") for ds in datasets) + " & Mean \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for method in methods:
        row_parts = [method.replace("_", "\\_")]
        scores = []
        for ds in datasets:
            metrics = all_results.get(method, {}).get(ds, {})
            pm = primary_metrics.get(ds, "balanced_accuracy")
            value = metrics.get(pm, metrics.get("accuracy", float("nan")))
            if not np.isnan(value):
                row_parts.append(f"{value:.3f}")
                scores.append(value)
            else:
                row_parts.append("--")
        mean = np.mean(scores) if scores else float("nan")
        row_parts.append(f"{mean:.3f}" if not np.isnan(mean) else "--")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Model merging benchmark results across medical imaging datasets.}",
        "\\label{tab:main_results}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)
    latex_path = output_dir / "main_results.tex"
    with open(latex_path, "w") as f:
        f.write(latex)

    logger.info(f"LaTeX table saved to {latex_path}")
    return latex


def plot_results(
    all_results: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    primary_metrics: dict[str, str] | None = None,
) -> None:
    """Generate result visualizations."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if primary_metrics is None:
        primary_metrics = {}

    methods = sorted(all_results.keys())
    datasets = set()
    for m in methods:
        datasets.update(all_results[m].keys())
    datasets = sorted(datasets)

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)

    for i, method in enumerate(methods):
        values = []
        for ds in datasets:
            metrics = all_results.get(method, {}).get(ds, {})
            pm = primary_metrics.get(ds, "balanced_accuracy")
            values.append(metrics.get(pm, 0))
        ax.bar(x + i * width, values, width, label=method)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Primary Metric")
    ax.set_title("Model Merging Benchmark Results")
    ax.set_xticks(x + width * len(methods) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Plots saved to {output_dir}")
