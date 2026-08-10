"""Plot-only helpers for experiment-07 sweep artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from .sae_sweep import METHOD_LABELS, METHOD_ORDER


METHOD_COLORS = {
    "vgsae": "tab:blue",
    "l1": "tab:orange",
    "topk": "tab:green",
    "batchtopk": "tab:purple",
    "jumprelu": "tab:brown",
    "gated": "tab:red",
}
GROUP_COLUMNS = ["method", "method_label", "control_name", "control_value"]


def load_sweep_results(
    sweep_dir: Path | str, checkpoint_kind: str = "last"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(sweep_dir)
    metrics = pd.read_csv(root / "summary" / checkpoint_kind / "final_metrics.csv")
    history = pd.read_csv(root / "summary" / "training_curves.csv")
    return metrics, history


def aggregate_seed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.select_dtypes(include=np.number).columns
        if column not in {"seed", "control_value"}
    ]
    means = (
        metrics.groupby(GROUP_COLUMNS, as_index=False)
        .agg({**{column: "mean" for column in numeric}, "seed": "nunique"})
        .rename(columns={"seed": "n_seeds"})
    )
    errors = (
        metrics.groupby(GROUP_COLUMNS, as_index=False)[numeric]
        .sem()
        .fillna(0.0)
        .rename(columns={column: f"{column}_se" for column in numeric})
    )
    return means.merge(errors, on=GROUP_COLUMNS).sort_values(["method", "rho_model"])


def _save(fig, output_path: Path | str | None) -> None:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)


def _metric_line(ax, table: pd.DataFrame, method: str, metric: str, n_features: int) -> None:
    subset = table[table["method"] == method].sort_values("rho_model")
    if subset.empty:
        return
    values = subset[metric].to_numpy(float)
    if metric in {"average_l0", "expected_l0"}:
        values = values / n_features
    seeds = sorted(subset["n_seeds"].dropna().astype(int).unique())
    label = METHOD_LABELS[method]
    if len(seeds) == 1:
        label += f" (n={seeds[0]})"
    ax.plot(
        subset["rho_model"],
        values,
        marker="o",
        linestyle="-",
        linewidth=1,
        color=METHOD_COLORS[method],
        label=label,
    )


def _finish_metric_axis(ax, support_density: float, n_features: int, ylabel: str) -> None:
    ax.axvline(support_density, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set(xlabel=r"$\rho_\text{model}$", ylabel=ylabel)
    ax.set_xscale("symlog", linthresh=1 / n_features)
    ax.grid(alpha=0.25)


def plot_data_overview(
    sweep_dir: Path | str, output_path: Path | str | None = None
):
    with np.load(Path(sweep_dir) / "summary" / "data_preview.npz") as preview:
        probabilities = preview["feature_probabilities"]
        dictionary = preview["dictionary"]
        z0 = preview["z0"]
        coherence = float(preview["coherence"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(probabilities, marker="o", markersize=3)
    axes[0].set_title("Feature probabilities")
    axes[1].imshow(dictionary, aspect="auto", cmap="RdBu_r")
    axes[1].set_title(f"Dictionary (coherence={coherence:g})")
    axes[2].stem(z0)
    axes[2].set_title("Example sparse code")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_reconstruction_metrics(
    metrics: pd.DataFrame,
    *,
    support_density: float,
    n_features: int,
    output_path: Path | str | None = None,
):
    table = aggregate_seed_metrics(metrics)
    panels = [("explained_variance", r"$R^2$"), ("reconstruction_error", "Recon. error")]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), sharex=False)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, n_features)
        _finish_metric_axis(ax, support_density, n_features, label)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_recovery_metrics(
    metrics: pd.DataFrame,
    *,
    support_density: float,
    n_features: int,
    output_path: Path | str | None = None,
):
    table = aggregate_seed_metrics(metrics)
    panels = [("generalization_error", "Gen. error"), ("decoder_recovery_cosine", "Dict. Cos sim.")]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), sharex=False)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, n_features)
        _finish_metric_axis(ax, support_density, n_features, label)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    axes[0].set_yscale("log")
    axes[1].set_ylim(top=1.01)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_support_metrics(
    metrics: pd.DataFrame,
    *,
    support_density: float,
    n_features: int,
    output_path: Path | str | None = None,
):
    table = aggregate_seed_metrics(metrics)
    panels = [
        ("support_f1", "F1"),
        ("support_average_precision", "AP"),
        ("support_precision", "Precision"),
        ("support_recall", "Recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.5), sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, n_features)
        _finish_metric_axis(ax, support_density, n_features, label)
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[0]:
        ax.set_ylim(-0.01, 0.61)
        ax.set_xlabel("")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_sparsity_diagnostics(
    metrics: pd.DataFrame,
    *,
    support_density: float,
    n_features: int,
    output_path: Path | str | None = None,
):
    table = aggregate_seed_metrics(metrics)
    panels = [
        ("selection_error", "Selection error"),
        ("dead_fraction", "dead latent fraction"),
        ("average_l0", "Avg. L0"),
        ("expected_l0", "Exp. L0"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.75), sharex=True)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, n_features)
        _finish_metric_axis(ax, support_density, n_features, label)
    density = np.arange(1, n_features + 1) / n_features
    for ax in axes[2:]:
        ax.plot(density, density, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_training_curves(history: pd.DataFrame, output_path: Path | str | None = None):
    panels = [("loss", "training loss"), ("reconstruction_mse", "train reconstruction MSE"), ("rho", "train rho")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            subset = history[history["method"] == method]
            if subset.empty or metric not in subset:
                continue
            for _, run in subset.groupby("run_id"):
                run = run.sort_values("step")
                ax.plot(run["step"], run[metric], color=METHOD_COLORS[method], alpha=0.35, linewidth=1)
        ax.set(xlabel="step", ylabel=label)
        ax.grid(alpha=0.25)
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    axes[0].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_mask_heatmaps(
    sweep_dir: Path | str,
    metrics: pd.DataFrame,
    *,
    support_density: float,
    checkpoint_kind: str = "last",
    n_show: int = 80,
    output_path: Path | str | None = None,
) -> tuple[Any, pd.DataFrame]:
    rows = []
    for method in METHOD_ORDER:
        subset = metrics[metrics["method"] == method]
        if not subset.empty:
            rows.append(subset.loc[(subset["rho_model"] - support_density).abs().idxmin()])
    representatives = pd.DataFrame(rows).reset_index(drop=True)
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 2.2 * len(rows)), sharex=True)
    axes = np.atleast_2d(axes)
    root = Path(sweep_dir)
    for row_index, row in representatives.iterrows():
        cache_path = root / "runs" / row["method"] / row["run_id"] / "eval" / checkpoint_kind / "cache.npz"
        with np.load(cache_path) as cache:
            support, mask = cache["true_support"][:n_show], cache["mask"][:n_show]
        axes[row_index, 0].imshow(support, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        axes[row_index, 0].set(ylabel=row["method_label"], title="true support")
        axes[row_index, 1].imshow(mask, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        axes[row_index, 1].set_title(f"mask, rho={row['rho_model']:.3f}, sel err={row['selection_error']:.3f}")
    for ax in axes.ravel():
        ax.set_xlabel("matched feature")
    fig.tight_layout()
    _save(fig, output_path)
    return fig, representatives


def plot_all(
    sweep_dir: Path | str,
    *,
    checkpoint_kind: str = "last",
    output_dir: Path | str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reproduce every visual from notebook 07 with collision-free names."""

    root = Path(sweep_dir)
    metrics, history = load_sweep_results(root, checkpoint_kind)
    config = json.loads((root / "sweep_config.json").read_text())
    data = config["data"]
    outputs = Path(output_dir) if output_dir is not None else None
    destination = lambda name: outputs / name if outputs is not None else None
    figures = {
        "data_overview": plot_data_overview(root, destination("data_overview.png")),
        "reconstruction": plot_reconstruction_metrics(
            metrics, support_density=data["support_density"], n_features=data["n_features"],
            output_path=destination("reconstruction_metrics.png")
        ),
        "recovery": plot_recovery_metrics(
            metrics, support_density=data["support_density"], n_features=data["n_features"],
            output_path=destination("recovery_metrics.png")
        ),
        "support": plot_support_metrics(
            metrics, support_density=data["support_density"], n_features=data["n_features"],
            output_path=destination("support_metrics.png")
        ),
        "sparsity": plot_sparsity_diagnostics(
            metrics, support_density=data["support_density"], n_features=data["n_features"],
            output_path=destination("sparsity_diagnostics.png")
        ),
        "training": plot_training_curves(history, destination("training_curves.png")),
    }
    heatmap, representatives = plot_mask_heatmaps(
        root,
        metrics,
        support_density=data["support_density"],
        checkpoint_kind=checkpoint_kind,
        output_path=destination("mask_heatmaps.png"),
    )
    figures["masks"] = heatmap
    return figures, representatives
