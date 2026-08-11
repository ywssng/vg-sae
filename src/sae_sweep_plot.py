"""Plot-only helpers for saved custom-baseline sweep artifacts."""

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


def load_sweep_plot_context(sweep_dir: Path | str) -> dict[str, float | int]:
    """Load width-aware plotting constants, including legacy sweep artifacts."""

    root = Path(sweep_dir)
    data = json.loads((root / "sweep_config.json").read_text())["data"]
    with np.load(root / "summary" / "data_preview.npz") as preview:
        probabilities = preview["feature_probabilities"]
    legacy_width = data.get("n_features", len(probabilities))
    ground_truth_num_features = int(
        data.get("ground_truth_num_features", legacy_width)
    )
    sae_width = int(data.get("sae_width", legacy_width))
    return {
        "ground_truth_num_features": ground_truth_num_features,
        "sae_width": sae_width,
        "support_density": float(data["support_density"]),
        "expected_true_l0": float(probabilities.sum()),
        "target_model_density": float(probabilities.sum() / sae_width),
    }


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


def _metric_line(ax, table: pd.DataFrame, method: str, metric: str, sae_width: int) -> None:
    subset = table[table["method"] == method].sort_values("rho_model")
    if subset.empty:
        return
    values = subset[metric].to_numpy(float)
    if metric in {"average_l0", "expected_l0"}:
        values = values / sae_width
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


def _finish_metric_axis(
    ax, target_model_density: float, sae_width: int, ylabel: str
) -> None:
    ax.axvline(
        target_model_density, color="black", linestyle="--", linewidth=1, alpha=0.6
    )
    ax.set(xlabel=r"$\rho_\text{model}$", ylabel=ylabel)
    ax.set_xscale("symlog", linthresh=1 / sae_width)
    ax.grid(alpha=0.25)


def _plot_axes(
    target_model_density: float | None,
    sae_width: int | None,
    support_density: float | None,
    n_features: int | None,
) -> tuple[float, int]:
    """Resolve new names while retaining the old equal-width plotting API."""

    target = target_model_density if target_model_density is not None else support_density
    width = sae_width if sae_width is not None else n_features
    if target is None or width is None:
        raise TypeError("target_model_density and sae_width are required.")
    return float(target), int(width)


def plot_data_overview(
    sweep_dir: Path | str, output_path: Path | str | None = None
):
    with np.load(Path(sweep_dir) / "summary" / "data_preview.npz") as preview:
        probabilities = preview["feature_probabilities"]
        dictionary = preview["dictionary"]
        z0 = preview["z0"]
    gram = dictionary.T @ dictionary
    np.fill_diagonal(gram, 0.0)
    max_pairwise_cosine = float(np.abs(gram).max(initial=0.0))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(probabilities, marker="o", markersize=3)
    axes[0].set_title("Feature probabilities")
    axes[1].imshow(dictionary, aspect="auto", cmap="RdBu_r")
    axes[1].set_title(f"Random unit dictionary\nmax |pairwise cosine|={max_pairwise_cosine:.2f}")
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
    target_model_density: float | None = None,
    sae_width: int | None = None,
    output_path: Path | str | None = None,
    support_density: float | None = None,
    n_features: int | None = None,
):
    target_model_density, sae_width = _plot_axes(
        target_model_density, sae_width, support_density, n_features
    )
    table = aggregate_seed_metrics(metrics)
    panels = [("explained_variance", r"$R^2$"), ("reconstruction_error", "Recon. error")]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), sharex=False)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, sae_width)
        _finish_metric_axis(ax, target_model_density, sae_width, label)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_recovery_metrics(
    metrics: pd.DataFrame,
    *,
    target_model_density: float | None = None,
    sae_width: int | None = None,
    output_path: Path | str | None = None,
    support_density: float | None = None,
    n_features: int | None = None,
):
    target_model_density, sae_width = _plot_axes(
        target_model_density, sae_width, support_density, n_features
    )
    table = aggregate_seed_metrics(metrics)
    panels = [("generalization_error", "Gen. error"), ("decoder_recovery_cosine", "Dict. Cos sim.")]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), sharex=False)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, sae_width)
        _finish_metric_axis(ax, target_model_density, sae_width, label)
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
    target_model_density: float | None = None,
    sae_width: int | None = None,
    output_path: Path | str | None = None,
    support_density: float | None = None,
    n_features: int | None = None,
):
    target_model_density, sae_width = _plot_axes(
        target_model_density, sae_width, support_density, n_features
    )
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
            _metric_line(ax, table, method, metric, sae_width)
        _finish_metric_axis(ax, target_model_density, sae_width, label)
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
    target_model_density: float | None = None,
    sae_width: int | None = None,
    output_path: Path | str | None = None,
    support_density: float | None = None,
    n_features: int | None = None,
):
    target_model_density, sae_width = _plot_axes(
        target_model_density, sae_width, support_density, n_features
    )
    table = aggregate_seed_metrics(metrics)
    panels = [
        ("selection_error", "Selection error"),
        ("dead_fraction", "dead latent fraction"),
        ("average_l0", r"Avg. L0 / $d_\mathrm{sae}$"),
        ("expected_l0", r"Exp. L0 / $d_\mathrm{sae}$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.75), sharex=True)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, sae_width)
        _finish_metric_axis(ax, target_model_density, sae_width, label)
    density = np.arange(1, sae_width + 1) / sae_width
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


def _mask_representatives(
    metrics: pd.DataFrame,
    target_model_density: float,
    representative_seed: int | None,
) -> pd.DataFrame:
    methods = [method for method in METHOD_ORDER if (metrics["method"] == method).any()]
    if not methods:
        raise ValueError("Mask heatmaps require at least one supported method.")
    common_seeds = set.intersection(
        *(
            {int(seed) for seed in metrics.loc[metrics["method"] == method, "seed"]}
            for method in methods
        )
    )
    if not common_seeds:
        raise ValueError("Mask heatmaps require one seed shared by every method.")
    seed = min(common_seeds) if representative_seed is None else representative_seed
    if seed not in common_seeds:
        raise ValueError(f"Seed {seed} is not available for every plotted method.")
    rows = []
    for method in methods:
        subset = metrics[
            (metrics["method"] == method) & (metrics["seed"].astype(int) == seed)
        ]
        rows.append(
            subset.loc[(subset["rho_model"] - target_model_density).abs().idxmin()]
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def plot_mask_heatmaps(
    sweep_dir: Path | str,
    metrics: pd.DataFrame,
    *,
    target_model_density: float | None = None,
    checkpoint_kind: str = "last",
    n_show: int = 80,
    output_path: Path | str | None = None,
    support_density: float | None = None,
    representative_seed: int | None = None,
) -> tuple[Any, pd.DataFrame]:
    if target_model_density is None:
        if support_density is None:
            raise TypeError("target_model_density is required.")
        target_model_density = support_density
    representatives = _mask_representatives(
        metrics, target_model_density, representative_seed
    )
    fig, axes = plt.subplots(
        len(representatives), 2, figsize=(10, 2.2 * len(representatives)), sharex=True
    )
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
        ground_truth_width = int(row.get("ground_truth_num_features", support.shape[1]))
        if support.shape[1] > ground_truth_width:
            for ax in axes[row_index]:
                ax.axvline(ground_truth_width - 0.5, color="white", linewidth=1)
    for ax in axes.ravel():
        ax.set_xlabel("GT features + unmatched SAE latents")
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
    context = load_sweep_plot_context(root)
    outputs = Path(output_dir) if output_dir is not None else None
    destination = lambda name: outputs / name if outputs is not None else None
    figures = {
        "data_overview": plot_data_overview(root, destination("data_overview.png")),
        "reconstruction": plot_reconstruction_metrics(
            metrics,
            target_model_density=context["target_model_density"],
            sae_width=context["sae_width"],
            output_path=destination("reconstruction_metrics.png")
        ),
        "recovery": plot_recovery_metrics(
            metrics,
            target_model_density=context["target_model_density"],
            sae_width=context["sae_width"],
            output_path=destination("recovery_metrics.png")
        ),
        "support": plot_support_metrics(
            metrics,
            target_model_density=context["target_model_density"],
            sae_width=context["sae_width"],
            output_path=destination("support_metrics.png")
        ),
        "sparsity": plot_sparsity_diagnostics(
            metrics,
            target_model_density=context["target_model_density"],
            sae_width=context["sae_width"],
            output_path=destination("sparsity_diagnostics.png")
        ),
        "training": plot_training_curves(history, destination("training_curves.png")),
    }
    heatmap, representatives = plot_mask_heatmaps(
        root,
        metrics,
        target_model_density=context["target_model_density"],
        checkpoint_kind=checkpoint_kind,
        output_path=destination("mask_heatmaps.png"),
    )
    figures["masks"] = heatmap
    return figures, representatives
