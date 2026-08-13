"""Plot-only helpers for saved CustomData and SynthSAEBench sweep artifacts."""

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
GROUP_COLUMNS = [
    "method",
    "method_label",
    "beta_mode",
    "control_name",
    "control_value",
]
VALID_BETA_MODES = frozenset({"profiled", "learned"})
VALID_DENSITY_MODES = frozenset({"reported", "hard"})
VALID_METRIC_SUITES = frozenset({"auto", "stage1", "stage2"})
CUSTOM_HARD_METRIC_MAP = {
    "explained_variance": "hard_explained_variance",
    "reconstruction_error": "hard_reconstruction_error",
    "generalization_error": "hard_generalization_error",
    "selection_error": "hard_selection_error",
    "support_f1": "hard_support_f1",
    "support_average_precision": "hard_support_average_precision",
    "support_precision": "hard_support_precision",
    "support_recall": "hard_support_recall",
    "support_roc_auc": "hard_support_roc_auc",
}


def _resolve_metric_suite(metrics: pd.DataFrame, metric_suite: str) -> str:
    """Choose Stage-1 diagnostics or the official Stage-2 metric panels."""

    if metric_suite not in VALID_METRIC_SUITES:
        choices = ", ".join(sorted(VALID_METRIC_SUITES))
        raise ValueError(
            f"metric_suite must be one of {{{choices}}}, got {metric_suite!r}"
        )
    if metric_suite == "auto":
        return "stage2" if "benchmark_model_id" in metrics.columns else "stage1"
    return metric_suite


def _annotate_stage1_style_on_stage2(
    fig, *, metric_suite: str, is_synthsaebench: bool
) -> None:
    if metric_suite == "stage1" and is_synthsaebench:
        fig.suptitle("Stage-1-style diagnostics on Stage-2 matching", fontsize=9)


def apply_density_axis(
    metrics: pd.DataFrame,
    *,
    sae_width: int,
    density_mode: str = "reported",
) -> pd.DataFrame:
    """Return metrics with ``rho_model`` set to the requested plotting density.

    Stage-1 VG reports the posterior occupancy ``mean(m)`` while its baselines
    report binary activation densities.  ``hard`` makes the comparison
    homogeneous by using thresholded average L0 for every method.  The original
    reported value is retained in ``rho_model_reported``.
    """

    if density_mode not in VALID_DENSITY_MODES:
        choices = ", ".join(sorted(VALID_DENSITY_MODES))
        raise ValueError(
            f"density_mode must be one of {{{choices}}}, got {density_mode!r}"
        )
    if not isinstance(sae_width, (int, np.integer)) or sae_width <= 0:
        raise ValueError("sae_width must be a positive integer.")
    if "rho_model" not in metrics and "rho_model_reported" not in metrics:
        raise ValueError("Metrics do not contain a reported rho_model column.")

    transformed = metrics.copy()
    if "rho_model_reported" not in transformed:
        transformed["rho_model_reported"] = transformed["rho_model"]
    if density_mode == "reported":
        transformed["rho_model"] = transformed["rho_model_reported"]
        transformed["density_axis"] = "reported"
        return transformed

    hard_l0: pd.Series | None = None
    if "average_l0" in transformed:
        hard_l0 = pd.to_numeric(transformed["average_l0"], errors="coerce")
    elif "sae_l0" in transformed:
        hard_l0 = pd.to_numeric(transformed["sae_l0"], errors="coerce")
    if hard_l0 is None:
        raise ValueError(
            "Hard-density plots require average_l0 (or SynthSAEBench sae_l0)."
        )
    values = hard_l0.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Hard L0 values must all be finite.")
    if ((values < 0.0) | (values > float(sae_width))).any():
        raise ValueError("Hard L0 values must lie between zero and sae_width.")
    transformed["rho_model"] = hard_l0 / float(sae_width)
    transformed["density_axis"] = "hard"
    return transformed


def _validate_beta_mode(beta_mode: str, *, source: str) -> str:
    if beta_mode not in VALID_BETA_MODES:
        choices = ", ".join(sorted(VALID_BETA_MODES))
        raise ValueError(
            f"{source} beta_mode must be one of {{{choices}}}, got {beta_mode!r}"
        )
    return beta_mode


def _frame_beta_modes(frame: pd.DataFrame, *, source: str) -> set[str]:
    if "beta_mode" not in frame:
        return set()
    observed = {str(value) for value in frame["beta_mode"].dropna()}
    invalid = sorted(observed - VALID_BETA_MODES)
    if invalid:
        raise ValueError(f"{source} contains invalid beta_mode values: {invalid}")
    if len(observed) > 1:
        raise ValueError(f"{source} mixes beta_mode values: {sorted(observed)}")
    return observed


def _resolve_root_beta_mode(
    root: Path,
    metrics: pd.DataFrame,
    history: pd.DataFrame,
    *,
    source: str,
    explicit: str | None = None,
) -> str:
    candidates: dict[str, str] = {}
    if explicit is not None:
        candidates["explicit"] = _validate_beta_mode(explicit, source="explicit")
    config_path = root / "sweep_config.json"
    if config_path.exists():
        config_mode = json.loads(config_path.read_text()).get("training", {}).get(
            "beta_mode"
        )
        if config_mode is not None:
            candidates["sweep config"] = _validate_beta_mode(
                str(config_mode), source=f"{source} sweep config"
            )
    metric_modes = _frame_beta_modes(metrics, source=f"{source} metrics")
    history_modes = _frame_beta_modes(history, source=f"{source} history")
    if metric_modes:
        candidates["metrics"] = next(iter(metric_modes))
    if history_modes:
        candidates["history"] = next(iter(history_modes))
    observed = set(candidates.values())
    if len(observed) > 1:
        details = ", ".join(f"{key}={value}" for key, value in candidates.items())
        raise ValueError(f"{source} beta_mode metadata conflicts: {details}")
    return next(iter(observed)) if observed else "profiled"


def load_sweep_results(
    sweep_dir: Path | str, checkpoint_kind: str = "last"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(sweep_dir)
    metrics = pd.read_csv(root / "summary" / checkpoint_kind / "final_metrics.csv")
    history = pd.read_csv(root / "summary" / "training_curves.csv")
    return metrics, history


def load_comparison_results(
    sweep_dir: Path | str,
    checkpoint_kind: str = "last",
    baseline_sweep_dir: Path | str | None = None,
    beta_mode: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], Path]]:
    """Load one sweep, optionally filling absent methods from a baseline root."""

    root = Path(sweep_dir)
    metrics, history = load_sweep_results(root, checkpoint_kind)
    mode = _resolve_root_beta_mode(
        root,
        metrics,
        history,
        source="primary",
        explicit=beta_mode,
    )
    if "beta_mode" not in metrics:
        metrics["beta_mode"] = mode
    else:
        metrics["beta_mode"] = metrics["beta_mode"].fillna(mode)
    if "beta_mode" not in history:
        history["beta_mode"] = mode
    else:
        history["beta_mode"] = history["beta_mode"].fillna(mode)
    run_roots = {
        (str(row.method), str(row.run_id)): root
        for row in metrics[["method", "run_id"]].itertuples(index=False)
    }
    if baseline_sweep_dir is None:
        return metrics, history, run_roots

    baseline_root = Path(baseline_sweep_dir)
    baseline_metrics, baseline_history = load_sweep_results(
        baseline_root, checkpoint_kind
    )
    baseline_mode = _resolve_root_beta_mode(
        baseline_root,
        baseline_metrics,
        baseline_history,
        source="baseline",
    )
    primary_data = json.loads((root / "sweep_config.json").read_text())["data"]
    baseline_data = json.loads(
        (baseline_root / "sweep_config.json").read_text()
    )["data"]
    identity_fields = sorted(set(primary_data) | set(baseline_data))
    mismatched = [
        field
        for field in identity_fields
        if primary_data.get(field) != baseline_data.get(field)
    ]
    if mismatched:
        raise ValueError(
            "Comparison roots use different data/evaluation conditions: "
            + ", ".join(mismatched)
        )
    present_methods = set(metrics["method"].astype(str))
    baseline_metrics = baseline_metrics[
        ~baseline_metrics["method"].astype(str).isin(present_methods)
    ].copy()
    baseline_history = baseline_history[
        ~baseline_history["method"].astype(str).isin(present_methods)
    ].copy()
    retained_vg = bool(
        (baseline_metrics["method"].astype(str) == "vgsae").any()
        or (baseline_history["method"].astype(str) == "vgsae").any()
    )
    if retained_vg and baseline_mode != mode:
        raise ValueError(
            "Cannot backfill VG-SAE from a baseline root with a different beta_mode: "
            f"primary={mode}, baseline={baseline_mode}."
        )
    for frame in (baseline_metrics, baseline_history):
        if "beta_mode" not in frame:
            frame["beta_mode"] = mode
        else:
            frame["beta_mode"] = frame["beta_mode"].fillna(mode)
        # Baseline methods are beta-invariant, but beta_mode remains a valid
        # experiment-axis value so every stored/result table uses the same
        # profiled-or-learned vocabulary as the train/eval interfaces.
        frame.loc[frame["method"] != "vgsae", "beta_mode"] = mode
    run_roots.update(
        {
            (str(row.method), str(row.run_id)): baseline_root
            for row in baseline_metrics[["method", "run_id"]].itertuples(
                index=False
            )
        }
    )
    return (
        pd.concat([metrics, baseline_metrics], ignore_index=True, sort=False),
        pd.concat([history, baseline_history], ignore_index=True, sort=False),
        run_roots,
    )


def load_sweep_plot_context(
    sweep_dir: Path | str,
) -> dict[str, float | int | str]:
    """Load width-aware plotting constants, including legacy sweep artifacts."""

    root = Path(sweep_dir)
    data = json.loads((root / "sweep_config.json").read_text())["data"]
    data_kind = str(data.get("kind", "synthetic_sparse_coding"))
    with np.load(root / "summary" / "data_preview.npz") as preview:
        probabilities = preview["feature_probabilities"]
        probability_expected_l0 = float(probabilities.sum())
        empirical_true_l0 = float(
            preview["empirical_true_l0"]
            if "empirical_true_l0" in preview
            else probability_expected_l0
        )
        width = int(data.get("sae_width", len(probabilities)))
        legacy_target = float(
            preview["target_model_density"]
            if "target_model_density" in preview
            else probability_expected_l0 / width
        )
        target_model_density_expected = float(
            preview["target_model_density_expected"]
            if "target_model_density_expected" in preview
            else legacy_target
        )
        target_model_density_empirical = float(
            preview["target_model_density_empirical"]
            if "target_model_density_empirical" in preview
            else (
                empirical_true_l0 / width
                if "empirical_true_l0" in preview
                else target_model_density_expected
            )
        )
        # SynthSAEBench's official finite evaluation stream defines its target
        # empirically; retain the legacy context value for existing consumers.
        expected_true_l0 = (
            empirical_true_l0
            if data_kind == "synthsaebench_pretrained"
            else probability_expected_l0
        )
    legacy_width = data.get("n_features", len(probabilities))
    ground_truth_num_features = int(
        data.get("ground_truth_num_features", legacy_width)
    )
    sae_width = int(data.get("sae_width", legacy_width))
    return {
        "ground_truth_num_features": ground_truth_num_features,
        "sae_width": sae_width,
        "support_density": float(
            data.get("support_density", expected_true_l0 / ground_truth_num_features)
        ),
        "amplitude_mode": str(data.get("amplitude_mode", "exponential")),
        "amplitude_scale": float(data.get("amplitude_scale", 1.0)),
        "frequency_skew": float(data.get("frequency_skew", 0.0)),
        "expected_true_l0": expected_true_l0,
        "empirical_true_l0": empirical_true_l0,
        "target_model_density": target_model_density_expected,
        "target_model_density_expected": target_model_density_expected,
        "target_model_density_empirical": target_model_density_empirical,
        "data_kind": data_kind,
    }


def aggregate_seed_metrics(
    metrics: pd.DataFrame, beta_mode: str | None = None
) -> pd.DataFrame:
    metric_modes = _frame_beta_modes(metrics, source="metrics")
    mode = (
        _validate_beta_mode(beta_mode, source="explicit")
        if beta_mode is not None
        else (next(iter(metric_modes)) if metric_modes else "profiled")
    )
    if metric_modes and next(iter(metric_modes)) != mode:
        raise ValueError(
            "beta_mode metadata conflicts: "
            f"explicit={mode}, metrics={next(iter(metric_modes))}"
        )
    if "beta_mode" not in metrics:
        metrics = metrics.assign(beta_mode=mode)
    else:
        metrics = metrics.assign(
            beta_mode=metrics["beta_mode"].fillna(mode)
        )
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
        fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.08)


def _format_step_tick(value: float, _position: int) -> str:
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def _metric_line(ax, table: pd.DataFrame, method: str, metric: str, sae_width: int) -> None:
    subset = table[table["method"] == method].sort_values("rho_model")
    if subset.empty:
        return
    values = subset[metric].to_numpy(float)
    if metric in {"average_l0", "expected_l0", "sae_l0", "true_l0"}:
        values = values / sae_width
    label = METHOD_LABELS[method]
    ax.plot(
        subset["rho_model"],
        values,
        marker="o",
        linestyle="-",
        linewidth=1,
        color=METHOD_COLORS[method],
        label=label,
    )


def _metric_for_density(
    metrics: pd.DataFrame,
    metric: str,
    *,
    density_mode: str,
    is_synthsaebench: bool,
) -> str:
    """Resolve plot y-values without mixing a hard x-axis with native codes."""

    if density_mode != "hard" or is_synthsaebench:
        return metric
    resolved = CUSTOM_HARD_METRIC_MAP.get(metric, metric)
    if resolved != metric and resolved not in metrics.columns:
        raise ValueError(
            f"Hard-density plot requires {resolved!r}; rerun Stage-1 evaluation "
            "with the current metric schema."
        )
    return resolved


def _finish_metric_axis(
    ax,
    target_model_density: float,
    sae_width: int,
    ylabel: str,
    density_mode: str = "reported",
) -> None:
    ax.axvline(
        target_model_density, color="black", linestyle="--", linewidth=1, alpha=0.6
    )
    xlabel = (
        r"Hard activation density (L0 / $d_\mathrm{sae}$)"
        if density_mode == "hard"
        else r"$\rho_\text{model}$"
    )
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_xscale("log")
    # The benchmark's useful density band is less than one decade wide.  The
    # default log formatter labels every minor tick there, which makes the
    # notebook-10 panels unreadable once six methods are overlaid.
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(alpha=0.25)


def _add_bottom_method_legend(fig, frame: pd.DataFrame) -> None:
    fig.set_layout_engine("constrained")
    methods = [
        method for method in METHOD_ORDER if (frame["method"] == method).any()
    ]
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker="o",
            linewidth=1,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    if not handles:
        return
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=len(handles),
        fontsize=8,
        frameon=True,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=0.8,
    )


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
        data_kind = (
            str(preview["data_kind"].item())
            if "data_kind" in preview
            else "synthetic_sparse_coding"
        )
    gram = dictionary.T @ dictionary
    np.fill_diagonal(gram, 0.0)
    max_pairwise_cosine = float(np.abs(gram).max(initial=0.0))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(probabilities, marker="o", markersize=3)
    axes[0].set_title(
        "Base feature probabilities" if data_kind == "synthsaebench_pretrained"
        else "Feature probabilities"
    )
    axes[1].imshow(dictionary, aspect="auto", cmap="RdBu_r")
    dictionary_name = (
        "Pretrained dictionary preview"
        if data_kind == "synthsaebench_pretrained"
        else "Random unit dictionary"
    )
    axes[1].set_title(
        f"{dictionary_name}\npreview max |pairwise cosine|={max_pairwise_cosine:.2f}"
    )
    axes[2].stem(z0)
    axes[2].set_title("Example sparse-code preview")
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
    density_mode: str = "reported",
    metric_suite: str = "auto",
):
    target_model_density, sae_width = _plot_axes(
        target_model_density, sae_width, support_density, n_features
    )
    metrics = apply_density_axis(
        metrics, sae_width=sae_width, density_mode=density_mode
    )
    table = aggregate_seed_metrics(metrics)
    is_synthsaebench = "benchmark_model_id" in metrics.columns
    resolved_suite = _resolve_metric_suite(metrics, metric_suite)
    panels = (
        [("explained_variance", r"$R^2$"), ("shrinkage", "Shrinkage")]
        if resolved_suite == "stage2"
        else [
            ("explained_variance", r"$R^2$"),
            ("reconstruction_error", "Recon. error"),
        ]
    )
    panels = [
        (
            _metric_for_density(
                metrics,
                metric,
                density_mode=density_mode,
                is_synthsaebench=is_synthsaebench,
            ),
            (
                {
                    "explained_variance": r"Hard-code $R^2$",
                    "reconstruction_error": "Hard-code recon. error",
                }[metric]
                if density_mode == "hard" and not is_synthsaebench
                else label
            ),
        )
        for metric, label in panels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), sharex=False)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, sae_width)
        _finish_metric_axis(
            ax, target_model_density, sae_width, label, density_mode
        )
    if resolved_suite == "stage1":
        axes[1].set_yscale("log")
    axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())
    _add_bottom_method_legend(fig, table)
    _annotate_stage1_style_on_stage2(
        fig,
        metric_suite=resolved_suite,
        is_synthsaebench=is_synthsaebench,
    )
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
    density_mode: str = "reported",
    metric_suite: str = "auto",
):
    target_model_density, sae_width = _plot_axes(
        target_model_density, sae_width, support_density, n_features
    )
    metrics = apply_density_axis(
        metrics, sae_width=sae_width, density_mode=density_mode
    )
    table = aggregate_seed_metrics(metrics)
    is_synthsaebench = "benchmark_model_id" in metrics.columns
    resolved_suite = _resolve_metric_suite(metrics, metric_suite)
    panels = (
        [("mcc", "MCC"), ("uniqueness", "Uniqueness")]
        if resolved_suite == "stage2"
        else [
            ("generalization_error", "Latent-code rel. error"),
            ("decoder_recovery_cosine", "Dict. Cos sim."),
        ]
    )
    panels = [
        (
            _metric_for_density(
                metrics,
                metric,
                density_mode=density_mode,
                is_synthsaebench=is_synthsaebench,
            ),
            (
                "Hard latent-code rel. error"
                if density_mode == "hard"
                and not is_synthsaebench
                and metric == "generalization_error"
                else label
            ),
        )
        for metric, label in panels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), sharex=False)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, sae_width)
        _finish_metric_axis(
            ax, target_model_density, sae_width, label, density_mode
        )
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    if resolved_suite == "stage1":
        axes[0].set_yscale("log")
    axes[1].set_ylim(top=1.01)
    _add_bottom_method_legend(fig, table)
    _annotate_stage1_style_on_stage2(
        fig,
        metric_suite=resolved_suite,
        is_synthsaebench=is_synthsaebench,
    )
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
    density_mode: str = "reported",
    metric_suite: str = "auto",
):
    target_model_density, sae_width = _plot_axes(
        target_model_density, sae_width, support_density, n_features
    )
    metrics = apply_density_axis(
        metrics, sae_width=sae_width, density_mode=density_mode
    )
    table = aggregate_seed_metrics(metrics)
    is_synthsaebench = "benchmark_model_id" in metrics.columns
    resolved_suite = _resolve_metric_suite(metrics, metric_suite)
    panels = (
        [
            ("classification_f1", "F1"),
            ("classification_precision", "Precision"),
            ("classification_recall", "Recall"),
            ("classification_accuracy", "Accuracy"),
        ]
        if resolved_suite == "stage2"
        else [
            ("support_f1", "F1"),
            ("support_average_precision", "AP"),
            ("support_precision", "Precision"),
            ("support_recall", "Recall"),
        ]
    )
    panels = [
        (
            _metric_for_density(
                metrics,
                metric,
                density_mode=density_mode,
                is_synthsaebench=is_synthsaebench,
            ),
            label,
        )
        for metric, label in panels
    ]
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.5), sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, sae_width)
        _finish_metric_axis(
            ax, target_model_density, sae_width, label, density_mode
        )
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[0]:
        if resolved_suite == "stage1" and not is_synthsaebench:
            ax.set_ylim(-0.01, 0.61)
        ax.set_xlabel("")
    _add_bottom_method_legend(fig, table)
    _annotate_stage1_style_on_stage2(
        fig,
        metric_suite=resolved_suite,
        is_synthsaebench=is_synthsaebench,
    )
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
    density_mode: str = "reported",
    metric_suite: str = "auto",
):
    target_model_density, sae_width = _plot_axes(
        target_model_density, sae_width, support_density, n_features
    )
    metrics = apply_density_axis(
        metrics, sae_width=sae_width, density_mode=density_mode
    )
    table = aggregate_seed_metrics(metrics)
    is_synthsaebench = "benchmark_model_id" in metrics.columns
    resolved_suite = _resolve_metric_suite(metrics, metric_suite)
    panels = (
        [
            ("sae_l0", r"SAE L0 / $d_\mathrm{sae}$"),
            ("true_l0", r"True L0 / $d_\mathrm{sae}$"),
            ("dead_fraction", "Dead latent fraction"),
            ("expected_l0", r"Expected L0 / $d_\mathrm{sae}$"),
        ]
        if resolved_suite == "stage2"
        else [
            ("selection_error", "Selection error"),
            ("dead_fraction", "dead latent fraction"),
            ("average_l0", r"Avg. L0 / $d_\mathrm{sae}$"),
            ("expected_l0", r"Exp. L0 / $d_\mathrm{sae}$"),
        ]
    )
    panels = [
        (
            _metric_for_density(
                metrics,
                metric,
                density_mode=density_mode,
                is_synthsaebench=is_synthsaebench,
            ),
            "Hard selection error"
            if density_mode == "hard"
            and not is_synthsaebench
            and metric == "selection_error"
            else label,
        )
        for metric, label in panels
    ]
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.4), sharex=True)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            _metric_line(ax, table, method, metric, sae_width)
        _finish_metric_axis(
            ax, target_model_density, sae_width, label, density_mode
        )
    if density_mode == "hard":
        for ax in axes:
            ax.set_xlabel(r"Hard density (L0/$d_\mathrm{sae}$)")
    observed_density = table["rho_model"].to_numpy(dtype=float)
    observed_density = observed_density[
        np.isfinite(observed_density) & (observed_density > 0.0)
    ]
    identity_bounds = np.append(observed_density, target_model_density)
    density_min = float(identity_bounds.min())
    density_max = float(identity_bounds.max())
    density = (
        np.asarray([density_min])
        if density_min == density_max
        else np.geomspace(density_min, density_max, num=100)
    )
    identity_axes = axes[:1] if resolved_suite == "stage2" else axes[2:]
    for ax in identity_axes:
        ax.plot(density, density, color="black", linestyle="--", linewidth=1, alpha=0.6)
    _add_bottom_method_legend(fig, table)
    _annotate_stage1_style_on_stage2(
        fig,
        metric_suite=resolved_suite,
        is_synthsaebench=is_synthsaebench,
    )
    _save(fig, output_path)
    return fig


def plot_vg_posterior_diagnostics(
    metrics: pd.DataFrame,
    *,
    target_model_density: float,
    sae_width: int,
    output_path: Path | str | None = None,
    density_mode: str = "reported",
):
    """Show whether VG hard inference agrees with its variational expectation."""

    required = {"vg_expected_explained_variance", "vg_expected_l0"}
    if not required.issubset(metrics.columns):
        raise ValueError("VG posterior diagnostic columns are absent from metrics.")
    metrics = apply_density_axis(
        metrics, sae_width=sae_width, density_mode=density_mode
    )
    table = aggregate_seed_metrics(metrics)
    subset = table[table["method"] == "vgsae"].sort_values("rho_model")
    if subset.empty:
        raise ValueError("VG posterior diagnostics require at least one VG-SAE run.")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    axes[0].plot(
        subset["rho_model"],
        subset["explained_variance"],
        marker="o",
        label="Hard inference",
    )
    axes[0].plot(
        subset["rho_model"],
        subset["vg_expected_explained_variance"],
        marker="s",
        label="Posterior expectation",
    )
    axes[1].plot(
        subset["rho_model"],
        subset["sae_l0"] / sae_width,
        marker="o",
        label="Hard inference",
    )
    axes[1].plot(
        subset["rho_model"],
        subset["vg_expected_l0"] / sae_width,
        marker="s",
        label="Posterior expectation",
    )
    _finish_metric_axis(
        axes[0],
        target_model_density,
        sae_width,
        r"VG explained variance ($R^2$)",
        density_mode,
    )
    _finish_metric_axis(
        axes[1],
        target_model_density,
        sae_width,
        r"VG L0 / $d_\mathrm{sae}$",
        density_mode,
    )
    axes[1].set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.set_layout_engine("constrained")
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=True)
    _save(fig, output_path)
    return fig


def plot_training_curves(history: pd.DataFrame, output_path: Path | str | None = None):
    panels = [
        ("loss", "method-specific training loss"),
        ("reconstruction_mse", "train reconstruction MSE"),
        ("rho", "train rho"),
    ]
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
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, min_n_ticks=4))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_step_tick))
        ax.grid(alpha=0.25)
    _add_bottom_method_legend(fig, history)
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
    run_roots: dict[tuple[str, str] | str, Path | str] | None = None,
    sae_width: int | None = None,
    density_mode: str = "reported",
) -> tuple[Any, pd.DataFrame]:
    if target_model_density is None:
        if support_density is None:
            raise TypeError("target_model_density is required.")
        target_model_density = support_density
    if sae_width is None:
        if density_mode == "hard":
            raise TypeError("sae_width is required for hard-density mask selection.")
    else:
        metrics = apply_density_axis(
            metrics, sae_width=sae_width, density_mode=density_mode
        )
    representatives = _mask_representatives(
        metrics, target_model_density, representative_seed
    )
    fig, axes = plt.subplots(
        len(representatives), 2, figsize=(10, 2.2 * len(representatives)), sharex=True
    )
    axes = np.atleast_2d(axes)
    root = Path(sweep_dir)
    for row_index, row in representatives.iterrows():
        root_key = (str(row["method"]), str(row["run_id"]))
        legacy_key = str(row["run_id"])
        cache_root = root
        if run_roots is not None:
            if root_key in run_roots:
                cache_root = Path(run_roots[root_key])
            elif legacy_key in run_roots:
                cache_root = Path(run_roots[legacy_key])
        cache_path = cache_root / "runs" / row["method"] / row["run_id"] / "eval" / checkpoint_kind / "cache.npz"
        with np.load(cache_path) as cache:
            support = cache["true_support"][:n_show]
            if density_mode == "hard":
                if "hard_mask" not in cache:
                    raise ValueError(
                        f"Hard-density heatmap requires hard_mask in {cache_path}; "
                        "rerun Stage-1 evaluation with the current metric schema."
                    )
                mask = cache["hard_mask"][:n_show]
            else:
                mask = cache["mask"][:n_show]
        axes[row_index, 0].imshow(support, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        axes[row_index, 0].set(ylabel=row["method_label"], title="true support")
        axes[row_index, 1].imshow(mask, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        density_name = "hard rho" if density_mode == "hard" else "rho"
        selection_metric = (
            "hard_selection_error" if density_mode == "hard" else "selection_error"
        )
        if selection_metric not in row:
            raise ValueError(
                f"Hard-density heatmap requires {selection_metric!r}; rerun evaluation."
            )
        selection_name = "hard sel err" if density_mode == "hard" else "sel err"
        axes[row_index, 1].set_title(
            f"mask, {density_name}={row['rho_model']:.3f}, "
            f"{selection_name}={row[selection_metric]:.3f}"
        )
        matching_policy = str(row.get("matching_policy", ""))
        ground_truth_width = int(row.get("ground_truth_num_features", support.shape[1]))
        if support.shape[1] > ground_truth_width and "per_latent_best" not in matching_policy:
            for ax in axes[row_index]:
                ax.axvline(ground_truth_width - 0.5, color="white", linewidth=1)
    synth_alignment = (
        "matching_policy" in representatives
        and representatives["matching_policy"].astype(str).str.contains("per_latent_best").any()
    )
    xlabel = (
        "SAE latent / best-matched GT feature"
        if synth_alignment
        else "GT features + unmatched SAE latents"
    )
    for ax in axes.ravel():
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    _save(fig, output_path)
    return fig, representatives


def plot_all(
    sweep_dir: Path | str,
    *,
    checkpoint_kind: str = "last",
    output_dir: Path | str | None = None,
    baseline_sweep_dir: Path | str | None = None,
    density_mode: str = "reported",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reproduce notebook-10 visuals on the reported or hard-density x-axis."""

    root = Path(sweep_dir)
    metrics, history, run_roots = load_comparison_results(
        root, checkpoint_kind, baseline_sweep_dir
    )
    context = load_sweep_plot_context(root)
    target_density = float(
        context[
            "target_model_density_empirical"
            if density_mode == "hard"
            else "target_model_density_expected"
        ]
    )
    outputs = Path(output_dir) if output_dir is not None else None
    destination = lambda name: outputs / name if outputs is not None else None
    figures = {
        "data_overview": plot_data_overview(root, destination("data_overview.png")),
        "reconstruction": plot_reconstruction_metrics(
            metrics,
            target_model_density=target_density,
            sae_width=context["sae_width"],
            output_path=destination("reconstruction_metrics.png"),
            density_mode=density_mode,
        ),
        "recovery": plot_recovery_metrics(
            metrics,
            target_model_density=target_density,
            sae_width=context["sae_width"],
            output_path=destination("recovery_metrics.png"),
            density_mode=density_mode,
        ),
        "support": plot_support_metrics(
            metrics,
            target_model_density=target_density,
            sae_width=context["sae_width"],
            output_path=destination("support_metrics.png"),
            density_mode=density_mode,
        ),
        "sparsity": plot_sparsity_diagnostics(
            metrics,
            target_model_density=target_density,
            sae_width=context["sae_width"],
            output_path=destination("sparsity_diagnostics.png"),
            density_mode=density_mode,
        ),
        "training": plot_training_curves(history, destination("training_curves.png")),
    }
    if "benchmark_model_id" in metrics.columns:
        figures.update(
            {
                "stage1_reconstruction": plot_reconstruction_metrics(
                    metrics,
                    target_model_density=target_density,
                    sae_width=context["sae_width"],
                    output_path=destination(
                        "stage1_style_reconstruction_metrics.png"
                    ),
                    density_mode=density_mode,
                    metric_suite="stage1",
                ),
                "stage1_recovery": plot_recovery_metrics(
                    metrics,
                    target_model_density=target_density,
                    sae_width=context["sae_width"],
                    output_path=destination("stage1_style_recovery_metrics.png"),
                    density_mode=density_mode,
                    metric_suite="stage1",
                ),
                "stage1_support": plot_support_metrics(
                    metrics,
                    target_model_density=target_density,
                    sae_width=context["sae_width"],
                    output_path=destination("stage1_style_support_metrics.png"),
                    density_mode=density_mode,
                    metric_suite="stage1",
                ),
                "stage1_sparsity": plot_sparsity_diagnostics(
                    metrics,
                    target_model_density=target_density,
                    sae_width=context["sae_width"],
                    output_path=destination(
                        "stage1_style_sparsity_diagnostics.png"
                    ),
                    density_mode=density_mode,
                    metric_suite="stage1",
                ),
            }
        )
    heatmap, representatives = plot_mask_heatmaps(
        root,
        metrics,
        target_model_density=target_density,
        checkpoint_kind=checkpoint_kind,
        output_path=destination("mask_heatmaps.png"),
        run_roots=run_roots,
        sae_width=int(context["sae_width"]),
        density_mode=density_mode,
    )
    figures["masks"] = heatmap
    if "vg_expected_l0" in metrics.columns and (metrics["method"] == "vgsae").any():
        figures["vg_posterior"] = plot_vg_posterior_diagnostics(
            metrics,
            target_model_density=target_density,
            sae_width=int(context["sae_width"]),
            output_path=destination("vg_posterior_diagnostics.png"),
            density_mode=density_mode,
        )
    return figures, representatives
