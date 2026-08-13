"""Headless plotting helpers for Stage-3 real-activation sweeps.

Every comparison uses the achieved hard L0 measured during evaluation.  This
is intentionally different from plotting a configured sparsity control: VG,
BatchTopK, and JumpReLU controls do not have a common numerical meaning.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HARD_L0_COLUMNS = ("average_l0", "sae_l0", "hard_l0", "achieved_l0")
METHOD_LABELS = {
    "vgsae": "VG-SAE",
    "l1": "L1/ReLU SAE",
    "batchtopk": "BatchTopK SAE",
    "jumprelu": "JumpReLU SAE",
}
METHOD_COLORS = {
    "vgsae": "tab:blue",
    "l1": "tab:orange",
    "batchtopk": "tab:purple",
    "jumprelu": "tab:brown",
}
RECONSTRUCTION_METRICS = (
    ("explained_variance", "Explained variance", ("explained_variance",)),
    ("reconstruction_mse", "Reconstruction MSE", ("reconstruction_mse", "mse")),
    (
        "reconstruction_cosine",
        "Reconstruction cosine",
        ("reconstruction_cosine", "cossim", "reconstruction_cossim"),
    ),
    ("ce_loss_score", "CE loss score", ("ce_loss_score", "ce_score")),
    (
        "kl_div_score",
        "KL divergence score",
        ("kl_div_score", "kl_score", "kl_divergence_score"),
    ),
)
SPARSITY_METRICS = (
    ("hard_l0", "Achieved hard L0", ("__hard_l0",)),
    (
        "density",
        "Hard activation density",
        ("rho_model", "rho_model_hard", "hard_density", "activation_density"),
    ),
    ("dead_fraction", "Dead latent fraction", ("dead_fraction",)),
    ("vg_expected_l0", "VG expected L0", ("vg_expected_l0", "expected_l0")),
)
PAIRWISE_METRIC = (
    "decoder_pairwise_cosine",
    "Decoder pairwise cosine",
    ("decoder_pairwise_cosine_similarity", "decoder_pairwise_cosine"),
)


def _coalesced_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Coalesce numeric aliases row-wise, treating strings such as None as missing."""

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[column], errors="coerce").astype(float)
        candidate[~np.isfinite(candidate)] = np.nan
        result = result.fillna(candidate)
    return result


def _nonempty_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _format_layer(value: object) -> str | None:
    text = _nonempty_text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _layer_from_hook(value: object) -> str | None:
    hook = _nonempty_text(value)
    if hook is None:
        return None
    match = re.search(r"(?:blocks|layers)\.(\d+)", hook)
    return match.group(1) if match is not None else hook


def _first_text(row: pd.Series, columns: Iterable[str]) -> str | None:
    for column in columns:
        if column in row.index:
            value = _nonempty_text(row[column])
            if value is not None:
                return value
    return None


def _facet_label(row: pd.Series) -> str:
    model = _first_text(row, ("model_name", "model_id", "model"))
    layer = _format_layer(row.get("layer")) if "layer" in row.index else None
    if layer is None:
        layer = _layer_from_hook(row.get("hook_name"))
    if model is not None and layer is not None:
        return f"{model} · layer {layer}"
    if model is not None:
        return model
    if layer is not None:
        return f"layer {layer}"
    return "all runs"


def _method_key_and_label(row: pd.Series) -> tuple[str, str]:
    method = _first_text(row, ("method", "method_label")) or "all"
    label = _first_text(row, ("method_label",))
    return method, label or METHOD_LABELS.get(method, method)


def _normalized_method(method: str) -> str:
    compact = re.sub(r"[^a-z0-9]", "", method.lower())
    if compact.startswith("vgsae") or compact == "vg":
        return "vgsae"
    if "batchtopk" in compact:
        return "batchtopk"
    if "jumprelu" in compact:
        return "jumprelu"
    if compact.startswith("l1"):
        return "l1"
    return compact


def _prepare_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["__hard_l0"] = _coalesced_numeric(prepared, HARD_L0_COLUMNS)
    if not ((prepared["__hard_l0"] >= 0.0) & prepared["__hard_l0"].notna()).any():
        choices = ", ".join(HARD_L0_COLUMNS)
        raise ValueError(
            "Stage-3 plots require a finite achieved hard L0 in one of: " + choices
        )

    prepared["__facet"] = prepared.apply(_facet_label, axis=1)
    method_metadata = prepared.apply(_method_key_and_label, axis=1)
    prepared["__method"] = [item[0] for item in method_metadata]
    prepared["__method_label"] = [item[1] for item in method_metadata]

    for canonical, _, aliases in (
        *RECONSTRUCTION_METRICS,
        *SPARSITY_METRICS,
        PAIRWISE_METRIC,
    ):
        prepared[f"__metric_{canonical}"] = _coalesced_numeric(prepared, aliases)

    is_vg = prepared["__method"].map(_normalized_method).eq("vgsae")
    prepared.loc[~is_vg, "__metric_vg_expected_l0"] = np.nan

    # Density is derivable from the two operational quantities if an older
    # result omitted rho_model.  It is never inferred from a configured target.
    density_column = "__metric_density"
    if "sae_width" in prepared.columns:
        width = _coalesced_numeric(prepared, ("sae_width",))
        derived = prepared["__hard_l0"] / width.where(width > 0.0)
        prepared[density_column] = prepared[density_column].fillna(derived)
    return prepared


def _method_order(frame: pd.DataFrame) -> list[str]:
    preferred = {name: index for index, name in enumerate(METHOD_LABELS)}
    methods = frame["__method"].drop_duplicates().astype(str).tolist()
    return sorted(
        methods,
        key=lambda method: (
            preferred.get(_normalized_method(method), len(preferred)),
            method,
        ),
    )


def _curve_points(frame: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "x": frame["__hard_l0"],
            "y": frame[metric_column],
        },
        index=frame.index,
    )
    valid = np.isfinite(work["x"]) & np.isfinite(work["y"]) & (work["x"] >= 0.0)
    work = work.loc[valid].copy()
    if work.empty:
        return pd.DataFrame(columns=("x", "y", "y_std", "count"))

    if (
        "control_value" in frame.columns
        and frame.loc[valid, "control_value"].notna().any()
    ):
        controls = frame.loc[valid, "control_value"].map(_nonempty_text)
        if "control_name" in frame.columns:
            names = frame.loc[valid, "control_name"].map(_nonempty_text)
        else:
            names = pd.Series(None, index=controls.index, dtype=object)
        keys = []
        for index in work.index:
            control = controls.loc[index]
            name = names.loc[index]
            keys.append(
                f"{name or 'control'}:{control}"
                if control is not None
                else f"achieved:{work.at[index, 'x']:.12g}"
            )
        work["group"] = keys
    else:
        work["group"] = work["x"].map(lambda value: f"achieved:{value:.12g}")

    points = (
        work.groupby("group", sort=False)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            y_std=("y", "std"),
            count=("y", "size"),
        )
        .sort_values("x")
        .reset_index(drop=True)
    )
    points["y_std"] = points["y_std"].fillna(0.0)
    return points


def _plot_family(
    frame: pd.DataFrame,
    metrics: Iterable[tuple[str, str, tuple[str, ...]]],
    output_path: Path,
) -> Path:
    available = [
        (canonical, label, aliases)
        for canonical, label, aliases in metrics
        if frame[f"__metric_{canonical}"].notna().any()
    ]
    if not available:
        available = [("missing", "No finite metrics available", ())]

    facets = frame["__facet"].drop_duplicates().astype(str).tolist() or ["all runs"]
    fig, axes = plt.subplots(
        len(facets),
        len(available),
        figsize=(3.65 * len(available), 2.75 * len(facets)),
        squeeze=False,
        constrained_layout=True,
    )
    method_order = _method_order(frame)
    legend_handles: dict[str, object] = {}

    for row_index, facet in enumerate(facets):
        facet_frame = frame[frame["__facet"] == facet]
        for column_index, (canonical, label, _) in enumerate(available):
            ax = axes[row_index, column_index]
            ax.set_title(f"{facet}\n{label}", fontsize=9)
            ax.set_xlabel("Achieved hard L0")
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
            if canonical == "missing":
                ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes)
                continue

            metric_column = f"__metric_{canonical}"
            plotted = False
            for color_index, method in enumerate(method_order):
                subset = facet_frame[facet_frame["__method"] == method]
                points = _curve_points(subset, metric_column)
                if points.empty:
                    continue
                display_values = subset["__method_label"].dropna().astype(str)
                display = display_values.iloc[0] if not display_values.empty else method
                normalized = _normalized_method(method)
                color = METHOD_COLORS.get(
                    normalized, plt.get_cmap("tab20")(color_index % 20)
                )
                (line,) = ax.plot(
                    points["x"],
                    points["y"],
                    color=color,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.4,
                    label=display,
                )
                if (points["count"] > 1).any() and (points["y_std"] > 0.0).any():
                    ax.fill_between(
                        points["x"],
                        points["y"] - points["y_std"],
                        points["y"] + points["y_std"],
                        color=color,
                        alpha=0.13,
                        linewidth=0.0,
                    )
                legend_handles.setdefault(display, line)
                plotted = True
            if not plotted:
                ax.text(
                    0.5,
                    0.5,
                    "No finite values",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="outside lower center",
            ncols=min(4, len(legend_handles)),
            frameon=True,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_all(sweep_dir: Path | str) -> list[Path]:
    """Create all Stage-3 figures and return their paths.

    Input is always ``summary/last/final_metrics.csv`` below ``sweep_dir`` and
    figures are written alongside it in ``summary/last/figures``.
    """

    root = Path(sweep_dir)
    metrics_path = root / "summary" / "last" / "final_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        raise ValueError(f"Stage-3 metrics file is empty: {metrics_path}")
    prepared = _prepare_metrics(metrics)
    output_dir = metrics_path.parent / "figures"

    paths = [
        _plot_family(
            prepared,
            RECONSTRUCTION_METRICS,
            output_dir / "reconstruction_metrics.png",
        ),
        _plot_family(
            prepared,
            SPARSITY_METRICS,
            output_dir / "sparsity_diagnostics.png",
        ),
        _plot_family(
            prepared,
            (PAIRWISE_METRIC,),
            output_dir / "decoder_pairwise_cosine.png",
        ),
    ]
    return paths
