"""Summarize the Stage-1 amplitude/frequency factorial on the hard-code axis."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sae_sweep import (  # noqa: E402
    CONTROL_NAMES,
    FULL_CONTROLS,
    METHOD_LABELS,
    METHOD_ORDER,
)
from src.sae_sweep_plot import (  # noqa: E402
    METHOD_COLORS,
    apply_density_axis,
    load_sweep_plot_context,
    load_sweep_results,
)


@dataclass(frozen=True)
class Condition:
    condition_id: str
    amplitude_mode: str
    frequency_skew: float
    root_token: str | None


CONDITIONS = (
    Condition("exp_skew05", "exponential", 0.5, None),
    Condition("constant_skew05", "constant", 0.5, "ablation2_constant"),
    Condition("uniform_skew05", "uniform", 0.5, "ablation2_uniform"),
    Condition("exp_uniformfreq", "exponential", 0.0, "ablation3_uniformfreq"),
    Condition(
        "constant_uniformfreq",
        "constant",
        0.0,
        "ablation23_constant_uniformfreq",
    ),
    Condition(
        "uniform_uniformfreq",
        "uniform",
        0.0,
        "ablation23_uniform_uniformfreq",
    ),
)
BETA_MODES = ("profiled", "learned")
CHECKPOINT_KINDS = ("last", "best")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs" / "stage1_ablation23_factorial_analysis"
EXPECTED_RUNS = sum(len(FULL_CONTROLS[method]) for method in METHOD_ORDER)
EXPECTED_STAGE1_AXES = {
    "input_dim": 128,
    "ground_truth_num_features": 1024,
    "sae_width": 1024,
    "support_density": 0.01,
    "amplitude_scale": 1.0,
}
TARGET_KINDS = ("expected", "empirical")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: set[str], *, source: str) -> None:
    if missing := sorted(columns - set(frame.columns)):
        raise ValueError(f"{source} is missing required columns: {missing}.")


def _constant_strings(frame: pd.DataFrame, column: str) -> set[str]:
    return set(frame[column].astype(str))


def _constant_numeric(
    frame: pd.DataFrame,
    column: str,
    expected: float,
    *,
    source: str,
    atol: float = 1.0e-12,
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not (
        np.isfinite(values).all()
        and np.allclose(values, expected, rtol=0.0, atol=atol)
    ):
        observed = sorted(set(values[np.isfinite(values)].tolist()))
        raise ValueError(
            f"{source} {column}={observed}; expected only {expected!r}."
        )


def _validate_run_grid(
    metrics: pd.DataFrame,
    *,
    root: Path,
    beta_mode: str,
    checkpoint_kind: str,
) -> None:
    source = f"{root.name}[{checkpoint_kind}]"
    _require_columns(
        metrics,
        {
            "run_id",
            "seed",
            "method",
            "beta_mode",
            "checkpoint_kind",
            "control_name",
            "control_value",
            "input_dim",
            "ground_truth_num_features",
            "sae_width",
            "support_density",
            "rho_model_hard",
            "average_l0",
            "hard_generalization_error",
            "decoder_recovery_cosine",
            "hard_metric_schema_version",
            "target_model_density_expected",
            "target_model_density_empirical",
            "train_source_fingerprint",
            "train_pipeline_fingerprint",
            "eval_source_fingerprint",
        },
        source=source,
    )
    if len(metrics) != EXPECTED_RUNS:
        raise ValueError(f"{source} has {len(metrics)} rows; expected {EXPECTED_RUNS}.")
    if metrics["run_id"].isna().any() or metrics["run_id"].duplicated().any():
        duplicates = sorted(
            set(metrics.loc[metrics["run_id"].duplicated(False), "run_id"].astype(str))
        )
        raise ValueError(f"{source} has missing or duplicate run_id values: {duplicates}.")
    if set(pd.to_numeric(metrics["seed"], errors="coerce")) != {0}:
        raise ValueError(f"{source} must contain exactly seed 0.")
    if _constant_strings(metrics, "beta_mode") != {beta_mode}:
        raise ValueError(f"{source} has the wrong beta_mode axis.")
    if _constant_strings(metrics, "checkpoint_kind") != {checkpoint_kind}:
        raise ValueError(f"{source} has the wrong checkpoint_kind axis.")
    if set(metrics["method"].astype(str)) != set(METHOD_ORDER):
        raise ValueError(f"{source} does not contain exactly the six methods.")

    for method in METHOD_ORDER:
        method_rows = metrics[metrics["method"] == method]
        expected_controls = np.sort(np.asarray(FULL_CONTROLS[method], dtype=float))
        observed_controls = np.sort(
            pd.to_numeric(method_rows["control_value"], errors="coerce").to_numpy(
                dtype=float
            )
        )
        if (
            len(method_rows) != len(expected_controls)
            or not np.isfinite(observed_controls).all()
            or not np.allclose(
                observed_controls, expected_controls, rtol=0.0, atol=1.0e-12
            )
        ):
            raise ValueError(f"{source} has an incomplete {method} control grid.")
        if set(method_rows["control_name"].astype(str)) != {CONTROL_NAMES[method]}:
            raise ValueError(f"{source} has the wrong {method} control_name.")


def _validate_condition_axes(
    metrics: pd.DataFrame,
    context: dict[str, float | int | str],
    condition: Condition,
    *,
    root: Path,
    checkpoint_kind: str,
) -> None:
    source = f"{root.name}[{checkpoint_kind}]"
    for column, expected in EXPECTED_STAGE1_AXES.items():
        if column in context:
            context_value = float(context[column])
            if not math.isclose(
                context_value, expected, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise ValueError(
                    f"{source} context {column}={context_value}; expected {expected}."
                )
        if column in metrics:
            _constant_numeric(
                metrics, column, expected, source=source, atol=1.0e-9
            )
        if column not in context and column not in metrics:
            raise ValueError(f"{source} does not record the {column} data axis.")
    if str(context["amplitude_mode"]) != condition.amplitude_mode:
        raise ValueError(
            f"{root.name}: amplitude_mode={context['amplitude_mode']!r}; "
            f"expected {condition.amplitude_mode!r}."
        )
    if not math.isclose(
        float(context["frequency_skew"]),
        condition.frequency_skew,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"{root.name}: frequency_skew={context['frequency_skew']!r}; "
            f"expected {condition.frequency_skew!r}."
        )
    if "amplitude_mode" in metrics and _constant_strings(
        metrics, "amplitude_mode"
    ) != {condition.amplitude_mode}:
        raise ValueError(f"{source} has the wrong amplitude_mode row metadata.")
    if "frequency_skew" in metrics:
        _constant_numeric(
            metrics,
            "frequency_skew",
            condition.frequency_skew,
            source=source,
        )
    for target_kind in TARGET_KINDS:
        context_target = float(context[f"target_model_density_{target_kind}"])
        _constant_numeric(
            metrics,
            f"target_model_density_{target_kind}",
            context_target,
            source=source,
            # CPU/GPU float32 reductions of the expected feature count differ
            # by roughly 1e-9 in legacy Stage-1 artifacts.
            atol=1.0e-8,
        )


def _root_audit(
    metrics: pd.DataFrame,
    context: dict[str, float | int | str],
    condition: Condition,
    beta_mode: str,
    checkpoint_kind: str,
    root: Path,
) -> dict[str, object]:
    source = f"{root.name}[{checkpoint_kind}]"
    summary_path = root / "summary" / checkpoint_kind / "summary.json"
    summary = json.loads(summary_path.read_text())
    sweep_config_path = root / "sweep_config.json"
    sweep_config = json.loads(sweep_config_path.read_text())
    data_config = sweep_config.get("data", {})
    training_config = sweep_config.get("training", {})
    expected_data_config = {
        "kind": "stage1_custom_baseline",
        "input_dim": 128,
        "ground_truth_num_features": 1024,
        "sae_width": 1024,
        "n_train": 8196,
        "n_test": 1024,
        "support_density": 0.01,
        "coherence": 0.0,
        "noise_std": 0.0,
        "frequency_skew": condition.frequency_skew,
        "amplitude_scale": 1.0,
    }
    for key, expected in expected_data_config.items():
        if data_config.get(key) != expected:
            raise ValueError(
                f"{source} sweep data {key}={data_config.get(key)!r}; "
                f"expected {expected!r}."
            )
    if data_config.get("amplitude_mode", "exponential") != condition.amplitude_mode:
        raise ValueError(f"{source} sweep data has the wrong amplitude_mode.")
    expected_training_config = {
        "beta_mode": beta_mode,
        "batch_size": 128,
        "train_steps": 1000,
    }
    for key, expected in expected_training_config.items():
        if training_config.get(key) != expected:
            raise ValueError(
                f"{source} sweep training {key}={training_config.get(key)!r}; "
                f"expected {expected!r}."
            )
    if set(sweep_config.get("methods", [])) != set(METHOD_ORDER):
        raise ValueError(f"{source} sweep config has the wrong method set.")
    if sweep_config.get("seeds") != [0]:
        raise ValueError(f"{source} sweep config must contain exactly seed 0.")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_runs = manifest.get("runs", [])
    if len(manifest_runs) != EXPECTED_RUNS or {
        str(item.get("run_id")) for item in manifest_runs
    } != set(metrics["run_id"].astype(str)):
        raise ValueError(f"{source} manifest and metric run IDs disagree.")
    provenance_sets = {
        column: _constant_strings(metrics, column)
        for column in (
            "train_source_fingerprint",
            "train_pipeline_fingerprint",
            "eval_source_fingerprint",
        )
    }
    for column, values in provenance_sets.items():
        if len(values) != 1 or not next(iter(values)) or "nan" in values:
            raise ValueError(f"{source} has mixed or empty {column} values.")
    train_source = next(iter(provenance_sets["train_source_fingerprint"]))
    train_pipeline = next(iter(provenance_sets["train_pipeline_fingerprint"]))
    eval_source = next(iter(provenance_sets["eval_source_fingerprint"]))
    if not train_source or not train_pipeline or not eval_source:
        raise ValueError(f"{source} has an empty provenance fingerprint.")
    expected_summary = {
        "checkpoint_kind": checkpoint_kind,
        "beta_mode": beta_mode,
        "n_evaluated_runs": EXPECTED_RUNS,
        "train_source_fingerprint": train_source,
        "train_pipeline_fingerprint": train_pipeline,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{source} summary {key}={summary.get(key)!r}; expected {expected!r}."
            )
    if set(summary.get("methods", [])) != set(METHOD_ORDER):
        raise ValueError(f"{source} summary has the wrong method set.")
    train_git_revisions: set[str] = set()
    for manifest_run in manifest_runs:
        run_config = json.loads(
            (
                root
                / str(manifest_run["relative_dir"])
                / "config.json"
            ).read_text()
        )
        train_provenance = run_config.get("train_provenance", {})
        if (
            train_provenance.get("source_fingerprint") != train_source
            or train_provenance.get("pipeline_fingerprint") != train_pipeline
        ):
            raise ValueError(
                f"{source} train provenance conflicts with metric rows."
            )
        train_git_revisions.add(str(train_provenance.get("git_revision", "")))
    if not train_git_revisions or "" in train_git_revisions:
        raise ValueError(f"{source} has missing training git revisions.")
    eval_provenance = summary.get("eval_provenance", {})
    eval_pipeline = str(summary.get("eval_fingerprint", ""))
    if (
        not eval_pipeline
        or eval_provenance.get("pipeline_fingerprint") != eval_pipeline
        or eval_provenance.get("source_fingerprint") != eval_source
    ):
        raise ValueError(f"{source} eval provenance conflicts with metric rows.")
    metrics_path = root / "summary" / checkpoint_kind / "final_metrics.csv"
    return {
        **asdict(condition),
        "beta_mode": beta_mode,
        "checkpoint_kind": checkpoint_kind,
        "sweep_root": root.name,
        "n_rows": len(metrics),
        "seeds": "0",
        "input_dim": EXPECTED_STAGE1_AXES["input_dim"],
        "ground_truth_num_features": EXPECTED_STAGE1_AXES[
            "ground_truth_num_features"
        ],
        "sae_width": EXPECTED_STAGE1_AXES["sae_width"],
        "support_density": EXPECTED_STAGE1_AXES["support_density"],
        "amplitude_scale": EXPECTED_STAGE1_AXES["amplitude_scale"],
        "target_density_expected": float(
            context["target_model_density_expected"]
        ),
        "target_density_expected_artifact": float(
            context["target_model_density_expected_artifact"]
        ),
        "target_density_empirical": float(
            context["target_model_density_empirical"]
        ),
        "train_source_fingerprint": train_source,
        "train_pipeline_fingerprint": train_pipeline,
        "train_git_revisions": ";".join(sorted(train_git_revisions)),
        "n_train_git_revisions": len(train_git_revisions),
        "eval_source_fingerprint": eval_source,
        "eval_pipeline_fingerprint": eval_pipeline,
        "eval_git_revision": str(eval_provenance.get("git_revision", "")),
        "metrics_sha256": _sha256(metrics_path),
        "summary_sha256": _sha256(summary_path),
        "manifest_sha256": _sha256(manifest_path),
        "sweep_config_sha256": _sha256(sweep_config_path),
        "data_preview_sha256": _sha256(root / "summary" / "data_preview.npz"),
    }


def condition_root(condition: Condition, beta_mode: str) -> Path:
    if condition.root_token is None:
        name = (
            f"stage1_beta_{beta_mode}"
            "_din128_gt1024_sae1024_sd001_seed0"
        )
    else:
        name = (
            f"stage1_{condition.root_token}_beta_{beta_mode}"
            "_din128_gt1024_sae1024_sd001_seed0"
        )
    return PROJECT_ROOT / "outputs" / "runs" / name


def load_condition(
    condition: Condition,
    beta_mode: str,
    checkpoint_kind: str,
) -> tuple[
    pd.DataFrame,
    dict[str, float | int | str],
    Path,
    dict[str, object],
]:
    root = condition_root(condition, beta_mode)
    metrics, _ = load_sweep_results(root, checkpoint_kind)
    context = load_sweep_plot_context(root)
    _validate_run_grid(
        metrics,
        root=root,
        beta_mode=beta_mode,
        checkpoint_kind=checkpoint_kind,
    )
    _validate_condition_axes(
        metrics,
        context,
        condition,
        root=root,
        checkpoint_kind=checkpoint_kind,
    )
    context = dict(context)
    artifact_expected_target = float(context["target_model_density_expected"])
    canonical_expected_target = (
        float(context["support_density"])
        * int(context["ground_truth_num_features"])
        / int(context["sae_width"])
    )
    if not math.isclose(
        artifact_expected_target,
        canonical_expected_target,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise ValueError(
            f"{root.name}: artifact expected target {artifact_expected_target} "
            f"disagrees with theoretical target {canonical_expected_target}."
        )
    context["target_model_density_expected_artifact"] = artifact_expected_target
    context["target_model_density_expected"] = canonical_expected_target
    metrics = apply_density_axis(
        metrics,
        sae_width=int(context["sae_width"]),
        density_mode="hard",
    )
    metrics["analysis_target_density_expected"] = canonical_expected_target
    metrics["analysis_target_density_empirical"] = float(
        context["target_model_density_empirical"]
    )
    if set(metrics["density_axis"].astype(str)) != {"hard"}:
        raise ValueError(f"{root.name}: analysis did not select the hard-density axis.")
    stored_hard_density = pd.to_numeric(
        metrics["rho_model_hard"], errors="coerce"
    ).to_numpy(dtype=float)
    analysis_density = pd.to_numeric(
        metrics["rho_model"], errors="coerce"
    ).to_numpy(dtype=float)
    if not (
        np.isfinite(stored_hard_density).all()
        and np.allclose(
            analysis_density,
            stored_hard_density,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError(
            f"{root.name}: average-L0 hard density disagrees with rho_model_hard."
        )
    if set(pd.to_numeric(metrics["hard_metric_schema_version"])) != {1}:
        raise ValueError(f"{root.name} is not entirely hard-metric schema v1.")
    audit = _root_audit(
        metrics, context, condition, beta_mode, checkpoint_kind, root
    )
    return metrics, context, root, audit


def _safe_log2_ratio(value: float, target: float) -> float:
    return math.log2(value / target) if value > 0.0 and target > 0.0 else math.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0.0 else math.nan


def _require_hard_axis(frame: pd.DataFrame, *, source: str) -> None:
    if "density_axis" not in frame or set(frame["density_axis"].astype(str)) != {
        "hard"
    }:
        raise ValueError(f"{source} requires density_axis='hard'.")
    density = pd.to_numeric(frame["rho_model"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(density).all():
        raise ValueError(f"{source} hard rho_model values must be finite.")
    if "rho_model_hard" in frame:
        stored = pd.to_numeric(
            frame["rho_model_hard"], errors="coerce"
        ).to_numpy(dtype=float)
        if not (
            np.isfinite(stored).all()
            and np.allclose(density, stored, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(f"{source} rho_model is not the stored hard density.")


def _metric_optimum(
    frame: pd.DataFrame,
    metric: str,
    *,
    maximize: bool,
) -> pd.Series:
    """Select an observed optimum with a stable, documented tie policy."""

    ordered = frame.assign(
        _metric_order=(
            -pd.to_numeric(frame[metric], errors="coerce")
            if maximize
            else pd.to_numeric(frame[metric], errors="coerce")
        )
    ).sort_values(
        ["_metric_order", "rho_model", "control_value", "run_id"],
        kind="stable",
    )
    return ordered.iloc[0]


def _nearest_target_error_row(frame: pd.DataFrame, target: float) -> pd.Series:
    """Choose the best-error row at the closest observed hard density."""

    ordered = frame.assign(
        _target_distance=(
            pd.to_numeric(frame["rho_model"], errors="coerce") - target
        ).abs()
    ).sort_values(
        [
            "_target_distance",
            "hard_generalization_error",
            "rho_model",
            "control_value",
            "run_id",
        ],
        kind="stable",
    )
    return ordered.iloc[0]


def _is_density_boundary(row: pd.Series, densities: np.ndarray) -> bool:
    density = float(row["rho_model"])
    return bool(
        np.isclose(density, float(densities.min()))
        or np.isclose(density, float(densities.max()))
    )


def _is_control_boundary(row: pd.Series, frame: pd.DataFrame) -> bool:
    controls = pd.to_numeric(frame["control_value"], errors="coerce").to_numpy(
        dtype=float
    )
    control = float(row["control_value"])
    return bool(
        np.isclose(control, float(controls.min()))
        or np.isclose(control, float(controls.max()))
    )


def _density_monotonicity(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("control_value", kind="stable")
    densities = pd.to_numeric(ordered["rho_model"], errors="coerce").to_numpy(
        dtype=float
    )
    differences = np.diff(densities)
    tolerance = 1.0e-12
    nondecreasing = bool((differences >= -tolerance).all())
    nonincreasing = bool((differences <= tolerance).all())
    if nondecreasing and nonincreasing:
        return "constant"
    if nondecreasing:
        return "nondecreasing"
    if nonincreasing:
        return "nonincreasing"
    return "nonmonotonic"


def _target_error_diagnostics(
    frame: pd.DataFrame,
    error_row: pd.Series,
    *,
    target_kind: str,
    target: float,
) -> dict[str, object]:
    nearest = _nearest_target_error_row(frame, target)
    optimum_density = float(error_row["rho_model"])
    nearest_density = float(nearest["rho_model"])
    minimum_error = float(error_row["hard_generalization_error"])
    nearest_error = float(nearest["hard_generalization_error"])
    density_min = float(frame["rho_model"].min())
    density_max = float(frame["rho_model"].max())
    regret = nearest_error - minimum_error
    prefix = target_kind
    return {
        f"target_{prefix}_bracketed": density_min <= target <= density_max,
        f"optimum_density_ratio_to_{prefix}_target": _safe_ratio(
            optimum_density, target
        ),
        f"optimum_density_signed_gap_{prefix}": optimum_density - target,
        f"optimum_density_log2_ratio_{prefix}": _safe_log2_ratio(
            optimum_density, target
        ),
        f"nearest_{prefix}_target_hard_density": nearest_density,
        f"nearest_{prefix}_target_density_signed_gap": nearest_density - target,
        f"nearest_{prefix}_target_density_abs_gap": abs(nearest_density - target),
        f"nearest_{prefix}_target_density_ratio": _safe_ratio(
            nearest_density, target
        ),
        f"nearest_{prefix}_target_density_log2_ratio": _safe_log2_ratio(
            nearest_density, target
        ),
        f"nearest_{prefix}_target_hard_generalization_error": nearest_error,
        f"{prefix}_target_regret_absolute": regret,
        f"{prefix}_target_regret_relative": (
            regret / minimum_error if minimum_error > 0.0 else math.nan
        ),
    }


def _cosine_target_diagnostics(
    cosine_row: pd.Series,
    *,
    target_kind: str,
    target: float,
) -> dict[str, float]:
    density = float(cosine_row["rho_model"])
    return {
        f"cosine_optimum_density_ratio_to_{target_kind}_target": _safe_ratio(
            density, target
        ),
        f"cosine_optimum_density_signed_gap_{target_kind}": density - target,
        f"cosine_optimum_density_log2_ratio_{target_kind}": _safe_log2_ratio(
            density, target
        ),
    }


def _near_optimal_diagnostics(
    frame: pd.DataFrame,
    minimum_error: float,
    fraction: float,
) -> dict[str, object]:
    label = f"{int(round(fraction * 100))}pct"
    threshold = minimum_error * (1.0 + fraction)
    near = frame[frame["hard_generalization_error"] <= threshold]
    densities = pd.to_numeric(near["rho_model"], errors="coerce")
    return {
        f"near_optimal_{label}_control_count": len(near),
        f"near_optimal_{label}_density_count": int(densities.nunique()),
        f"near_optimal_{label}_density_min": float(densities.min()),
        f"near_optimal_{label}_density_max": float(densities.max()),
    }


def summarize_condition(
    condition: Condition,
    beta_mode: str,
    checkpoint_kind: str,
    metrics: pd.DataFrame,
    context: dict[str, float | int | str],
    root: Path,
) -> list[dict[str, object]]:
    _require_hard_axis(metrics, source=f"{root.name}[{checkpoint_kind}]")
    rows: list[dict[str, object]] = []
    target_expected = float(context["target_model_density_expected"])
    target_empirical = float(context["target_model_density_empirical"])
    for method in METHOD_ORDER:
        frame = metrics[metrics["method"] == method].copy()
        if "run_id" not in frame:
            frame["run_id"] = [f"{method}-{index}" for index in range(len(frame))]
        frame = frame.sort_values(
            ["rho_model", "control_value", "run_id"], kind="stable"
        )
        densities = frame["rho_model"].to_numpy(dtype=float)
        errors = frame["hard_generalization_error"].to_numpy(dtype=float)
        cosines = frame["decoder_recovery_cosine"].to_numpy(dtype=float)
        if not (
            np.isfinite(densities).all()
            and np.isfinite(errors).all()
            and np.isfinite(cosines).all()
        ):
            raise ValueError(f"{root.name}: non-finite recovery values for {method}.")
        error_row = _metric_optimum(
            frame, "hard_generalization_error", maximize=False
        )
        cosine_row = _metric_optimum(
            frame, "decoder_recovery_cosine", maximize=True
        )
        optimum_density = float(error_row["rho_model"])
        density_min = float(densities.min())
        density_max = float(densities.max())
        control_name = str(error_row["control_name"])
        minimum_error = float(error_row["hard_generalization_error"])
        monotonicity = _density_monotonicity(frame)
        duplicate_groups = int(
            (frame.groupby("rho_model", dropna=False).size() > 1).sum()
        )
        row = {
            "condition_id": condition.condition_id,
            "amplitude_mode": condition.amplitude_mode,
            "frequency_skew": condition.frequency_skew,
            "beta_mode": beta_mode,
            "checkpoint_kind": checkpoint_kind,
            "sweep_root": root.name,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "control_name": control_name,
            "control_value_at_error_optimum": float(error_row["control_value"]),
            "n_controls": len(frame),
            "n_unique_hard_densities": int(frame["rho_model"].nunique()),
            "duplicate_hard_density_groups": duplicate_groups,
            "duplicate_hard_density_excess_rows": (
                len(frame) - int(frame["rho_model"].nunique())
            ),
            "duplicate_density_tie_policy": (
                "nearest absolute density; then minimum hard error; then lower "
                "density, control, run_id"
            ),
            "hard_density_control_monotonicity": monotonicity,
            "hard_density_monotonic_with_control": monotonicity != "nonmonotonic",
            "hard_density_observed_min": density_min,
            "hard_density_observed_max": density_max,
            "target_density_expected": target_expected,
            "target_density_empirical": target_empirical,
            "optimum_hard_density": optimum_density,
            "minimum_hard_generalization_error": minimum_error,
            "optimum_is_density_boundary": _is_density_boundary(
                error_row, densities
            ),
            "optimum_is_control_boundary": _is_control_boundary(error_row, frame),
            "decoder_recovery_cosine_at_error_optimum": float(
                error_row["decoder_recovery_cosine"]
            ),
            "control_value_at_cosine_optimum": float(cosine_row["control_value"]),
            "cosine_optimum_hard_density": float(cosine_row["rho_model"]),
            "maximum_decoder_recovery_cosine": float(
                cosine_row["decoder_recovery_cosine"]
            ),
            "hard_generalization_error_at_cosine_optimum": float(
                cosine_row["hard_generalization_error"]
            ),
            "cosine_optimum_is_density_boundary": _is_density_boundary(
                cosine_row, densities
            ),
            "cosine_optimum_is_control_boundary": _is_control_boundary(
                cosine_row, frame
            ),
            "train_source_fingerprint": str(
                error_row.get("train_source_fingerprint", "")
            ),
            "train_pipeline_fingerprint": str(
                error_row.get("train_pipeline_fingerprint", "")
            ),
            "eval_source_fingerprint": str(
                error_row.get("eval_source_fingerprint", "")
            ),
        }
        row.update(
            _target_error_diagnostics(
                frame,
                error_row,
                target_kind="expected",
                target=target_expected,
            )
        )
        row.update(
            _target_error_diagnostics(
                frame,
                error_row,
                target_kind="empirical",
                target=target_empirical,
            )
        )
        row.update(
            _cosine_target_diagnostics(
                cosine_row,
                target_kind="expected",
                target=target_expected,
            )
        )
        row.update(
            _cosine_target_diagnostics(
                cosine_row,
                target_kind="empirical",
                target=target_empirical,
            )
        )
        row.update(_near_optimal_diagnostics(frame, minimum_error, 0.01))
        row.update(_near_optimal_diagnostics(frame, minimum_error, 0.05))

        # Backward-compatible aliases retain their previous empirical-target
        # semantics while the explicit expected/empirical columns remove any
        # ambiguity for new consumers.
        row.update(
            target_bracketed=row["target_empirical_bracketed"],
            nearest_target_hard_density=row[
                "nearest_empirical_target_hard_density"
            ],
            nearest_target_hard_generalization_error=row[
                "nearest_empirical_target_hard_generalization_error"
            ],
            target_regret_absolute=row["empirical_target_regret_absolute"],
            target_regret_relative=row["empirical_target_regret_relative"],
        )
        rows.append(row)
    return rows


def build_root_records(audits: list[dict[str, object]]) -> pd.DataFrame:
    """Collapse last/best audits after enforcing within-root provenance parity."""

    table = pd.DataFrame(audits)
    expected_rows = len(CONDITIONS) * len(BETA_MODES) * len(CHECKPOINT_KINDS)
    if len(table) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} checkpoint root audits, got {len(table)}."
        )
    records: list[dict[str, object]] = []
    group_columns = [
        "condition_id",
        "amplitude_mode",
        "frequency_skew",
        "root_token",
        "beta_mode",
        "sweep_root",
    ]
    for keys, group in table.groupby(group_columns, dropna=False, sort=False):
        if set(group["checkpoint_kind"].astype(str)) != set(CHECKPOINT_KINDS):
            raise ValueError(f"{keys}: root audit does not contain last and best.")
        indexed = group.set_index("checkpoint_kind")
        parity_columns = (
            "seeds",
            "input_dim",
            "ground_truth_num_features",
            "sae_width",
            "support_density",
            "amplitude_scale",
            "train_source_fingerprint",
            "train_pipeline_fingerprint",
            "train_git_revisions",
            "n_train_git_revisions",
            "eval_source_fingerprint",
            "eval_pipeline_fingerprint",
            "eval_git_revision",
            "sweep_config_sha256",
            "data_preview_sha256",
            "manifest_sha256",
            "target_density_expected",
            "target_density_expected_artifact",
            "target_density_empirical",
        )
        mismatched = [
            column
            for column in parity_columns
            if indexed.loc["last", column] != indexed.loc["best", column]
        ]
        if mismatched:
            raise ValueError(
                f"{keys}: last/best root provenance differs for {mismatched}."
            )
        record = dict(zip(group_columns, keys))
        for column in parity_columns:
            record[column] = indexed.loc["last", column]
        for checkpoint_kind in CHECKPOINT_KINDS:
            record[f"n_rows_{checkpoint_kind}"] = int(
                indexed.loc[checkpoint_kind, "n_rows"]
            )
            record[f"metrics_sha256_{checkpoint_kind}"] = indexed.loc[
                checkpoint_kind, "metrics_sha256"
            ]
            record[f"summary_sha256_{checkpoint_kind}"] = indexed.loc[
                checkpoint_kind, "summary_sha256"
            ]
        record["last_best_provenance_match"] = True
        records.append(record)
    result = pd.DataFrame(records)
    if len(result) != len(CONDITIONS) * len(BETA_MODES):
        raise ValueError("Root audit did not collapse to twelve unique roots.")
    return result


def _summary_row(
    group: pd.DataFrame,
    *,
    amplitude_mode: str,
    frequency_skew: float,
) -> pd.Series:
    selected = group[
        (group["amplitude_mode"] == amplitude_mode)
        & np.isclose(
            pd.to_numeric(group["frequency_skew"], errors="coerce"),
            frequency_skew,
            rtol=0.0,
            atol=1.0e-12,
        )
    ]
    if len(selected) != 1:
        raise ValueError(
            "Factorial summary must have exactly one row for "
            f"amplitude={amplitude_mode}, frequency_skew={frequency_skew}."
        )
    return selected.iloc[0]


def _contrast_record(
    source: pd.Series,
    destination: pd.Series,
    *,
    factor: str,
    level_from: str,
    level_to: str,
    held_constant_factor: str,
    held_constant_level: str,
) -> dict[str, object]:
    density_from = float(source["optimum_hard_density"])
    density_to = float(destination["optimum_hard_density"])
    error_from = float(source["minimum_hard_generalization_error"])
    error_to = float(destination["minimum_hard_generalization_error"])
    cosine_density_from = float(source["cosine_optimum_hard_density"])
    cosine_density_to = float(destination["cosine_optimum_hard_density"])
    cosine_from = float(source["maximum_decoder_recovery_cosine"])
    cosine_to = float(destination["maximum_decoder_recovery_cosine"])
    return {
        "beta_mode": source["beta_mode"],
        "checkpoint_kind": source["checkpoint_kind"],
        "method": source["method"],
        "method_label": source["method_label"],
        "factor": factor,
        "level_from": level_from,
        "level_to": level_to,
        "held_constant_factor": held_constant_factor,
        "held_constant_level": held_constant_level,
        "condition_from": source["condition_id"],
        "condition_to": destination["condition_id"],
        "optimum_hard_density_from": density_from,
        "optimum_hard_density_to": density_to,
        "optimum_hard_density_delta": density_to - density_from,
        "optimum_hard_density_ratio": _safe_ratio(density_to, density_from),
        "optimum_hard_density_log2_ratio": _safe_log2_ratio(
            density_to, density_from
        ),
        "expected_alignment_log2_from": float(
            source["optimum_density_log2_ratio_expected"]
        ),
        "expected_alignment_log2_to": float(
            destination["optimum_density_log2_ratio_expected"]
        ),
        "expected_alignment_log2_delta": float(
            destination["optimum_density_log2_ratio_expected"]
            - source["optimum_density_log2_ratio_expected"]
        ),
        "minimum_hard_generalization_error_from": error_from,
        "minimum_hard_generalization_error_to": error_to,
        "minimum_hard_generalization_error_delta": error_to - error_from,
        "minimum_hard_generalization_error_ratio": _safe_ratio(
            error_to, error_from
        ),
        "minimum_hard_generalization_error_relative_delta": (
            _safe_ratio(error_to - error_from, error_from)
        ),
        "maximum_decoder_recovery_cosine_from": cosine_from,
        "maximum_decoder_recovery_cosine_to": cosine_to,
        "maximum_decoder_recovery_cosine_delta": cosine_to - cosine_from,
        "cosine_optimum_hard_density_from": cosine_density_from,
        "cosine_optimum_hard_density_to": cosine_density_to,
        "cosine_optimum_hard_density_delta": (
            cosine_density_to - cosine_density_from
        ),
        "cosine_optimum_hard_density_log2_ratio": _safe_log2_ratio(
            cosine_density_to, cosine_density_from
        ),
    }


def build_factorial_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    """Return all simple effects in the amplitude x frequency factorial."""

    records: list[dict[str, object]] = []
    groups = summary.groupby(
        ["beta_mode", "checkpoint_kind", "method"], sort=False
    )
    for _keys, group in groups:
        observed = set(group["condition_id"].astype(str))
        expected = {condition.condition_id for condition in CONDITIONS}
        if observed != expected or len(group) != len(CONDITIONS):
            raise ValueError("Each factorial contrast group must contain six conditions.")
        for frequency_skew, frequency_label in (
            (0.5, "skew_0.5"),
            (0.0, "uniform_0"),
        ):
            exponential = _summary_row(
                group,
                amplitude_mode="exponential",
                frequency_skew=frequency_skew,
            )
            for amplitude_mode in ("constant", "uniform"):
                destination = _summary_row(
                    group,
                    amplitude_mode=amplitude_mode,
                    frequency_skew=frequency_skew,
                )
                records.append(
                    _contrast_record(
                        exponential,
                        destination,
                        factor="amplitude_mode",
                        level_from="exponential",
                        level_to=amplitude_mode,
                        held_constant_factor="frequency_skew",
                        held_constant_level=frequency_label,
                    )
                )
        for amplitude_mode in ("exponential", "constant", "uniform"):
            skewed = _summary_row(
                group, amplitude_mode=amplitude_mode, frequency_skew=0.5
            )
            uniform_frequency = _summary_row(
                group, amplitude_mode=amplitude_mode, frequency_skew=0.0
            )
            records.append(
                _contrast_record(
                    skewed,
                    uniform_frequency,
                    factor="frequency_skew",
                    level_from="skew_0.5",
                    level_to="uniform_0",
                    held_constant_factor="amplitude_mode",
                    held_constant_level=amplitude_mode,
                )
            )
    return pd.DataFrame(records)


def build_checkpoint_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare the primary last checkpoint with the best-checkpoint sensitivity."""

    records: list[dict[str, object]] = []
    group_columns = [
        "condition_id",
        "amplitude_mode",
        "frequency_skew",
        "beta_mode",
        "method",
        "method_label",
        "sweep_root",
    ]
    for keys, group in summary.groupby(group_columns, sort=False):
        if set(group["checkpoint_kind"].astype(str)) != set(CHECKPOINT_KINDS):
            raise ValueError(f"{keys}: checkpoint sensitivity requires last and best.")
        indexed = group.set_index("checkpoint_kind")
        last = indexed.loc["last"]
        best = indexed.loc["best"]
        density_last = float(last["optimum_hard_density"])
        density_best = float(best["optimum_hard_density"])
        error_last = float(last["minimum_hard_generalization_error"])
        error_best = float(best["minimum_hard_generalization_error"])
        record = dict(zip(group_columns, keys))
        record.update(
            control_value_last=float(last["control_value_at_error_optimum"]),
            control_value_best=float(best["control_value_at_error_optimum"]),
            optimum_hard_density_last=density_last,
            optimum_hard_density_best=density_best,
            optimum_hard_density_delta_best_minus_last=(
                density_best - density_last
            ),
            optimum_hard_density_log2_ratio_best_over_last=_safe_log2_ratio(
                density_best, density_last
            ),
            minimum_hard_generalization_error_last=error_last,
            minimum_hard_generalization_error_best=error_best,
            minimum_hard_generalization_error_delta_best_minus_last=(
                error_best - error_last
            ),
            minimum_hard_generalization_error_ratio_best_over_last=_safe_ratio(
                error_best, error_last
            ),
            expected_alignment_log2_last=float(
                last["optimum_density_log2_ratio_expected"]
            ),
            expected_alignment_log2_best=float(
                best["optimum_density_log2_ratio_expected"]
            ),
            maximum_decoder_recovery_cosine_last=float(
                last["maximum_decoder_recovery_cosine"]
            ),
            maximum_decoder_recovery_cosine_best=float(
                best["maximum_decoder_recovery_cosine"]
            ),
            cosine_optimum_hard_density_last=float(
                last["cosine_optimum_hard_density"]
            ),
            cosine_optimum_hard_density_best=float(
                best["cosine_optimum_hard_density"]
            ),
        )
        records.append(record)
    return pd.DataFrame(records)


def _condition_labels() -> list[str]:
    return [
        "Exp\nskew",
        "Constant\nskew",
        "Uniform\nskew",
        "Exp\nuniform freq",
        "Constant\nuniform freq",
        "Uniform\nuniform freq",
    ]


def plot_optima(summary: pd.DataFrame, output_path: Path) -> None:
    last = summary[summary["checkpoint_kind"] == "last"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    labels = _condition_labels()
    x = np.arange(len(CONDITIONS))
    condition_ids = [condition.condition_id for condition in CONDITIONS]
    for column, beta_mode in enumerate(BETA_MODES):
        frame = last[last["beta_mode"] == beta_mode]
        for method in METHOD_ORDER:
            method_rows = (
                frame[frame["method"] == method]
                .set_index("condition_id")
                .loc[condition_ids]
            )
            axes[0, column].plot(
                x,
                method_rows["optimum_hard_density"],
                marker="o",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            axes[1, column].plot(
                x,
                method_rows["minimum_hard_generalization_error"],
                marker="o",
                color=METHOD_COLORS[method],
            )
        target_rows = (
            frame.drop_duplicates("condition_id")
            .set_index("condition_id")
            .loc[condition_ids]
        )
        expected_targets = target_rows["target_density_expected"].to_numpy(float)
        empirical_targets = target_rows["target_density_empirical"].to_numpy(float)
        if not np.allclose(
            expected_targets,
            expected_targets[0],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Expected target density is not constant across conditions.")
        axes[0, column].axhline(
            float(expected_targets[0]),
            color="black",
            linestyle="--",
            label="Expected GT",
        )
        axes[0, column].plot(
            x,
            empirical_targets,
            color="0.35",
            linestyle=":",
            marker="x",
            label="Empirical GT by condition",
        )
        axes[0, column].set_yscale("log")
        axes[1, column].set_yscale("log")
        axes[0, column].set_title(beta_mode.capitalize())
        axes[0, column].set_ylabel("Observed optimum hard density")
        axes[1, column].set_ylabel("Minimum hard latent error")
        axes[1, column].set_xticks(x, labels)
        for axis in axes[:, column]:
            axis.axvline(2.5, color="0.75", linewidth=1)
            axis.grid(alpha=0.2)
    axes[0, 1].legend(ncol=2, fontsize=8)
    fig.suptitle("Stage-1 amplitude × frequency ablation (last checkpoint)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_heatmaps(summary: pd.DataFrame, output_path: Path) -> None:
    last = summary[summary["checkpoint_kind"] == "last"].copy()
    columns = [
        (beta_mode, condition.condition_id)
        for beta_mode in BETA_MODES
        for condition in CONDITIONS
    ]
    column_labels = [
        f"{mode[0].upper()}:{label.replace(chr(10), ' ')}"
        for mode in BETA_MODES
        for label in _condition_labels()
    ]
    expected_log_ratios = np.empty((len(METHOD_ORDER), len(columns)))
    empirical_log_ratios = np.empty_like(expected_log_ratios)
    errors = np.empty_like(expected_log_ratios)
    for row_index, method in enumerate(METHOD_ORDER):
        for column_index, (beta_mode, condition_id) in enumerate(columns):
            row = last[
                (last["method"] == method)
                & (last["beta_mode"] == beta_mode)
                & (last["condition_id"] == condition_id)
            ].iloc[0]
            expected_log_ratios[row_index, column_index] = row[
                "optimum_density_log2_ratio_expected"
            ]
            empirical_log_ratios[row_index, column_index] = row[
                "optimum_density_log2_ratio_empirical"
            ]
            errors[row_index, column_index] = row[
                "minimum_hard_generalization_error"
            ]

    fig, axes = plt.subplots(3, 1, figsize=(18, 10.5), sharex=True)
    limit = max(
        1.0,
        float(
            np.nanmax(
                np.abs(np.concatenate([expected_log_ratios, empirical_log_ratios]))
            )
        ),
    )
    expected_image = axes[0].imshow(
        expected_log_ratios,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    empirical_image = axes[1].imshow(
        empirical_log_ratios,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    error_image = axes[2].imshow(errors, aspect="auto", cmap="viridis_r")
    for axis, values, formatter in (
        (axes[0], expected_log_ratios, lambda value: f"{value:+.2f}"),
        (axes[1], empirical_log_ratios, lambda value: f"{value:+.2f}"),
        (axes[2], errors, lambda value: f"{value:.3f}"),
    ):
        axis.set_yticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER])
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    formatter(values[row_index, column_index]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        axis.axvline(5.5, color="white", linewidth=2)
    axes[0].set_title("Primary: log2(optimum hard density / expected GT density)")
    axes[1].set_title("Sensitivity: log2(optimum hard density / empirical GT density)")
    axes[2].set_title("Minimum hard latent error")
    axes[2].set_xticks(np.arange(len(columns)), column_labels, rotation=35, ha="right")
    fig.colorbar(expected_image, ax=axes[0], shrink=0.75)
    fig.colorbar(empirical_image, ax=axes[1], shrink=0.75)
    fig.colorbar(error_image, ax=axes[2], shrink=0.75)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_factorial_curves(
    curves: dict[tuple[str, str, str], pd.DataFrame],
    beta_mode: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    arranged = (
        (CONDITIONS[0], CONDITIONS[1], CONDITIONS[2]),
        (CONDITIONS[3], CONDITIONS[4], CONDITIONS[5]),
    )
    for row_index, condition_row in enumerate(arranged):
        for column_index, condition in enumerate(condition_row):
            axis = axes[row_index, column_index]
            frame = curves[(condition.condition_id, beta_mode, "last")]
            _require_hard_axis(
                frame,
                source=f"{condition.condition_id}[{beta_mode},last]",
            )
            expected_target = float(
                frame["analysis_target_density_expected"].iloc[0]
            )
            empirical_target = float(
                frame["analysis_target_density_empirical"].iloc[0]
            )
            for method in METHOD_ORDER:
                method_rows = frame[(frame["method"] == method) & (frame["rho_model"] > 0)]
                method_rows = method_rows.sort_values(
                    ["rho_model", "hard_generalization_error", "control_value"],
                    kind="stable",
                )
                envelope = method_rows.drop_duplicates("rho_model", keep="first")
                axis.scatter(
                    method_rows["rho_model"],
                    method_rows["hard_generalization_error"],
                    s=7,
                    alpha=0.28,
                    color=METHOD_COLORS[method],
                )
                axis.plot(
                    envelope["rho_model"],
                    envelope["hard_generalization_error"],
                    marker="o",
                    markersize=2.5,
                    linewidth=1,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
            axis.axvline(
                expected_target,
                color="black",
                linestyle="--",
                linewidth=1,
                label="Expected GT",
            )
            axis.axvline(
                empirical_target,
                color="0.35",
                linestyle=":",
                linewidth=1,
                label="Empirical GT",
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(alpha=0.2)
            axis.set_title(condition.condition_id)
            if column_index == 0:
                axis.set_ylabel("Hard latent error")
            if row_index == 1:
                axis.set_xlabel("Hard density")
    axes[0, 2].legend(ncol=2, fontsize=8)
    fig.suptitle(f"Stage-1 hard recovery curves — {beta_mode}, last")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_factor_interactions(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot amplitude effects as paired skew/uniform-frequency interactions."""

    last = summary[summary["checkpoint_kind"] == "last"]
    amplitudes = ("exponential", "constant", "uniform")
    x = np.arange(len(amplitudes))
    frequency_styles = {
        0.5: ("tab:orange", "skew=0.5"),
        0.0: ("tab:blue", "uniform frequency"),
    }
    beta_styles = {"profiled": "-", "learned": "--"}
    fig, axes = plt.subplots(
        2,
        len(METHOD_ORDER),
        figsize=(20, 7.5),
        sharex=True,
        squeeze=False,
    )
    for column, method in enumerate(METHOD_ORDER):
        method_rows = last[last["method"] == method]
        for beta_mode in BETA_MODES:
            for frequency_skew, (color, _label) in frequency_styles.items():
                selected = (
                    method_rows[
                        (method_rows["beta_mode"] == beta_mode)
                        & np.isclose(
                            pd.to_numeric(
                                method_rows["frequency_skew"], errors="coerce"
                            ),
                            frequency_skew,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                    ]
                    .set_index("amplitude_mode")
                    .loc[list(amplitudes)]
                )
                style = beta_styles[beta_mode]
                axes[0, column].plot(
                    x,
                    selected["optimum_density_log2_ratio_expected"],
                    color=color,
                    linestyle=style,
                    marker="o",
                    linewidth=1.2,
                )
                axes[1, column].plot(
                    x,
                    selected["minimum_hard_generalization_error"],
                    color=color,
                    linestyle=style,
                    marker="o",
                    linewidth=1.2,
                )
        axes[0, column].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axes[0, column].set_title(METHOD_LABELS[method])
        axes[1, column].set_yscale("log")
        axes[1, column].set_xticks(x, ("Exp", "Constant", "Uniform"), rotation=20)
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("log2(optimum / expected GT)")
    axes[1, 0].set_ylabel("Minimum hard latent error")
    frequency_handles = [
        plt.Line2D([0], [0], color=color, label=label)
        for color, label in frequency_styles.values()
    ]
    beta_handles = [
        plt.Line2D([0], [0], color="black", linestyle=style, label=beta_mode)
        for beta_mode, style in beta_styles.items()
    ]
    fig.legend(
        handles=[*frequency_handles, *beta_handles],
        loc="outside lower center",
        ncols=4,
        fontsize=9,
    )
    fig.suptitle("Stage-1 factorial interactions (last checkpoint)")
    fig.set_layout_engine("constrained")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    output_rows: list[dict[str, object]] = []
    curves: dict[tuple[str, str, str], pd.DataFrame] = {}
    root_audits: list[dict[str, object]] = []
    for checkpoint_kind in CHECKPOINT_KINDS:
        for beta_mode in BETA_MODES:
            for condition in CONDITIONS:
                metrics, context, root, audit = load_condition(
                    condition, beta_mode, checkpoint_kind
                )
                root_audits.append(audit)
                curves[(condition.condition_id, beta_mode, checkpoint_kind)] = metrics
                output_rows.extend(
                    summarize_condition(
                        condition,
                        beta_mode,
                        checkpoint_kind,
                        metrics,
                        context,
                        root,
                    )
                )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(output_rows)
    expected_optimum_rows = (
        len(CONDITIONS)
        * len(BETA_MODES)
        * len(CHECKPOINT_KINDS)
        * len(METHOD_ORDER)
    )
    if len(summary) != expected_optimum_rows:
        raise ValueError(
            f"Expected {expected_optimum_rows} optimum rows, got {len(summary)}."
        )
    root_records = build_root_records(root_audits)
    contrasts = build_factorial_contrasts(summary)
    checkpoint_sensitivity = build_checkpoint_sensitivity(summary)
    summary.to_csv(OUTPUT_ROOT / "hard_generalization_optima.csv", index=False)
    root_records.to_csv(OUTPUT_ROOT / "condition_roots.csv", index=False)
    contrasts.to_csv(OUTPUT_ROOT / "factorial_contrasts.csv", index=False)
    checkpoint_sensitivity.to_csv(
        OUTPUT_ROOT / "checkpoint_sensitivity.csv", index=False
    )
    plot_optima(summary, OUTPUT_ROOT / "hard_generalization_optima_last.png")
    plot_alignment_heatmaps(
        summary, OUTPUT_ROOT / "hard_generalization_alignment_last.png"
    )
    for beta_mode in BETA_MODES:
        plot_factorial_curves(
            curves,
            beta_mode,
            OUTPUT_ROOT / f"hard_generalization_curves_last_{beta_mode}.png",
        )
    plot_factor_interactions(
        summary, OUTPUT_ROOT / "factorial_interactions_last.png"
    )
    provenance = {
        column: sorted(set(root_records[column].astype(str)))
        for column in (
            "train_source_fingerprint",
            "train_pipeline_fingerprint",
            "train_git_revisions",
            "eval_source_fingerprint",
            "eval_pipeline_fingerprint",
            "eval_git_revision",
        )
    }
    (OUTPUT_ROOT / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "density_mode": "hard",
                "primary_target": "expected",
                "sensitivity_target": "empirical",
                "primary_checkpoint": "last",
                "sensitivity_checkpoint": "best",
                "n_optimum_rows": len(summary),
                "n_factorial_contrasts": len(contrasts),
                "n_checkpoint_sensitivity_rows": len(checkpoint_sensitivity),
                "n_roots": len(root_records),
                "conditions": [asdict(condition) for condition in CONDITIONS],
                "beta_modes": list(BETA_MODES),
                "methods": list(METHOD_ORDER),
                "nearest_target_tie_policy": (
                    "minimum absolute hard-density distance, then minimum hard "
                    "generalization error, lower density, control, run_id"
                ),
                "optimum_tie_policy": (
                    "best metric, then lower hard density, control, run_id"
                ),
                "cross_root_fingerprint_equality_required": False,
                "cross_root_fingerprint_note": (
                    "Fingerprints are validated within each root and across its "
                    "last/best evaluations, then recorded here. Legacy control and "
                    "new ablation roots may legitimately have different source "
                    "fingerprints."
                ),
                "provenance": provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Analysis artifacts: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
