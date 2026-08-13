from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.analyze_CustomData_ablation_matrix import (
    BETA_MODES,
    CONDITIONS,
    Condition,
    apply_density_axis,
    build_checkpoint_sensitivity,
    build_factorial_contrasts,
    condition_root,
    plot_factorial_curves,
    summarize_condition,
)
from src.sae_sweep import METHOD_ORDER


def test_conditions_and_roots_cover_the_three_by_two_factorial() -> None:
    expected = {
        ("exponential", 0.5): (
            "exp_skew05",
            None,
            "stage1_beta_{beta}_din128_gt1024_sae1024_sd001_seed0",
        ),
        ("constant", 0.5): (
            "constant_skew05",
            "ablation2_constant",
            "stage1_ablation2_constant_beta_{beta}"
            "_din128_gt1024_sae1024_sd001_seed0",
        ),
        ("uniform", 0.5): (
            "uniform_skew05",
            "ablation2_uniform",
            "stage1_ablation2_uniform_beta_{beta}"
            "_din128_gt1024_sae1024_sd001_seed0",
        ),
        ("exponential", 0.0): (
            "exp_uniformfreq",
            "ablation3_uniformfreq",
            "stage1_ablation3_uniformfreq_beta_{beta}"
            "_din128_gt1024_sae1024_sd001_seed0",
        ),
        ("constant", 0.0): (
            "constant_uniformfreq",
            "ablation23_constant_uniformfreq",
            "stage1_ablation23_constant_uniformfreq_beta_{beta}"
            "_din128_gt1024_sae1024_sd001_seed0",
        ),
        ("uniform", 0.0): (
            "uniform_uniformfreq",
            "ablation23_uniform_uniformfreq",
            "stage1_ablation23_uniform_uniformfreq_beta_{beta}"
            "_din128_gt1024_sae1024_sd001_seed0",
        ),
    }

    observed = {
        (condition.amplitude_mode, condition.frequency_skew): (
            condition.condition_id,
            condition.root_token,
        )
        for condition in CONDITIONS
    }
    assert observed == {
        axes: condition_and_root[:2]
        for axes, condition_and_root in expected.items()
    }
    assert set(BETA_MODES) == {"profiled", "learned"}

    for beta_mode in BETA_MODES:
        roots = {
            condition_root(condition, beta_mode).name for condition in CONDITIONS
        }
        assert roots == {
            root_template.format(beta=beta_mode)
            for _, _, root_template in expected.values()
        }
        assert len(roots) == 6


def _synthetic_metrics() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for method_index, method in enumerate(METHOD_ORDER):
        if method == "vgsae":
            hard_l0 = (2.0, 5.0, 8.0)
            errors = (1.0, 2.0, 3.0)
        else:
            hard_l0 = (2.0, 8.0, 20.0)
            errors = (3.0, 1.0, 2.0)
        for control_index, (l0, error) in enumerate(zip(hard_l0, errors)):
            rows.append(
                {
                    "method": method,
                    "control_name": "sparsity_control",
                    "control_value": float(control_index),
                    # Deliberately unrelated reported/soft densities: the
                    # summary must use average_l0 / sae_width instead.
                    "rho_model": 0.7 - 0.1 * control_index,
                    "average_l0": l0,
                    "hard_generalization_error": error,
                    "decoder_recovery_cosine": (
                        0.1 * control_index + 0.01 * method_index
                    ),
                    "train_source_fingerprint": "train-source",
                    "eval_source_fingerprint": "eval-source",
                }
            )
    return pd.DataFrame(rows)


def test_optimum_summary_uses_hard_density_and_computes_target_diagnostics() -> None:
    raw_metrics = _synthetic_metrics()
    metrics = apply_density_axis(
        raw_metrics,
        sae_width=1_000,
        density_mode="hard",
    )
    np.testing.assert_allclose(metrics["rho_model_reported"], raw_metrics["rho_model"])
    np.testing.assert_allclose(metrics["rho_model"], metrics["average_l0"] / 1_000)
    assert set(metrics["density_axis"]) == {"hard"}
    condition = Condition("synthetic", "exponential", 0.5, None)
    rows = summarize_condition(
        condition,
        "profiled",
        "last",
        metrics,
        {
            "target_model_density_expected": 0.01,
            "target_model_density_empirical": 0.006,
        },
        Path("synthetic-root"),
    )
    summary = pd.DataFrame(rows).set_index("method")

    assert len(summary) == len(METHOD_ORDER)
    vg = summary.loc["vgsae"]
    assert vg["optimum_hard_density"] == pytest.approx(0.002)
    assert vg["optimum_hard_density"] != pytest.approx(0.7)
    assert bool(vg["target_expected_bracketed"]) is False
    assert bool(vg["target_empirical_bracketed"]) is True
    assert bool(vg["target_bracketed"]) is True
    assert bool(vg["optimum_is_density_boundary"]) is True
    assert bool(vg["optimum_is_control_boundary"]) is True
    assert vg["optimum_density_signed_gap_expected"] == pytest.approx(-0.008)
    assert vg["optimum_density_ratio_to_expected_target"] == pytest.approx(0.2)
    assert vg["optimum_density_log2_ratio_expected"] == pytest.approx(
        math.log2(0.2)
    )
    assert vg["nearest_expected_target_hard_density"] == pytest.approx(0.008)
    assert vg["expected_target_regret_absolute"] == pytest.approx(2.0)
    assert vg["expected_target_regret_relative"] == pytest.approx(2.0)

    assert vg["optimum_density_signed_gap_empirical"] == pytest.approx(-0.004)
    assert vg["optimum_density_ratio_to_empirical_target"] == pytest.approx(1 / 3)
    assert vg["optimum_density_log2_ratio_empirical"] == pytest.approx(
        math.log2(1 / 3)
    )
    assert vg["nearest_empirical_target_hard_density"] == pytest.approx(0.005)
    assert vg["nearest_empirical_target_density_signed_gap"] == pytest.approx(
        -0.001
    )
    assert vg["nearest_empirical_target_density_ratio"] == pytest.approx(5 / 6)
    assert vg["empirical_target_regret_absolute"] == pytest.approx(1.0)
    assert vg["empirical_target_regret_relative"] == pytest.approx(1.0)
    # The legacy unqualified aliases deliberately retain empirical semantics.
    assert vg["nearest_target_hard_density"] == pytest.approx(0.005)
    assert vg["target_regret_absolute"] == pytest.approx(1.0)

    assert vg["cosine_optimum_hard_density"] == pytest.approx(0.008)
    assert vg["cosine_optimum_density_ratio_to_expected_target"] == pytest.approx(
        0.8
    )
    assert vg["cosine_optimum_density_ratio_to_empirical_target"] == pytest.approx(
        4 / 3
    )
    assert vg["decoder_recovery_cosine_at_error_optimum"] == pytest.approx(0.0)
    assert vg["hard_generalization_error_at_cosine_optimum"] == pytest.approx(3.0)

    l1 = summary.loc["l1"]
    assert l1["optimum_hard_density"] == pytest.approx(0.008)
    assert bool(l1["target_expected_bracketed"]) is True
    assert bool(l1["target_empirical_bracketed"]) is True
    assert bool(l1["optimum_is_density_boundary"]) is False
    assert l1["optimum_density_signed_gap_expected"] == pytest.approx(-0.002)
    assert l1["nearest_expected_target_hard_density"] == pytest.approx(0.008)
    assert l1["expected_target_regret_absolute"] == pytest.approx(0.0)
    assert l1["nearest_empirical_target_hard_density"] == pytest.approx(0.008)
    assert l1["empirical_target_regret_absolute"] == pytest.approx(0.0)


def test_summary_rejects_untransformed_reported_density() -> None:
    with pytest.raises(ValueError, match="density_axis='hard'"):
        summarize_condition(
            Condition("synthetic", "exponential", 0.5, None),
            "profiled",
            "last",
            _synthetic_metrics(),
            {
                "target_model_density_expected": 0.01,
                "target_model_density_empirical": 0.01,
            },
            Path("synthetic-root"),
        )


def test_factorial_curve_plot_rejects_untransformed_reported_density(
    tmp_path: Path,
) -> None:
    curves = {
        (CONDITIONS[0].condition_id, "profiled", "last"): _synthetic_metrics()
    }
    try:
        with pytest.raises(ValueError, match="density_axis='hard'"):
            plot_factorial_curves(
                curves,
                "profiled",
                tmp_path / "must-not-be-written.png",
            )
    finally:
        plt.close("all")
    assert not (tmp_path / "must-not-be-written.png").exists()


def _duplicate_nonmonotonic_metrics() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for method_index, method in enumerate(METHOD_ORDER):
        for control, l0, error, cosine in (
            (0.0, 6.0, 0.301, 0.1),
            (1.0, 2.0, 0.8, 0.4),
            (2.0, 12.0, 0.314, 0.3),
            (3.0, 6.0, 0.3, 0.2),
        ):
            rows.append(
                {
                    "run_id": f"{method}-{control:g}",
                    "method": method,
                    "control_name": "sparsity_control",
                    "control_value": control,
                    "rho_model": 0.5,
                    "average_l0": l0,
                    "hard_generalization_error": error + 0.01 * method_index,
                    "decoder_recovery_cosine": cosine,
                    "train_source_fingerprint": "train-source",
                    "train_pipeline_fingerprint": "train-pipeline",
                    "eval_source_fingerprint": "eval-source",
                }
            )
    return pd.DataFrame(rows)


def test_duplicate_density_tie_nonmonotonicity_and_control_boundary() -> None:
    metrics = apply_density_axis(
        _duplicate_nonmonotonic_metrics(), sae_width=1_000, density_mode="hard"
    )
    rows = summarize_condition(
        Condition("synthetic", "exponential", 0.5, None),
        "profiled",
        "last",
        metrics,
        {
            "target_model_density_expected": 0.006,
            "target_model_density_empirical": 0.006,
        },
        Path("synthetic-root"),
    )
    vg = pd.DataFrame(rows).set_index("method").loc["vgsae"]

    # Density 0.006 is reached by controls 0 and 3. The deterministic policy
    # chooses the lower-error control 3, which is a control boundary even
    # though 0.006 is not a density boundary.
    assert vg["nearest_expected_target_hard_density"] == pytest.approx(0.006)
    assert vg["nearest_expected_target_hard_generalization_error"] == pytest.approx(
        0.3
    )
    assert vg["control_value_at_error_optimum"] == pytest.approx(3.0)
    assert bool(vg["optimum_is_control_boundary"]) is True
    assert bool(vg["optimum_is_density_boundary"]) is False
    assert bool(vg["hard_density_monotonic_with_control"]) is False
    assert vg["hard_density_control_monotonicity"] == "nonmonotonic"
    assert vg["duplicate_hard_density_groups"] == 1
    assert vg["duplicate_hard_density_excess_rows"] == 1

    assert vg["near_optimal_1pct_control_count"] == 2
    assert vg["near_optimal_1pct_density_count"] == 1
    assert vg["near_optimal_1pct_density_min"] == pytest.approx(0.006)
    assert vg["near_optimal_1pct_density_max"] == pytest.approx(0.006)
    assert vg["near_optimal_5pct_control_count"] == 3
    assert vg["near_optimal_5pct_density_count"] == 2
    assert vg["near_optimal_5pct_density_min"] == pytest.approx(0.006)
    assert vg["near_optimal_5pct_density_max"] == pytest.approx(0.012)


def _factorial_summary(checkpoints: tuple[str, ...] = ("last",)) -> pd.DataFrame:
    density = {
        ("exponential", 0.5): 0.004,
        ("constant", 0.5): 0.008,
        ("uniform", 0.5): 0.006,
        ("exponential", 0.0): 0.010,
        ("constant", 0.0): 0.012,
        ("uniform", 0.0): 0.009,
    }
    error = {
        ("exponential", 0.5): 1.0,
        ("constant", 0.5): 1.1,
        ("uniform", 0.5): 0.9,
        ("exponential", 0.0): 0.8,
        ("constant", 0.0): 0.7,
        ("uniform", 0.0): 0.6,
    }
    records: list[dict[str, float | str]] = []
    for checkpoint_kind in checkpoints:
        checkpoint_multiplier = 1.0 if checkpoint_kind == "last" else 0.5
        for condition in CONDITIONS:
            axes = (condition.amplitude_mode, condition.frequency_skew)
            optimum_density = density[axes] * checkpoint_multiplier
            minimum_error = error[axes] * checkpoint_multiplier
            records.append(
                {
                    "condition_id": condition.condition_id,
                    "amplitude_mode": condition.amplitude_mode,
                    "frequency_skew": condition.frequency_skew,
                    "beta_mode": "profiled",
                    "checkpoint_kind": checkpoint_kind,
                    "sweep_root": f"root-{condition.condition_id}",
                    "method": "vgsae",
                    "method_label": "VG-SAE",
                    "control_value_at_error_optimum": 1.0,
                    "optimum_hard_density": optimum_density,
                    "minimum_hard_generalization_error": minimum_error,
                    "optimum_density_log2_ratio_expected": math.log2(
                        optimum_density / 0.01
                    ),
                    "cosine_optimum_hard_density": optimum_density * 1.25,
                    "maximum_decoder_recovery_cosine": 0.5 + optimum_density,
                }
            )
    return pd.DataFrame(records)


def test_factorial_contrasts_encode_amplitude_and_frequency_simple_effects() -> None:
    contrasts = build_factorial_contrasts(_factorial_summary())
    assert len(contrasts) == 7
    assert (contrasts["factor"] == "amplitude_mode").sum() == 4
    assert (contrasts["factor"] == "frequency_skew").sum() == 3

    constant_at_skew = contrasts[
        (contrasts["factor"] == "amplitude_mode")
        & (contrasts["level_to"] == "constant")
        & (contrasts["held_constant_level"] == "skew_0.5")
    ].iloc[0]
    assert constant_at_skew["optimum_hard_density_ratio"] == pytest.approx(2.0)
    assert constant_at_skew["optimum_hard_density_log2_ratio"] == pytest.approx(1.0)
    assert constant_at_skew[
        "minimum_hard_generalization_error_delta"
    ] == pytest.approx(0.1)

    exponential_frequency = contrasts[
        (contrasts["factor"] == "frequency_skew")
        & (contrasts["held_constant_level"] == "exponential")
    ].iloc[0]
    assert exponential_frequency["optimum_hard_density_ratio"] == pytest.approx(
        2.5
    )
    assert exponential_frequency[
        "minimum_hard_generalization_error_relative_delta"
    ] == pytest.approx(-0.2)


def test_checkpoint_sensitivity_pairs_last_and_best() -> None:
    sensitivity = build_checkpoint_sensitivity(
        _factorial_summary(("last", "best"))
    )
    assert len(sensitivity) == len(CONDITIONS)
    assert set(
        sensitivity["optimum_hard_density_log2_ratio_best_over_last"]
    ) == {-1.0}
    assert set(
        sensitivity["minimum_hard_generalization_error_ratio_best_over_last"]
    ) == {0.5}
