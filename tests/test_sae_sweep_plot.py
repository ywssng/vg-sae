from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pytest

from src.sae_sweep import METHOD_LABELS, METHOD_ORDER
from src.sae_sweep_plot import (
    _mask_representatives,
    aggregate_seed_metrics,
    load_sweep_plot_context,
    load_comparison_results,
    plot_data_overview,
    plot_reconstruction_metrics,
    plot_recovery_metrics,
    plot_sparsity_diagnostics,
    plot_support_metrics,
    plot_training_curves,
    plot_vg_posterior_diagnostics,
    plot_mask_heatmaps,
)


def test_comparison_loader_fills_only_absent_methods_from_baseline_root(
    tmp_path,
) -> None:
    vg_root = tmp_path / "vg"
    baseline_root = tmp_path / "baseline"
    for root in (vg_root, baseline_root):
        (root / "summary" / "last").mkdir(parents=True)
        (root / "summary").mkdir(exist_ok=True)
        (root / "sweep_config.json").write_text(
            json.dumps(
                {
                    "data": {
                        "kind": "synthsaebench_pretrained",
                        "model_id": "benchmark",
                        "revision": "revision",
                        "model_config_sha256": "sha",
                        "input_dim": 3,
                        "ground_truth_num_features": 6,
                        "sae_width": 5,
                        "n_test": 4,
                    }
                }
            )
        )
    pd.DataFrame(
        [{"method": "vgsae", "run_id": "vg-new", "beta_mode": "learned"}]
    ).to_csv(vg_root / "summary" / "last" / "final_metrics.csv", index=False)
    pd.DataFrame([{"method": "vgsae", "run_id": "vg-new", "step": 0}]).to_csv(
        vg_root / "summary" / "training_curves.csv", index=False
    )
    pd.DataFrame(
        [
            {"method": "vgsae", "run_id": "vg-old"},
            {"method": "topk", "run_id": "topk-old"},
        ]
    ).to_csv(
        baseline_root / "summary" / "last" / "final_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {"method": "vgsae", "run_id": "vg-old", "step": 0},
            {"method": "topk", "run_id": "topk-old", "step": 0},
        ]
    ).to_csv(baseline_root / "summary" / "training_curves.csv", index=False)

    metrics, history, roots = load_comparison_results(
        vg_root, baseline_sweep_dir=baseline_root
    )

    assert set(metrics["run_id"]) == {"vg-new", "topk-old"}
    assert set(history["run_id"]) == {"vg-new", "topk-old"}
    assert metrics.loc[
        metrics["method"] == "topk", "beta_mode"
    ].item() == "learned"
    assert roots[("vgsae", "vg-new")] == vg_root
    assert roots[("topk", "topk-old")] == baseline_root


def test_comparison_loader_labels_invariant_baselines_with_primary_mode(
    tmp_path: Path,
) -> None:
    vg_root = tmp_path / "vg"
    baseline_root = tmp_path / "baseline"
    for root in (vg_root, baseline_root):
        (root / "summary" / "last").mkdir(parents=True)
        (root / "summary").mkdir(exist_ok=True)
    data = {"kind": "same", "input_dim": 4}
    (vg_root / "sweep_config.json").write_text(
        json.dumps({"data": data, "training": {"beta_mode": "learned"}})
    )
    (baseline_root / "sweep_config.json").write_text(json.dumps({"data": data}))
    pd.DataFrame(
        [{"method": "vgsae", "run_id": "vg-new", "beta_mode": "learned"}]
    ).to_csv(vg_root / "summary" / "last" / "final_metrics.csv", index=False)
    pd.DataFrame(
        [{"method": "vgsae", "run_id": "vg-new", "beta_mode": "learned"}]
    ).to_csv(vg_root / "summary" / "training_curves.csv", index=False)
    pd.DataFrame([{"method": "topk", "run_id": "topk-old"}]).to_csv(
        baseline_root / "summary" / "last" / "final_metrics.csv", index=False
    )
    pd.DataFrame([{"method": "topk", "run_id": "topk-old"}]).to_csv(
        baseline_root / "summary" / "training_curves.csv", index=False
    )

    metrics, history, _ = load_comparison_results(
        vg_root, baseline_sweep_dir=baseline_root
    )

    assert set(metrics["beta_mode"]) == {"learned"}
    assert set(history["beta_mode"]) == {"learned"}


def test_comparison_loader_rejects_different_data_conditions(tmp_path) -> None:
    primary = tmp_path / "primary"
    baseline = tmp_path / "baseline"
    for root, n_test in ((primary, 4), (baseline, 8)):
        (root / "summary" / "last").mkdir(parents=True)
        (root / "summary").mkdir(exist_ok=True)
        pd.DataFrame([{"method": "vgsae", "run_id": root.name}]).to_csv(
            root / "summary" / "last" / "final_metrics.csv", index=False
        )
        pd.DataFrame([{"method": "vgsae", "run_id": root.name}]).to_csv(
            root / "summary" / "training_curves.csv", index=False
        )
        (root / "sweep_config.json").write_text(
            json.dumps({"data": {"n_test": n_test}})
        )

    with pytest.raises(ValueError, match="different data/evaluation.*n_test"):
        load_comparison_results(primary, baseline_sweep_dir=baseline)


def test_legacy_primary_mode_survives_baseline_merge_and_aggregation(
    tmp_path,
) -> None:
    primary = tmp_path / "primary"
    baseline = tmp_path / "baseline"
    data = {
        "kind": "synthsaebench_pretrained",
        "model_id": "benchmark",
        "revision": "revision",
        "model_config_sha256": "sha",
        "input_dim": 3,
        "ground_truth_num_features": 6,
        "sae_width": 5,
        "n_test": 4,
    }
    for root, method in ((primary, "vgsae"), (baseline, "topk")):
        (root / "summary" / "last").mkdir(parents=True)
        (root / "summary").mkdir(exist_ok=True)
        (root / "sweep_config.json").write_text(json.dumps({"data": data}))
        pd.DataFrame(
            [
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "run_id": f"{method}-legacy",
                    "control_name": "control",
                    "control_value": 1.0,
                    "seed": 0,
                    "rho_model": 0.1,
                }
            ]
        ).to_csv(root / "summary" / "last" / "final_metrics.csv", index=False)
        pd.DataFrame(
            [{"method": method, "run_id": f"{method}-legacy", "step": 0}]
        ).to_csv(root / "summary" / "training_curves.csv", index=False)

    metrics, _, _ = load_comparison_results(
        primary, baseline_sweep_dir=baseline
    )
    table = aggregate_seed_metrics(metrics)

    assert set(table["method"]) == {"vgsae", "topk"}
    assert metrics.loc[
        metrics["method"] == "vgsae", "beta_mode"
    ].item() == "profiled"


def test_baseline_vg_never_gets_invariant_mode_label(tmp_path) -> None:
    primary = tmp_path / "primary"
    baseline = tmp_path / "baseline"
    data = {"kind": "same", "n_test": 4}
    for root in (primary, baseline):
        (root / "summary" / "last").mkdir(parents=True)
        (root / "summary").mkdir(exist_ok=True)
        (root / "sweep_config.json").write_text(json.dumps({"data": data}))
    pd.DataFrame([{"method": "topk", "run_id": "topk-new"}]).to_csv(
        primary / "summary" / "last" / "final_metrics.csv", index=False
    )
    pd.DataFrame([{"method": "topk", "run_id": "topk-new"}]).to_csv(
        primary / "summary" / "training_curves.csv", index=False
    )
    pd.DataFrame(
        [{"method": "vgsae", "run_id": "vg-old", "beta_mode": "profiled"}]
    ).to_csv(baseline / "summary" / "last" / "final_metrics.csv", index=False)
    pd.DataFrame(
        [{"method": "vgsae", "run_id": "vg-old", "beta_mode": "profiled"}]
    ).to_csv(baseline / "summary" / "training_curves.csv", index=False)

    metrics, _, _ = load_comparison_results(
        primary, baseline_sweep_dir=baseline
    )

    assert metrics.loc[
        metrics["method"] == "vgsae", "beta_mode"
    ].item() == "profiled"


def _all_method_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "control_name": "control",
                "control_value": float(index),
                "seed": 0,
                "rho_model": 0.05 * (index + 1),
                "explained_variance": 0.8,
                "reconstruction_error": 0.2,
                "generalization_error": 0.1,
                "decoder_recovery_cosine": 0.7,
                "support_f1": 0.4,
                "support_average_precision": 0.5,
                "support_precision": 0.6,
                "support_recall": 0.3,
                "selection_error": 0.2,
                "dead_fraction": 0.1,
                "average_l0": 2.0,
                "expected_l0": 1.5,
            }
            for index, method in enumerate(METHOD_ORDER)
        ]
    )


def _all_method_synth_metrics() -> pd.DataFrame:
    metrics = _all_method_metrics()
    metrics["benchmark_model_id"] = "decoderesearch/synth-sae-bench-16k-v1"
    metrics["mcc"] = 0.8
    metrics["uniqueness"] = 0.7
    metrics["classification_f1"] = 0.9
    metrics["classification_precision"] = 0.91
    metrics["classification_recall"] = 0.89
    metrics["classification_accuracy"] = 0.95
    metrics["sae_l0"] = 32.0
    metrics["true_l0"] = 35.0
    metrics["shrinkage"] = 0.85
    metrics["vg_expected_l0"] = metrics["sae_l0"]
    metrics["vg_expected_explained_variance"] = metrics["explained_variance"]
    metrics.loc[metrics["method"] == "vgsae", "vg_expected_l0"] = 1_500.0
    metrics.loc[
        metrics["method"] == "vgsae", "vg_expected_explained_variance"
    ] = 0.95
    return metrics


def _assert_single_bottom_legend(figure) -> None:
    expected = [METHOD_LABELS[method] for method in METHOD_ORDER]
    assert all(axis.get_legend() is None for axis in figure.axes)
    assert len(figure.legends) == 1
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == expected
    assert legend._ncols == len(expected)
    assert legend.get_frame_on()
    assert all(handle.get_marker() == "o" for handle in legend.legend_handles)
    figure.canvas.draw()
    legend_box = legend.get_window_extent()
    assert figure.bbox.contains(legend_box.x0, legend_box.y0)
    assert figure.bbox.contains(legend_box.x1, legend_box.y1)
    text_rows = {
        round(text.get_window_extent().y0, 3) for text in legend.get_texts()
    }
    assert len(text_rows) == 1
    assert all(
        not legend_box.overlaps(axis.get_window_extent()) for axis in figure.axes
    )


def test_plot_context_separates_ground_truth_and_sae_width(tmp_path) -> None:
    (tmp_path / "summary").mkdir()
    (tmp_path / "sweep_config.json").write_text(
        json.dumps(
            {
                "data": {
                    "ground_truth_num_features": 6,
                    "sae_width": 4,
                    "support_density": 0.2,
                }
            }
        )
    )
    np.savez_compressed(
        tmp_path / "summary" / "data_preview.npz",
        feature_probabilities=np.full(6, 0.2),
        dictionary=np.eye(2, 6),
        z0=np.zeros(6),
    )

    context = load_sweep_plot_context(tmp_path)
    assert context["ground_truth_num_features"] == 6
    assert context["sae_width"] == 4
    assert context["expected_true_l0"] == pytest.approx(1.2)
    assert context["target_model_density"] == pytest.approx(0.3)

    (tmp_path / "sweep_config.json").write_text(
        json.dumps({"data": {"n_features": 6, "support_density": 0.2}})
    )
    legacy_context = load_sweep_plot_context(tmp_path)
    assert legacy_context["ground_truth_num_features"] == 6
    assert legacy_context["sae_width"] == 6
    assert legacy_context["target_model_density"] == pytest.approx(0.2)

    figure = plot_data_overview(tmp_path)
    assert "pairwise cosine" in figure.axes[1].get_title()
    plt.close(figure)


def test_synth_plot_context_uses_empirical_true_l0_and_preview_dictionary(
    tmp_path,
) -> None:
    (tmp_path / "summary").mkdir()
    (tmp_path / "sweep_config.json").write_text(
        json.dumps(
            {
                "data": {
                    "kind": "synthsaebench_pretrained",
                    "ground_truth_num_features": 16_384,
                    "sae_width": 4_096,
                }
            }
        )
    )
    np.savez_compressed(
        tmp_path / "summary" / "data_preview.npz",
        feature_probabilities=np.full(16, 0.01),
        dictionary=np.eye(4, 8),
        z0=np.zeros(8),
        empirical_true_l0=np.asarray(34.5),
        target_model_density=np.asarray(34.5 / 4_096),
        data_kind=np.asarray("synthsaebench_pretrained"),
    )

    context = load_sweep_plot_context(tmp_path)
    assert context["expected_true_l0"] == pytest.approx(34.5)
    assert context["target_model_density"] == pytest.approx(34.5 / 4_096)
    assert context["ground_truth_num_features"] == 16_384
    assert context["sae_width"] == 4_096

    figure = plot_data_overview(tmp_path)
    assert "Pretrained dictionary preview" in figure.axes[1].get_title()
    plt.close(figure)


def test_l0_diagnostics_are_normalized_by_sae_width() -> None:
    metrics = pd.DataFrame(
        [
            {
                "method": "vgsae",
                "method_label": "VG-SAE",
                "control_name": "gamma",
                "control_value": 1.0,
                "seed": 0,
                "rho_model": 0.25,
                "selection_error": 0.1,
                "dead_fraction": 0.2,
                "average_l0": 2.0,
                "expected_l0": 1.0,
            }
        ]
    )

    figure = plot_sparsity_diagnostics(
        metrics, target_model_density=0.3, sae_width=4
    )
    assert figure.axes[2].lines[0].get_ydata()[0] == pytest.approx(0.5)
    assert figure.axes[3].lines[0].get_ydata()[0] == pytest.approx(0.25)
    assert figure.axes[0].lines[1].get_xdata()[0] == pytest.approx(0.3)
    plt.close(figure)


@pytest.mark.parametrize(
    "plotter",
    [
        plot_reconstruction_metrics,
        plot_recovery_metrics,
        plot_support_metrics,
        plot_sparsity_diagnostics,
    ],
)
def test_metric_figures_use_one_bottom_legend(plotter) -> None:
    figure = plotter(
        _all_method_metrics(), target_model_density=0.1, sae_width=4
    )
    _assert_single_bottom_legend(figure)
    plt.close(figure)


@pytest.mark.parametrize(
    ("plotter", "expected_labels"),
    [
        (plot_reconstruction_metrics, [r"$R^2$", "Shrinkage"]),
        (plot_recovery_metrics, ["MCC", "Uniqueness"]),
        (
            plot_support_metrics,
            ["F1", "Precision", "Recall", "Accuracy"],
        ),
        (
            plot_sparsity_diagnostics,
            [
                r"SAE L0 / $d_\mathrm{sae}$",
                r"True L0 / $d_\mathrm{sae}$",
                "Dead latent fraction",
                r"Expected L0 / $d_\mathrm{sae}$",
            ],
        ),
    ],
)
def test_synth_metric_figures_use_official_panels(plotter, expected_labels) -> None:
    figure = plotter(
        _all_method_synth_metrics(),
        target_model_density=35.0 / 4_096,
        sae_width=4_096,
    )

    assert [axis.get_ylabel() for axis in figure.axes] == expected_labels
    if plotter is plot_support_metrics:
        assert all(axis.get_ylim()[1] == pytest.approx(1.02) for axis in figure.axes)
    assert all(
        isinstance(axis.xaxis.get_minor_formatter(), ticker.NullFormatter)
        for axis in figure.axes
    )
    _assert_single_bottom_legend(figure)
    plt.close(figure)


def test_synth_sparsity_identity_line_stays_within_swept_density_band() -> None:
    metrics = _all_method_synth_metrics()
    target_density = 35.0 / 4_096
    figure = plot_sparsity_diagnostics(
        metrics,
        target_model_density=target_density,
        sae_width=4_096,
    )

    identity_x = np.asarray(figure.axes[0].lines[-1].get_xdata(), dtype=float)
    expected_min = min(target_density, float(metrics["rho_model"].min()))
    expected_max = max(target_density, float(metrics["rho_model"].max()))
    assert identity_x.min() == pytest.approx(expected_min)
    assert identity_x.max() == pytest.approx(expected_max)
    assert figure.axes[0].get_xlim()[1] < 0.5
    plt.close(figure)


def test_vg_posterior_figure_separates_hard_and_expected_paths() -> None:
    figure = plot_vg_posterior_diagnostics(
        _all_method_synth_metrics(),
        target_model_density=35.0 / 4_096,
        sae_width=4_096,
    )

    assert [line.get_label() for line in figure.axes[0].lines[:2]] == [
        "Hard inference",
        "Posterior expectation",
    ]
    assert figure.axes[1].get_yscale() == "log"
    plt.close(figure)


def test_training_curves_use_one_bottom_legend() -> None:
    history = pd.DataFrame(
        [
            {
                "method": method,
                "run_id": f"{method}-0",
                "step": 0,
                "loss": 1.0,
                "reconstruction_mse": 0.5,
                "rho": 0.1,
            }
            for method in METHOD_ORDER
        ]
    )
    figure = plot_training_curves(history)
    _assert_single_bottom_legend(figure)
    plt.close(figure)


def test_mask_representatives_use_one_common_seed() -> None:
    metrics = pd.DataFrame(
        [
            {"method": "vgsae", "seed": 0, "rho_model": 0.1},
            {"method": "vgsae", "seed": 1, "rho_model": 0.3},
            {"method": "l1", "seed": 0, "rho_model": 0.4},
            {"method": "l1", "seed": 1, "rho_model": 0.2},
        ]
    )

    representatives = _mask_representatives(metrics, 0.25, None)

    assert representatives["seed"].tolist() == [0, 0]


def test_mask_heatmaps_can_resolve_rows_from_multiple_sweep_roots(tmp_path) -> None:
    primary = tmp_path / "primary"
    corrective = tmp_path / "corrective"
    metrics = pd.DataFrame(
        [
            {
                "method": "vgsae",
                "method_label": "VG-SAE",
                "run_id": "vg-primary",
                "seed": 0,
                "rho_model": 0.1,
                "selection_error": 0.2,
                "ground_truth_num_features": 2,
                "matching_policy": "per_latent_best",
            },
            {
                "method": "l1",
                "method_label": "L1-ReLU",
                "run_id": "l1-corrective",
                "seed": 0,
                "rho_model": 0.1,
                "selection_error": 0.2,
                "ground_truth_num_features": 2,
                "matching_policy": "per_latent_best",
            },
        ]
    )
    for root, method, run_id in (
        (primary, "vgsae", "vg-primary"),
        (corrective, "l1", "l1-corrective"),
    ):
        destination = root / "runs" / method / run_id / "eval" / "last"
        destination.mkdir(parents=True)
        np.savez_compressed(
            destination / "cache.npz",
            true_support=np.zeros((2, 2)),
            mask=np.zeros((2, 2)),
        )

    figure, representatives = plot_mask_heatmaps(
        primary,
        metrics,
        target_model_density=0.1,
        run_roots={"l1-corrective": corrective},
    )

    assert representatives["run_id"].tolist() == ["vg-primary", "l1-corrective"]
    plt.close(figure)
