from __future__ import annotations

import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pytest

from src.sae_sweep import METHOD_LABELS, METHOD_ORDER
from src.sae_sweep_plot import (
    _mask_representatives,
    load_sweep_plot_context,
    plot_data_overview,
    plot_reconstruction_metrics,
    plot_recovery_metrics,
    plot_sparsity_diagnostics,
    plot_support_metrics,
    plot_training_curves,
    plot_vg_posterior_diagnostics,
)


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
