from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.sae_sweep_plot import (
    _mask_representatives,
    load_sweep_plot_context,
    plot_data_overview,
    plot_sparsity_diagnostics,
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
