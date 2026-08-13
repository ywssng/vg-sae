from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.real_activation_plot import plot_all


def _write_metrics(root: Path, rows: list[dict[str, object]]) -> None:
    destination = root / "summary" / "last"
    destination.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(destination / "final_metrics.csv", index=False)


def test_plot_all_writes_stage3_panels_with_partial_method_rows(tmp_path: Path) -> None:
    _write_metrics(
        tmp_path,
        [
            {
                "model_name": "google/gemma-2-2b",
                "layer": 5,
                "hook_name": "blocks.5.hook_resid_post",
                "method": "vgsae",
                "method_label": "VG-SAE",
                "control_name": "lambda",
                "control_value": 0.5,
                "average_l0": 42.0,
                "sae_width": 32768,
                "rho_model": 42.0 / 32768,
                "dead_fraction": 0.2,
                "explained_variance": 0.81,
                "reconstruction_mse": 0.12,
                "reconstruction_cosine": 0.91,
                "ce_loss_score": 0.75,
                "kl_div_score": 0.04,
                "decoder_pairwise_cosine_similarity": 0.08,
                "vg_expected_l0": 48.0,
            },
            {
                "model_name": "google/gemma-2-2b",
                "layer": 5,
                "hook_name": "blocks.5.hook_resid_post",
                "method": "batchtopk",
                "method_label": "BatchTopK SAE",
                "control_name": "k",
                "control_value": 40,
                "average_l0": 40.0,
                "sae_width": 32768,
                "rho_model": 40.0 / 32768,
                "dead_fraction": 0.1,
                "explained_variance": 0.82,
                "reconstruction_mse": 0.11,
                "reconstruction_cosine": 0.92,
                "ce_loss_score": None,
                "kl_div_score": None,
                "decoder_pairwise_cosine_similarity": 0.07,
                "vg_expected_l0": None,
            },
            {
                "model_name": "meta-llama/Llama-3.2-1B",
                "layer": 7,
                "hook_name": "blocks.7.hook_resid_post",
                "method": "jumprelu",
                "method_label": "JumpReLU SAE",
                "control_name": "lambda",
                "control_value": 1.0,
                "average_l0": 115.0,
                "sae_width": 32768,
                "rho_model": 115.0 / 32768,
                "dead_fraction": 0.3,
                "explained_variance": 0.79,
                "reconstruction_mse": 0.14,
                "reconstruction_cosine": 0.89,
                "ce_loss_score": 0.7,
                "kl_div_score": 0.06,
                "decoder_pairwise_cosine_similarity": 0.09,
                "vg_expected_l0": None,
            },
        ],
    )

    paths = plot_all(tmp_path)

    assert [path.name for path in paths] == [
        "reconstruction_metrics.png",
        "sparsity_diagnostics.png",
        "decoder_pairwise_cosine.png",
    ]
    assert all(
        path.parent == tmp_path / "summary" / "last" / "figures"
        for path in paths
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_plot_all_accepts_aliases_and_missing_optional_metrics(tmp_path: Path) -> None:
    _write_metrics(
        tmp_path,
        [
            {
                "model_name": "google/gemma-2-2b",
                "hook_name": "blocks.12.hook_resid_post",
                "method": "l1",
                "method_label": "L1/ReLU SAE",
                "sae_l0": 80.0,
                "sae_width": 32768,
                "dead_fraction": None,
                "explained_variance": 0.7,
                "mse": 0.2,
                "cossim": 0.85,
                "decoder_pairwise_cosine": 0.1,
            }
        ],
    )

    paths = plot_all(tmp_path)

    assert len(paths) == 3
    assert all(path.exists() for path in paths)
