import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sae_lens import TrainingSAE

import scripts.create_experiment_notebooks as notebook_generator
from src.sae_baselines import to_inference_sae
from src.sae_model import (
    BatchTopKSAE,
    GatedSAE,
    JumpReLUSAE,
    StandardSAE,
    StandardSAEConfig,
    TopKSAE,
    VariationalGarroteSAE,
)


METHODS = ("vgsae", "l1", "topk", "batchtopk", "jumprelu", "gated")
METHOD_ORDER_SOURCE = "METHOD_ORDER = (" + ", ".join(f'\"{method}\"' for method in METHODS) + ")"


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _full_configuration(notebook: dict) -> dict:
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("id") == "41316ace"
    )
    namespace = {"os": SimpleNamespace(environ={})}
    exec(compile(source.split("EXP_DIR =")[0], "notebook-07-config", "exec"), namespace)
    return namespace


def test_notebook_07_covers_all_methods_with_one_shared_order() -> None:
    path = Path("notebooks/07_synthetic_sparse_coding_rho_model_comparison.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = _source(notebook)
    assert METHOD_ORDER_SOURCE in source
    for method in METHODS:
        assert f'"{method}"' in source
    assert 'for method in ["vgsae"' not in source
    assert "BatchTopKSAEConfig" in source and "JumpReLUSAEConfig" in source
    assert "hard_gate_and_positive_magnitude" in source and "sigmoid_gate" not in source
    assert "exp07_saelens_v647_all_official" in source
    assert "embedded outputs" in source and "pre-migration" in source
    assert "n_experiment_seeds = 1" in source
    assert "n_experiment_seeds = 5" not in source
    assert "standard_l1_coefficients" in source
    assert "-0.9, -1.0, -1.1" in source
    assert "topk_values = list(range(1, n_features + 1))" in source
    assert "0.0625, 0.09375, 0.125, 0.1875" in source
    assert "30.0, 10.0, 5.0, 4.3, 4.25" in source
    assert "10.0, 9.0, 7.0, 5.0" in source
    assert 'METHOD_ORDER.index(spec["method"])' in source
    assert "spec_index" not in source
    assert 'row["init_seed"] = init_seed' in source
    assert "dead_feature_window = 100" in source
    assert "dead_feature_window=dead_feature_window" in source
    assert "plt.subplots(1, 4" in source
    assert "zip(axes, metrics, strict=True)" in source
    assert 'metric == "reconstruction_error"' in source
    assert 'set_xscale("symlog"' in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_notebook_07_uses_the_calibrated_single_seed_grids() -> None:
    notebook = json.loads(
        Path("notebooks/07_synthetic_sparse_coding_rho_model_comparison.ipynb").read_text(
            encoding="utf-8"
        )
    )
    config = _full_configuration(notebook)

    assert config["seeds"] == [0]
    assert tuple(config["gamma_values"]) == (
        12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5,
        2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.35, 0.25, 0.125, 0.0,
        -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1,
    )
    assert tuple(config["standard_l1_coefficients"]) == (
        5.0, 4.5, 3.5, 3.0, 2.0, 1.0, 0.5, 0.3,
        0.25, 0.2, 0.1, 0.05, 0.02, 0.002, 0.0003, 0.0,
    )
    assert tuple(config["topk_values"]) == tuple(range(1, 33))
    assert tuple(config["batchtopk_values"]) == (
        0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.5, 0.75,
        1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0,
        14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0,
        30.0, 31.0, 32.0,
    )
    assert tuple(config["jumprelu_l0_coefficients"]) == (
        30.0, 10.0, 5.0, 4.3, 4.25, 4.2, 4.1, 4.0,
        3.5, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3,
        0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.015,
        0.01, 0.007, 0.006, 0.005, 0.003, 0.001, 0.0003, 0.0,
    )
    assert tuple(config["gated_l1_coefficients"]) == (
        10.0, 9.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03,
        0.025, 0.02, 0.015, 0.0125, 0.01, 0.005, 0.0,
    )
    assert sum(
        len(config[name])
        for name in (
            "gamma_values",
            "standard_l1_coefficients",
            "topk_values",
            "batchtopk_values",
            "jumprelu_l0_coefficients",
            "gated_l1_coefficients",
        )
    ) == 163


def test_bulk_generator_preserves_curated_notebooks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(notebook_generator, "NOTEBOOK_DIR", tmp_path)
    curated = [
        tmp_path / "02_synthetic_sparse_coding_transition.ipynb",
        tmp_path / "07_synthetic_sparse_coding_rho_model_comparison.ipynb",
    ]
    for path in curated:
        path.write_text("curated", encoding="utf-8")
    notebook_generator.write_notebooks()
    assert all(path.read_text(encoding="utf-8") == "curated" for path in curated)
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == Path(
        "notebooks/README.md"
    ).read_text(encoding="utf-8")


def test_notebook_02_uses_official_standard_and_new_output_directory() -> None:
    path = Path("notebooks/02_synthetic_sparse_coding_transition.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = _source(notebook)

    assert "StandardSAEConfig(d_in=data.x.shape[1], d_sae=width" in source
    assert "exp02_synthetic_sparse_coding_all_official" in source
    assert "embedded outputs" in source and "pre-migration" in source
    assert "official SAELens Standard/L1, TopK, and Gated SAEs" in source
    assert "deterministic sigmoid-gated SAE" not in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-02-cell-{index}", "exec")


def test_notebook_l1_mask_reuses_training_threshold() -> None:
    notebook = json.loads(
        Path("notebooks/07_synthetic_sparse_coding_rho_model_comparison.ipynb").read_text(
            encoding="utf-8"
        )
    )
    helper_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def latent_values_and_masks" in "".join(cell.get("source", []))
    )
    namespace = {
        "np": np,
        "torch": torch,
        "BatchTopKSAE": BatchTopKSAE,
        "GatedSAE": GatedSAE,
        "JumpReLUSAE": JumpReLUSAE,
        "StandardSAE": StandardSAE,
        "TopKSAE": TopKSAE,
        "TrainingSAE": TrainingSAE,
        "VariationalGarroteSAE": VariationalGarroteSAE,
        "to_inference_sae": to_inference_sae,
    }
    exec(compile(helper_source, "notebook-helpers", "exec"), namespace)
    model = StandardSAE(StandardSAEConfig(d_in=1, d_sae=1))
    with torch.no_grad():
        model.W_enc.fill_(1.0)
        model.W_dec.fill_(1.0)
        model.b_enc.zero_()
        model.b_dec.zero_()
    _, mask, info = namespace["latent_values_and_masks"](
        model, torch.tensor([[1.0], [3.0]]), l1_threshold=2.0
    )
    assert torch.equal(mask, torch.tensor([[0.0], [1.0]]))
    assert info["l1_gmm_threshold"] == 2.0
