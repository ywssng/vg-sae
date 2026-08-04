import json
from pathlib import Path

import numpy as np
import torch
from sae_lens import TrainingSAE

import scripts.create_experiment_notebooks as notebook_generator
from src.sae_baselines import to_inference_sae
from src.sae_model import (
    BatchTopKSAE,
    GatedSAE,
    JumpReLUSAE,
    L1ReLUSAE,
    L1SAEConfig,
    TopKSAE,
    VariationalGarroteSAE,
)


METHODS = ("vgsae", "l1", "topk", "batchtopk", "jumprelu", "gated")
METHOD_ORDER_SOURCE = "METHOD_ORDER = (" + ", ".join(f'\"{method}\"' for method in METHODS) + ")"


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


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
    assert "exp07_saelens_v647_six_method" in source
    assert "dead_feature_window = 100" in source
    assert "dead_feature_window=dead_feature_window" in source
    assert "plt.subplots(1, 4" in source
    assert "zip(axes, metrics, strict=True)" in source
    assert 'metric == "reconstruction_error"' in source
    assert 'set_xscale("symlog"' in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")
def test_bulk_generator_preserves_curated_notebook_07(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(notebook_generator, "NOTEBOOK_DIR", tmp_path)
    path = tmp_path / "07_synthetic_sparse_coding_rho_model_comparison.ipynb"
    path.write_text("curated", encoding="utf-8")
    notebook_generator.write_notebooks()
    assert path.read_text(encoding="utf-8") == "curated"


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
        "L1ReLUSAE": L1ReLUSAE,
        "TopKSAE": TopKSAE,
        "TrainingSAE": TrainingSAE,
        "VariationalGarroteSAE": VariationalGarroteSAE,
        "to_inference_sae": to_inference_sae,
    }
    exec(compile(helper_source, "notebook-helpers", "exec"), namespace)
    model = L1ReLUSAE(L1SAEConfig(1, 1, decoder_bias=False))
    with torch.no_grad():
        model.encoder.weight.fill_(1.0)
        model.encoder.bias.zero_()
    _, mask, info = namespace["latent_values_and_masks"](
        model, torch.tensor([[1.0], [3.0]]), l1_threshold=2.0
    )
    assert torch.equal(mask, torch.tensor([[0.0], [1.0]]))
    assert info["l1_gmm_threshold"] == 2.0
