from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/09_saelens_synthsaebench_rho_model_comparison.ipynb")
METHOD_ORDER = '("vgsae", "l1", "topk", "batchtopk", "jumprelu", "gated")'


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source(notebook: dict, *, code_only: bool = False) -> str:
    cells = notebook["cells"]
    if code_only:
        cells = [cell for cell in cells if cell["cell_type"] == "code"]
    return "\n".join("".join(cell.get("source", [])) for cell in cells)


def test_notebook_09_is_clean_and_compiles() -> None:
    notebook = _notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None
        assert not cell.get("outputs")
        compile("".join(cell["source"]), f"notebook-09-cell-{index}", "exec")


def test_notebook_09_uses_official_full_model_and_six_native_paths() -> None:
    notebook = _notebook()
    source = _source(notebook)
    code = _source(notebook, code_only=True)

    assert f"METHOD_ORDER = {METHOD_ORDER}" in code
    assert 'SAELENS_VERSION = "6.47.0"' in code
    assert "assert sae_lens.__version__ == SAELENS_VERSION" in code
    assert "8be14080485952f729ed58d674bcddf9778e0aa4" in source
    assert 'BENCHMARK_MODEL_ID = "decoderesearch/synth-sae-bench-16k-v1"' in code
    assert 'BENCHMARK_REVISION = "b2efd8b919ae46d6d487c73d46db5ee52813621d"' in code
    assert "snapshot_download" in code
    assert "revision=BENCHMARK_REVISION" in code
    assert "SyntheticModel.load_from_disk" in code
    assert '"benchmark_revision": BENCHMARK_REVISION' in code
    assert "if FAST_DEV_RUN:" in code and "else:" in code
    assert "hidden_dim, n_true_features, d_sae = 768, 16_384, 4_096" in code
    assert 'env_values("VGSAE_TOPK_VALUES", [15, 25, 35, 45], int)' in code

    for class_name in (
        "StandardTrainingSAE",
        "TopKTrainingSAE",
        "BatchTopKTrainingSAE",
        "JumpReLUTrainingSAE",
        "GatedTrainingSAE",
        "VGTrainingSAE",
        "VGSAETrainer",
    ):
        assert class_name in code


def test_notebook_09_records_fairness_and_inference_audits() -> None:
    code = _source(_notebook(), code_only=True)
    assert 'VG_BETA_MODE not in {"fixed", "learned"}' in code
    assert 'beta_mode="profiled"' not in code
    assert "SyntheticActivationIterator" in code
    assert "temporary_seed(train_seed_offset + seed)" in code
    assert "OfficialActivationScaler" in code
    assert "calibrate_scale" in code
    assert "TensorActivationProvider" not in code
    assert "split_activations" not in code
    assert "batch_schedule_digest" not in code
    assert 'results["n_training_samples"].eq(total_training_samples).all()' in code
    assert "from src.sae_baselines import to_inference_sae" in code
    assert "to_inference_sae(trained, fold_decoder_norm=True)" in code
    assert "eval_sae_on_synthetic_data" in code
    assert 'posterior["m"]' in code
    assert "vg_posterior_rho" in code
    assert "vg_expected_explained_variance" in code


def test_notebook_09_uses_only_official_streaming_recovery_metrics() -> None:
    source = _source(_notebook())
    code = _source(_notebook(), code_only=True)

    for metric in (
        "sae_l0",
        "true_l0",
        "mcc",
        "uniqueness",
        "classification_precision",
        "classification_recall",
        "classification_f1",
        "classification_accuracy",
        "explained_variance",
        "dead_latents",
        "shrinkage",
    ):
        assert f'"{metric}"' in code
    assert '"rho_model": result.sae_l0 / d_sae' in code
    assert '"true_l0_over_d_sae": result.true_l0 / d_sae' in code
    assert "linear_sum_assignment" not in source
    assert "align_latents" not in source
    assert "mask_cache" not in source


def test_notebook_09_fast_mode_is_six_by_two_and_full_is_configurable() -> None:
    source = _source(_notebook())
    code = _source(_notebook(), code_only=True)

    assert "FAST 6x2 integration" in code
    assert "gamma_values = [0.0, 1.0]" in code
    assert "l1_values = [1.0e-3, 1.0e-2]" in code
    assert "topk_values = [2, 4]" in code
    assert "jumprelu_values = [1.0e-3, 1.0e-2]" in code
    for name in (
        "VGSAE_TRAINING_SAMPLES",
        "VGSAE_CALIBRATION_SAMPLES",
        "VGSAE_EVAL_SAMPLES",
        "VGSAE_EVAL_BATCH_SIZE",
        "VGSAE_SEEDS",
    ):
        assert name in code
    assert "200M" in source
