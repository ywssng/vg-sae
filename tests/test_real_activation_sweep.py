from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from sae_lens import (
    BatchTopKTrainingSAEConfig,
    JumpReLUTrainingSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)

import src.real_activation_sweep as real_sweep
from src.real_activation_sweep import (
    CONTROL_GRID_PROVENANCE,
    CONTROL_GRID_RATIONALE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_DOWNSTREAM_EVAL_TOKENS,
    DEFAULT_EVAL_TOKENS,
    DEFAULT_STAGE3_CONTROLS,
    DEFAULT_TRAIN_TOKENS,
    GEMMA_DATASET_ID,
    GEMMA_DATASET_REVISION,
    GEMMA_LAYER12_PAPER_REPORTED_TRAIN_TOKENS,
    GEMMA_LAYER12_TRAIN_TOKENS,
    GEMMA_LAYER12_DATASET_ID,
    GEMMA_LAYER12_DATASET_REVISION,
    GEMMA_MODEL_REVISION,
    JUMPRELU_WARMUP_TOKENS,
    LLAMA_DATASET_ID,
    LLAMA_DATASET_REVISION,
    LLAMA_MODEL_REVISION,
    PAPER_BATCHTOPK_K_GRID,
    PAPER_BATCHTOPK_K_GRIDS,
    PAPER_JUMPRELU_COEFFICIENT_GRID,
    PAPER_REPORTED_TRAIN_TOKENS,
    PAPER_TARGET_DENSITIES,
    REAL_MODEL_TARGETS,
    SAELENS_REVISION,
    SAE_WIDTH,
    STAGE2_DENSITY_ANCHORS,
    STAGE2_L1_CONTROL_ANCHORS,
    STAGE2_VG_CONTROL_ANCHORS,
    STAGE3_CONTROL_NAMES,
    STAGE3_METHOD_ORDER,
    TRAINING_BUDGET_RATIONALE,
    RealActivationSweepConfig,
    RealActivationTrainingConfig,
    build_model,
    build_specs,
    default_sweep_config,
    default_sweep_configs,
    default_sweep_dir,
    interpolate_stage2_control,
    load_checkpoint,
    method_learning_rate,
    run_directory,
    save_checkpoint,
    sweep_experiment_id,
)
from src.saelens_vg import VGTrainingSAEConfig


def test_default_protocol_pins_targets_data_and_four_method_scope() -> None:
    assert STAGE3_METHOD_ORDER == ("vgsae", "l1", "batchtopk", "jumprelu")
    assert tuple(REAL_MODEL_TARGETS) == (
        "gemma-2-2b-layer5",
        "gemma-2-2b-layer12",
        "llama-3.2-1b-layer7",
    )
    layer5 = REAL_MODEL_TARGETS["gemma-2-2b-layer5"]
    layer12 = REAL_MODEL_TARGETS["gemma-2-2b-layer12"]
    llama7 = REAL_MODEL_TARGETS["llama-3.2-1b-layer7"]
    assert (layer5.input_dim, layer5.hook_name, layer5.paper_hook_name) == (
        2_304,
        "model.layers.5",
        "blocks.5.hook_resid_post",
    )
    assert (layer12.input_dim, layer12.hook_name, layer12.paper_hook_name) == (
        2_304,
        "model.layers.12",
        "blocks.12.hook_resid_post",
    )
    assert (llama7.input_dim, llama7.hook_name, llama7.paper_hook_name) == (
        2_048,
        "model.layers.7",
        "blocks.7.hook_resid_post",
    )
    assert layer5.default_seeds == llama7.default_seeds == (0, 1, 2)
    assert layer12.default_seeds == (0,)
    assert layer5.model_revision == layer12.model_revision == GEMMA_MODEL_REVISION
    assert llama7.model_revision == LLAMA_MODEL_REVISION
    assert (layer5.dataset_id, layer5.dataset_revision) == (
        GEMMA_DATASET_ID,
        GEMMA_DATASET_REVISION,
    )
    assert (layer12.dataset_id, layer12.dataset_revision) == (
        GEMMA_LAYER12_DATASET_ID,
        GEMMA_LAYER12_DATASET_REVISION,
    )
    assert (llama7.dataset_id, llama7.dataset_revision) == (
        LLAMA_DATASET_ID,
        LLAMA_DATASET_REVISION,
    )

    configs = default_sweep_configs()
    assert [config.data.target_name for config in configs] == list(REAL_MODEL_TARGETS)
    assert len({id(config.controls) for config in configs}) == len(configs)
    for config in configs:
        assert config.data.sae_width == SAE_WIDTH
        assert config.data.context_size == DEFAULT_CONTEXT_SIZE
        target = REAL_MODEL_TARGETS[config.data.target_name]
        assert config.seeds == list(target.default_seeds)
        assert config.data.n_train_tokens == target.train_tokens
        assert config.data.paper_reported_train_tokens == target.paper_reported_train_tokens
        assert config.data.training_budget_rationale == TRAINING_BUDGET_RATIONALE
        assert config.data.n_eval_tokens == DEFAULT_EVAL_TOKENS
        assert (
            config.data.n_downstream_eval_tokens
            == DEFAULT_DOWNSTREAM_EVAL_TOKENS
        )
        assert config.data.train_token_offset == 0
        assert config.data.eval_token_offset == target.train_tokens
        assert config.data.is_dataset_tokenized is True
        assert config.data.prepend_bos is True
        assert config.training.batch_size == DEFAULT_BATCH_SIZE
        assert config.training.beta_mode == "learned"
        assert config.training.preview_tokens == 80
        assert config.methods == list(STAGE3_METHOD_ORDER)
        assert config.control_grid_provenance == CONTROL_GRID_PROVENANCE
        assert config.control_grid_rationale == CONTROL_GRID_RATIONALE
        assert config.saelens_revision == SAELENS_REVISION

    assert tuple(DEFAULT_STAGE3_CONTROLS["batchtopk"]) == PAPER_BATCHTOPK_K_GRID
    assert tuple(DEFAULT_STAGE3_CONTROLS["jumprelu"]) == (
        PAPER_JUMPRELU_COEFFICIENT_GRID
    )
    assert REAL_MODEL_TARGETS["gemma-2-2b-layer12"].train_tokens == (
        GEMMA_LAYER12_TRAIN_TOKENS
    )
    assert REAL_MODEL_TARGETS["gemma-2-2b-layer12"].paper_reported_train_tokens == (
        GEMMA_LAYER12_PAPER_REPORTED_TRAIN_TOKENS
    )


def test_log_density_interpolation_and_extrapolation_are_deterministic() -> None:
    for density, l1, vg in zip(
        STAGE2_DENSITY_ANCHORS,
        STAGE2_L1_CONTROL_ANCHORS,
        STAGE2_VG_CONTROL_ANCHORS,
        strict=True,
    ):
        assert interpolate_stage2_control("l1", density) == pytest.approx(
            l1, abs=2.0e-14
        )
        assert interpolate_stage2_control("vgsae", density) == pytest.approx(
            vg, abs=2.0e-14
        )

    expected_l1 = tuple(
        interpolate_stage2_control("l1", density)
        for density in PAPER_TARGET_DENSITIES
    )
    expected_vg = tuple(
        interpolate_stage2_control("vgsae", density)
        for density in PAPER_TARGET_DENSITIES
    )
    assert tuple(DEFAULT_STAGE3_CONTROLS["l1"]) == expected_l1
    assert tuple(DEFAULT_STAGE3_CONTROLS["vgsae"]) == expected_vg
    # K=10 and K=20 are below the 1e-3 Stage-2 range and therefore exercise
    # endpoint extrapolation rather than clamping to the first anchor.
    assert expected_l1[:2] == pytest.approx((22.076061443392226, 17.184324014232867))
    assert expected_vg[:2] == pytest.approx((35.174873211392224, 26.89654833063275))
    assert all(left > right for left, right in zip(expected_l1, expected_l1[1:]))
    assert all(left > right for left, right in zip(expected_vg, expected_vg[1:]))
    with pytest.raises(ValueError, match="only for"):
        interpolate_stage2_control("batchtopk", 0.01)
    with pytest.raises(ValueError, match="positive"):
        interpolate_stage2_control("l1", 0.0)


def test_validation_rejects_identity_beta_and_heldout_overlap() -> None:
    config = default_sweep_config()
    config.data = replace(config.data, model_revision="main")
    with pytest.raises(ValueError, match="model_revision"):
        config.validate()

    config = default_sweep_config()
    config.data = replace(config.data, dataset_revision="main")
    with pytest.raises(ValueError, match="dataset_revision"):
        config.validate()

    config = default_sweep_config()
    config.data = replace(config.data, hook_name="blocks.12.hook_resid_post")
    with pytest.raises(ValueError, match="hook_name"):
        config.validate()

    config = default_sweep_config()
    config.data = replace(config.data, eval_token_offset=DEFAULT_TRAIN_TOKENS - 1)
    with pytest.raises(ValueError, match="Held-out"):
        config.validate()

    config = default_sweep_config()
    config.data = replace(
        config.data, n_downstream_eval_tokens=config.data.n_eval_tokens + 1
    )
    with pytest.raises(ValueError, match="n_downstream_eval_tokens"):
        config.validate()

    config = default_sweep_config()
    config.data = replace(config.data, n_train_tokens=500_000_000)
    with pytest.raises(ValueError, match="cached contexts"):
        config.validate()

    with pytest.raises(ValueError, match="beta_mode='learned'"):
        RealActivationTrainingConfig(beta_mode="profiled")  # type: ignore[arg-type]

    config = default_sweep_config()
    config.methods.append("topk")
    with pytest.raises(ValueError, match="Unknown Stage-3 methods"):
        config.validate()


def test_specs_ids_and_serialization_separate_target_and_budget(tmp_path: Path) -> None:
    config = default_sweep_config("gemma-2-2b-layer5")
    specs = build_specs(config)
    target_grid = PAPER_BATCHTOPK_K_GRIDS["gemma-2-2b-layer5"]
    assert len(specs) == (len(target_grid) * 3 + 9) * 3
    assert [spec.method for spec in specs[: len(target_grid)]] == ["vgsae"] * len(target_grid)
    assert [spec.control_value for spec in specs[-9:]] == list(
        PAPER_JUMPRELU_COEFFICIENT_GRID
    )
    assert {
        (
            spec.init_seed,
            spec.calibration_seed,
            spec.train_stream_seed,
            spec.eval_stream_seed,
        )
        for spec in specs
    } == {
        (50_000, 20_000, 30_000, 40_000),
        (50_001, 20_001, 30_001, 40_001),
        (50_002, 20_002, 30_002, 40_002),
    }
    assert all(spec.control_name == STAGE3_CONTROL_NAMES[spec.method] for spec in specs)

    experiment_id = sweep_experiment_id(config)
    assert "gemma-2-2b-layer5" in experiment_id
    assert "train500m" in experiment_id
    assert "ctx1024" in experiment_id
    assert "beta_learned" in experiment_id
    assert default_sweep_dir(tmp_path, config) == (
        tmp_path / "outputs" / "runs" / experiment_id
    )
    assert run_directory(tmp_path, specs[0]) == (
        tmp_path / "runs" / "vgsae" / specs[0].run_id
    )
    close_left = real_sweep.RealActivationRunSpec(
        method="vgsae",
        control_name="lambda_sparsity",
        control_value=1.0000000000001,
        seed=0,
        init_seed=1,
        calibration_seed=2,
        train_stream_seed=3,
        eval_stream_seed=4,
    )
    close_right = replace(close_left, control_value=1.0000000000002)
    assert close_left.run_id != close_right.run_id

    llama = default_sweep_config("llama-3.2-1b-layer7", n_eval_tokens=2_097_152)
    assert sweep_experiment_id(llama) != experiment_id
    assert "eval2097152" in sweep_experiment_id(llama)
    restored = RealActivationSweepConfig.from_dict(llama.to_dict())
    assert restored.to_dict() == llama.to_dict()


class _CapturedModel:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg


def test_factories_use_official_configs_and_published_settings(monkeypatch) -> None:
    monkeypatch.setattr(real_sweep, "VGTrainingSAE", _CapturedModel)
    monkeypatch.setattr(real_sweep, "StandardTrainingSAE", _CapturedModel)
    monkeypatch.setattr(real_sweep, "BatchTopKTrainingSAE", _CapturedModel)
    monkeypatch.setattr(real_sweep, "JumpReLUTrainingSAE", _CapturedModel)
    config = default_sweep_config()
    first_by_method = {
        method: next(spec for spec in build_specs(config) if spec.method == method)
        for method in STAGE3_METHOD_ORDER
    }

    vg = build_model(config, first_by_method["vgsae"]).cfg
    assert isinstance(vg, VGTrainingSAEConfig)
    assert vg.beta_mode == "learned"
    assert vg.lambda_sparsity == first_by_method["vgsae"].control_value

    l1 = build_model(config, first_by_method["l1"]).cfg
    assert isinstance(l1, StandardTrainingSAEConfig)
    assert l1.l1_coefficient == first_by_method["l1"].control_value
    assert l1.l1_warm_up_steps == (
        GEMMA_LAYER12_TRAIN_TOKENS // DEFAULT_BATCH_SIZE
    ) // 3

    batchtopk = build_model(config, first_by_method["batchtopk"]).cfg
    assert isinstance(batchtopk, BatchTopKTrainingSAEConfig)
    assert batchtopk.k == 10.0
    assert batchtopk.rescale_acts_by_decoder_norm is True
    assert batchtopk.decoder_init_norm == 0.1

    jumprelu = build_model(config, first_by_method["jumprelu"]).cfg
    assert isinstance(jumprelu, JumpReLUTrainingSAEConfig)
    assert jumprelu.l0_coefficient == 0.125
    assert jumprelu.l0_warm_up_steps == JUMPRELU_WARMUP_TOKENS // DEFAULT_BATCH_SIZE
    assert jumprelu.jumprelu_sparsity_loss_mode == "tanh"
    assert jumprelu.jumprelu_bandwidth == 2.0
    assert jumprelu.jumprelu_tanh_scale == 4.0
    assert jumprelu.pre_act_loss_coefficient == 3.0e-6
    assert jumprelu.jumprelu_init_threshold == 0.1
    assert jumprelu.decoder_init_norm == 0.1
    assert jumprelu.normalize_activations == "none"
    assert jumprelu.metadata.model_name == "google/gemma-2-2b"
    assert jumprelu.metadata.hook_name == "model.layers.12"
    assert jumprelu.metadata.paper_hook_name == "blocks.12.hook_resid_post"
    assert jumprelu.metadata.dataset_path == GEMMA_LAYER12_DATASET_ID
    assert jumprelu.metadata.prepend_bos is True

    assert method_learning_rate(config, "vgsae") == 3.0e-4
    assert method_learning_rate(config, "l1") == 3.0e-4
    assert method_learning_rate(config, "batchtopk") == 3.0e-4
    assert method_learning_rate(config, "jumprelu") == 2.0e-4


def test_final_checkpoint_round_trip_uses_native_registry(tmp_path: Path) -> None:
    config = default_sweep_config()
    spec = next(spec for spec in build_specs(config) if spec.method == "l1")
    model = StandardTrainingSAE(
        StandardTrainingSAEConfig(d_in=3, d_sae=5, device="cpu")
    )
    with torch.no_grad():
        model.W_dec.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        config=config,
        spec=spec,
        step=12,
        n_training_tokens=53_248,
        loss=0.125,
        metadata={"train_fingerprint": "fingerprint"},
    )

    restored, payload = load_checkpoint(checkpoint)
    assert isinstance(restored, StandardTrainingSAE)
    assert torch.equal(restored.W_dec, model.W_dec)
    assert payload["format_version"] == 1
    assert payload["checkpoint_kind"] == "last"
    assert payload["run_spec"] == spec.to_dict()
    assert payload["sweep_config"] == config.to_dict()
    assert payload["n_training_tokens"] == 53_248
    assert payload["metadata"]["train_fingerprint"] == "fingerprint"
    assert not list(tmp_path.glob(".*.tmp.pt"))
