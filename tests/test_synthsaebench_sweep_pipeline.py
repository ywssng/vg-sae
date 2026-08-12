from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sae_lens = pytest.importorskip("sae_lens")
if sae_lens.__version__ != "6.47.0":
    pytest.skip(
        "SynthSAEBench integration is pinned to sae-lens 6.47.0",
        allow_module_level=True,
    )

from sae_lens import (
    BatchTopKTrainingSAE,
    GatedTrainingSAE,
    JumpReLUTrainingSAE,
    StandardTrainingSAE,
    TopKTrainingSAE,
)
from sae_lens.synthetic import (
    ConstantFiringProbabilityConfig,
    SyntheticModel,
    SyntheticModelConfig,
    eval_sae_on_synthetic_data,
)

import src.synthsaebench_sweep as sweep_module
from runs.run_SynthSAEBench_sweep import (
    _final_beta_precision,
    _load_resume_checkpoint,
    _preflight_wandb,
    _save_resume_checkpoint,
    _trainer,
    _wandb_run,
    configured_sweep,
    parse_args,
)
import runs.run_SynthSAEBench_sweep_eval as synth_eval_runner
from runs.run_SynthSAEBench_sweep_eval import parse_args as parse_eval_args
from runs._sweep_io import write_json, write_rows
from src.sae_baselines import to_inference_sae
from src.saelens_vg import VGTrainingSAE
from src.synthsaebench_eval import evaluate_model
from src.synthsaebench_sweep import (
    BENCHMARK_CONFIG_SHA256,
    BENCHMARK_INPUT_DIM,
    BENCHMARK_MODEL_ID,
    BENCHMARK_NUM_FEATURES,
    BENCHMARK_REVISION,
    BENCHMARK_SAE_WIDTH,
    BENCHMARK_SCALE_CHILDREN_BY_PARENT,
    CALIBRATION_CONTROLS,
    DEFAULT_MAX_PER_DEVICE,
    FINAL_CONTROLS,
    FINAL_VG_CONTROLS_BY_BETA_MODE,
    SAELENS_REVISION,
    SynthSAEBenchDataConfig,
    SynthSAEBenchRunSpec,
    SynthSAEBenchSweepConfig,
    SynthSAEBenchTrainingConfig,
    build_model,
    build_specs,
    default_sweep_config,
    default_sweep_dir,
    final_controls_for_beta_mode,
    load_benchmark_model,
    load_checkpoint,
    save_checkpoint,
    sweep_experiment_id,
    temporary_seed_for_device,
)


def _small_config(monkeypatch: pytest.MonkeyPatch) -> SynthSAEBenchSweepConfig:
    monkeypatch.setattr(sweep_module, "BENCHMARK_INPUT_DIM", 3)
    monkeypatch.setattr(sweep_module, "BENCHMARK_NUM_FEATURES", 6)
    monkeypatch.setattr(sweep_module, "BENCHMARK_SAE_WIDTH", 5)
    return SynthSAEBenchSweepConfig(
        data=SynthSAEBenchDataConfig(
            input_dim=3,
            ground_truth_num_features=6,
            sae_width=5,
            n_train=12,
            n_test=4,
        ),
        training=SynthSAEBenchTrainingConfig(
            batch_size=4,
            history_every=1,
            dead_feature_window=2,
            feature_sampling_window=2,
            n_batches_for_norm_estimate=2,
            heatmap_samples=3,
            resume_every=1,
            autocast_sae=False,
            autocast_data=False,
        ),
        controls={
            "vgsae": [0.4],
            "l1": [4.0],
            "topk": [2],
            "batchtopk": [2.0],
            "jumprelu": [1.0],
            "gated": [8.0],
        },
    )


def test_default_protocol_fixes_pretrained_generator_and_exact_eighth() -> None:
    config = default_sweep_config()

    assert config.data.model_id == BENCHMARK_MODEL_ID
    assert config.data.revision == BENCHMARK_REVISION
    assert config.data.model_config_sha256 == BENCHMARK_CONFIG_SHA256
    assert config.data.input_dim == BENCHMARK_INPUT_DIM
    assert config.data.ground_truth_num_features == BENCHMARK_NUM_FEATURES
    assert config.data.sae_width == BENCHMARK_SAE_WIDTH
    assert config.data.scale_children_by_parent is BENCHMARK_SCALE_CHILDREN_BY_PARENT
    assert config.data.n_test * 8 == config.data.n_train
    assert config.data.n_train % config.training.batch_size == 0
    assert config.data.n_test % config.training.batch_size == 0
    assert config.total_training_steps == 195_312
    assert config.training.batch_size == 1_024
    assert config.training.lr == pytest.approx(3.0e-4)
    assert config.training.lr_decay_fraction == 0.0
    assert config.training.beta_mode == "profiled"
    assert config.training.resume_every == 10_000
    assert SAELENS_REVISION == "8be14080485952f729ed58d674bcddf9778e0aa4"
    assert config.controls["topk"] == [15, 20, 25, 30, 35, 40, 45]
    assert config.controls == FINAL_CONTROLS
    assert config.experiment_name == "stage2_synthsaebench16k_l0calibrated"
    assert config.controls["vgsae"] == [1.64, 1.72, 1.84, 2.01, 2.26, 2.84, 6.12]
    assert config.controls["l1"] == [0.99, 1.07, 1.17, 1.36, 1.69, 2.42, 4.26]
    assert config.controls["jumprelu"] == [0.41, 0.46, 0.52, 0.61, 0.78, 1.16, 1.8]
    assert config.controls["gated"] == [1.07, 1.1, 1.21, 1.38, 1.7, 2.17, 3.28]
    expected_id = (
        "stage2_synthsaebench16k_l0calibrated_"
        "sae4096_train200m_test25m_beta_profiled_seed0"
    )
    assert sweep_experiment_id(config) == expected_id
    assert default_sweep_dir(Path("/project"), config) == (
        Path("/project") / "outputs" / "runs" / expected_id
    )
    learned = default_sweep_config(beta_mode="learned")
    assert learned.training.beta_mode == "learned"
    assert learned.controls["vgsae"] == [
        1.63,
        1.71,
        1.82,
        1.99,
        2.22,
        2.80,
        6.00,
    ]
    assert learned.controls["l1"] == config.controls["l1"]
    assert sweep_experiment_id(learned) == expected_id.replace(
        "beta_profiled", "beta_learned"
    )
    assert default_sweep_dir(Path("/project"), learned) != default_sweep_dir(
        Path("/project"), config
    )

    calibration = default_sweep_config(calibration=True)
    assert calibration.controls == CALIBRATION_CONTROLS
    assert calibration.experiment_name.endswith("_calibration")


def test_final_vg_controls_are_mode_specific_independent_copies() -> None:
    profiled = final_controls_for_beta_mode("profiled")
    learned = final_controls_for_beta_mode("learned")

    assert profiled["vgsae"] == FINAL_VG_CONTROLS_BY_BETA_MODE["profiled"]
    assert learned["vgsae"] == FINAL_VG_CONTROLS_BY_BETA_MODE["learned"]
    assert profiled["vgsae"] != learned["vgsae"]
    assert profiled["l1"] == learned["l1"]
    profiled["vgsae"].append(999.0)
    assert 999.0 not in FINAL_VG_CONTROLS_BY_BETA_MODE["profiled"]

    with pytest.raises(ValueError, match="profiled or learned"):
        final_controls_for_beta_mode("fixed")  # type: ignore[arg-type]


def test_generator_identity_and_dimensions_cannot_be_overridden() -> None:
    config = default_sweep_config()
    raw = config.to_dict()
    raw["data"]["revision"] = "main"
    with pytest.raises(ValueError, match="revision"):
        SynthSAEBenchSweepConfig.from_dict(raw)

    raw = config.to_dict()
    raw["data"]["sae_width"] = 2_048
    with pytest.raises(ValueError, match="sae_width"):
        SynthSAEBenchSweepConfig.from_dict(raw)

    raw = config.to_dict()
    raw["data"]["n_train"] -= 1
    with pytest.raises(ValueError, match="divisible by batch_size"):
        SynthSAEBenchSweepConfig.from_dict(raw)

    raw = config.to_dict()
    raw["training"]["beta_mode"] = "fixed"
    with pytest.raises(ValueError, match="profiled or learned"):
        SynthSAEBenchSweepConfig.from_dict(raw)

    with pytest.raises(ValueError, match="profiled or learned"):
        SynthSAEBenchTrainingConfig(beta_mode="fixed")  # type: ignore[arg-type]
    for beta in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            SynthSAEBenchTrainingConfig(beta=beta)


def test_specs_share_stream_seeds_across_methods_and_controls() -> None:
    config = default_sweep_config(fast=True)
    specs = build_specs(config)

    assert len(specs) == 12
    assert len({spec.train_stream_seed for spec in specs}) == 1
    assert len({spec.eval_stream_seed for spec in specs}) == 1
    assert len({spec.calibration_seed for spec in specs}) == 1
    assert len({spec.run_id for spec in specs}) == len(specs)


def test_cli_training_budget_derives_exact_one_eighth_test_and_direct_grid() -> None:
    config = configured_sweep(
        SimpleNamespace(
            config=None,
            fast_dev_run=False,
            seed=3,
            seeds=None,
            sparsity_controls=["topk=15,30,45", "l1=3,6,10"],
            batch_size=1_024,
            beta_mode="learned",
            training_samples=8_192,
            test_samples=None,
            history_every=2,
            lr_decay_fraction=1.0 / 3.0,
        )
    )

    assert config.seeds == [3]
    assert config.data.n_train == 8_192
    assert config.data.n_test == 1_024
    assert config.controls["topk"] == [15, 30, 45]
    assert config.controls["l1"] == [3.0, 6.0, 10.0]
    assert config.training.lr_decay_fraction == pytest.approx(1.0 / 3.0)
    assert config.training.beta_mode == "learned"
    assert config.controls["vgsae"] == FINAL_VG_CONTROLS_BY_BETA_MODE["learned"]


def test_synth_sweep_wandb_is_forced_online_with_filterable_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _small_config(monkeypatch)
    spec = build_specs(config)[0]
    sweep_dir = tmp_path / "stage2_synthsaebench16k_test"
    (sweep_dir / "manifest.json").parent.mkdir(parents=True)
    (sweep_dir / "manifest.json").write_text("{}")
    run_dir = sweep_dir / "runs" / spec.method / spec.run_id
    captured: dict[str, object] = {}
    sentinel = object()
    monkeypatch.setattr(
        "wandb.init", lambda **kwargs: captured.update(kwargs) or sentinel
    )

    result = _wandb_run({"sweep_config": config.to_dict()}, spec, run_dir)

    assert result is sentinel
    assert captured["group"] == "stage2_synthsaebench16k_test"
    assert captured["job_type"] == spec.method
    assert captured["mode"] == "online"
    assert captured["force"] is True
    assert captured["tags"] == [
        "stage:stage2_synthsaebench16k_l0calibrated",
        f"method:{spec.method}",
        "beta_mode:profiled",
    ]
    wandb_config = captured["config"]
    assert isinstance(wandb_config, dict)
    assert wandb_config["exp_id"] == sweep_experiment_id(config)
    assert wandb_config["stage"] == "stage2_synthsaebench16k_l0calibrated"
    assert wandb_config["sweep_root"] == "stage2_synthsaebench16k_test"
    assert wandb_config["method"] == spec.method
    assert wandb_config["beta_mode"] == "profiled"


def test_synth_wandb_records_learned_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _small_config(monkeypatch)
    raw = config.to_dict()
    raw["training"]["beta_mode"] = "learned"
    config = SynthSAEBenchSweepConfig.from_dict(raw)
    spec = build_specs(config)[0]
    sweep_dir = tmp_path / "stage2_beta_learned"
    (sweep_dir / "manifest.json").parent.mkdir(parents=True)
    (sweep_dir / "manifest.json").write_text("{}")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "wandb.init", lambda **kwargs: captured.update(kwargs) or object()
    )

    _wandb_run(
        {"sweep_config": config.to_dict()},
        spec,
        sweep_dir / "runs" / spec.method / spec.run_id,
    )

    assert "beta_mode:learned" in captured["tags"]
    assert captured["config"]["beta_mode"] == "learned"


@pytest.mark.parametrize("beta_mode", ["profiled", "learned"])
def test_synth_sweep_cli_accepts_supported_beta_modes(
    beta_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_SynthSAEBench_sweep.py", "--beta-mode", beta_mode],
    )
    assert parse_args().beta_mode == beta_mode


def test_synth_sweep_cli_rejects_fixed_beta_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_SynthSAEBench_sweep.py", "--beta-mode", "fixed"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_synth_eval_cli_selects_mode_and_rejects_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_SynthSAEBench_sweep_eval.py", "--beta-mode", "learned"],
    )
    assert parse_eval_args().beta_mode == "learned"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_SynthSAEBench_sweep_eval.py", "--beta-mode", "fixed"],
    )
    with pytest.raises(SystemExit):
        parse_eval_args()


def test_synth_wandb_preflight_requires_verified_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("wandb.login", lambda **kwargs: calls.append(kwargs) or True)

    _preflight_wandb()

    assert calls == [{"verify": True, "force": True}]


@pytest.mark.parametrize("option", ["--no-wandb", "--wandb-mode=offline"])
def test_synth_sweep_rejects_wandb_bypass_options(
    option: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_SynthSAEBench_sweep.py", option])

    with pytest.raises(SystemExit):
        parse_args()


def test_synth_sweep_defaults_to_benchmarked_worker_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_SynthSAEBench_sweep.py"])

    assert DEFAULT_MAX_PER_DEVICE == 2
    assert parse_args().max_per_device == DEFAULT_MAX_PER_DEVICE
    monkeypatch.setattr(sys, "argv", ["run_SynthSAEBench_sweep_eval.py"])
    assert parse_eval_args().max_per_device == DEFAULT_MAX_PER_DEVICE


def test_synth_wandb_rejects_run_without_sweep_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _small_config(monkeypatch)
    spec = build_specs(config)[0]

    with pytest.raises(ValueError, match="Cannot identify sweep root"):
        _wandb_run(
            {"sweep_config": config.to_dict()},
            spec,
            tmp_path / "runs" / spec.method / spec.run_id,
        )


def test_all_method_factories_use_benchmark_specific_settings(monkeypatch) -> None:
    config = _small_config(monkeypatch)
    expected_classes = {
        "vgsae": VGTrainingSAE,
        "l1": StandardTrainingSAE,
        "topk": TopKTrainingSAE,
        "batchtopk": BatchTopKTrainingSAE,
        "jumprelu": JumpReLUTrainingSAE,
        "gated": GatedTrainingSAE,
    }

    for spec in build_specs(config):
        model = build_model(config, spec)
        assert isinstance(model, expected_classes[spec.method])
        assert model.cfg.d_in == 3
        assert model.cfg.d_sae == 5
        if spec.method in {"l1", "gated"}:
            assert model.cfg.l1_warm_up_steps == 1
        if spec.method == "jumprelu":
            assert model.cfg.normalize_activations == "expected_average_only_in"
            assert model.cfg.jumprelu_init_threshold == pytest.approx(1.0)
            assert model.cfg.jumprelu_bandwidth == pytest.approx(1.0)
            assert model.cfg.decoder_init_norm == pytest.approx(0.5)
        else:
            assert model.cfg.normalize_activations == "none"
        del model
        gc.collect()


def test_pinned_loader_verifies_revision_hash_and_dimensions(
    monkeypatch, tmp_path
) -> None:
    config_bytes = b'{"pinned": true}\n'
    digest = hashlib.sha256(config_bytes).hexdigest()
    (tmp_path / "synthetic_model_config.json").write_bytes(config_bytes)
    monkeypatch.setattr(sweep_module, "BENCHMARK_CONFIG_SHA256", digest)
    config = SynthSAEBenchSweepConfig(
        data=SynthSAEBenchDataConfig(model_config_sha256=digest)
    )
    calls: dict[str, object] = {}

    def fake_download(*, repo_id: str, revision: str) -> str:
        calls.update(repo_id=repo_id, revision=revision)
        return str(tmp_path)

    fake_model = SimpleNamespace(
        cfg=SimpleNamespace(
            hidden_dim=BENCHMARK_INPUT_DIM,
            num_features=BENCHMARK_NUM_FEATURES,
            hierarchy=SimpleNamespace(scale_children_by_parent=False),
        )
    )

    def fake_load(path, device: str):
        calls.update(path=path, device=device)
        return fake_model

    monkeypatch.setattr(sweep_module, "snapshot_download", fake_download)
    monkeypatch.setattr(SyntheticModel, "load_from_disk", fake_load)

    loaded, snapshot = load_benchmark_model(config, "cpu")

    assert loaded is fake_model
    assert snapshot == tmp_path
    assert calls["repo_id"] == BENCHMARK_MODEL_ID
    assert calls["revision"] == BENCHMARK_REVISION
    assert calls["device"] == "cpu"


def test_native_training_checkpoint_round_trip(monkeypatch, tmp_path) -> None:
    config = _small_config(monkeypatch)
    spec = next(spec for spec in build_specs(config) if spec.method == "vgsae")
    model = build_model(config, spec)
    path = save_checkpoint(
        tmp_path / "last.pt",
        model=model,
        config=config,
        spec=spec,
        step=2,
        n_training_samples=12,
        loss=1.25,
        metadata={"sentinel": True},
    )

    loaded, payload = load_checkpoint(path)

    assert isinstance(loaded, VGTrainingSAE)
    assert payload["run_spec"] == spec.to_dict()
    assert payload["metadata"]["sentinel"] is True
    assert payload["n_training_samples"] == 12
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)


def test_learned_training_checkpoint_round_trip_preserves_mode(
    monkeypatch, tmp_path
) -> None:
    config = _small_config(monkeypatch)
    raw = config.to_dict()
    raw["training"]["beta_mode"] = "learned"
    config = SynthSAEBenchSweepConfig.from_dict(raw)
    spec = next(spec for spec in build_specs(config) if spec.method == "vgsae")
    model = build_model(config, spec)
    assert model.core.log_beta is not None
    path = save_checkpoint(
        tmp_path / "learned.pt",
        model=model,
        config=config,
        spec=spec,
        step=0,
        n_training_samples=4,
        loss=1.0,
    )

    loaded, payload = load_checkpoint(path)

    assert payload["sweep_config"]["training"]["beta_mode"] == "learned"
    assert loaded.cfg.beta_mode == "learned"
    assert loaded.core.log_beta is not None


def test_final_learned_beta_precision_reads_post_update_model_state(
    monkeypatch,
) -> None:
    config = _small_config(monkeypatch)
    raw = config.to_dict()
    raw["training"]["beta_mode"] = "learned"
    config = SynthSAEBenchSweepConfig.from_dict(raw)
    spec = next(spec for spec in build_specs(config) if spec.method == "vgsae")
    model = build_model(config, spec)
    assert model.core.log_beta is not None
    with torch.no_grad():
        model.core.log_beta.fill_(math.log(3.25))

    actual = _final_beta_precision(
        config,
        spec,
        model,
        [{"beta_precision": 1.0}],
    )

    assert actual == pytest.approx(3.25)


def test_synth_eval_summary_retains_one_beta_mode(tmp_path, monkeypatch) -> None:
    run_dirs = []
    config = _small_config(monkeypatch)
    for control_value in (0.4, 0.8):
        run_dir = tmp_path / "runs" / "vgsae" / f"vg-{control_value}"
        run_dirs.append(run_dir)
        row = {
            "run_id": run_dir.name,
            "seed": 0,
            "method": "vgsae",
            "method_label": "VG-SAE",
            "beta_mode": "learned",
            "control_name": "gamma",
            "control_value": control_value,
            "rho_model": control_value / 10,
            "true_l0": 2.0,
            "train_source_fingerprint": "source",
            "train_pipeline_fingerprint": "pipeline",
        }
        spec = next(
            spec
            for spec in build_specs(config)
            if spec.method == "vgsae" and spec.control_value == control_value
        ) if control_value in config.controls["vgsae"] else SynthSAEBenchRunSpec(
            method="vgsae",
            control_name="gamma",
            control_value=control_value,
            seed=0,
            init_seed=50_000,
            calibration_seed=20_000,
            train_stream_seed=30_000,
            eval_stream_seed=40_000,
        )
        write_json(
            run_dir / "config.json",
            {"sweep_config": config.to_dict(), "spec": spec.to_dict()},
        )
        write_json(run_dir / "eval" / "last" / "metrics.json", row)
        write_json(
            run_dir / "eval" / "last" / "status.json",
            {
                "eval_fingerprint": "eval",
                "eval_provenance": {"source_fingerprint": "eval-source"},
            },
        )
        write_rows(
            run_dir / "training_history.csv",
            [{"method": "vgsae", "run_id": run_dir.name, "step": 0}],
        )
    monkeypatch.setattr(synth_eval_runner, "_write_data_preview", lambda *_: None)

    synth_eval_runner._aggregate(tmp_path, run_dirs, run_dirs)

    summary = json.loads(
        (tmp_path / "summary" / "last" / "summary.json").read_text()
    )
    assert summary["beta_mode"] == "learned"
    assert "beta_mode" in (
        tmp_path / "summary" / "last" / "final_metrics_seed_mean.csv"
    ).read_text().splitlines()[0]


def test_synth_eval_rejects_mixed_beta_modes(tmp_path) -> None:
    run_dirs = []
    for index, beta_mode in enumerate(("profiled", "learned")):
        run_dir = tmp_path / "runs" / "vgsae" / f"vg-{index}"
        run_dirs.append(run_dir)
        write_json(
            run_dir / "eval" / "last" / "metrics.json",
            {
                "run_id": run_dir.name,
                "seed": 0,
                "method": "vgsae",
                "method_label": "VG-SAE",
                "beta_mode": beta_mode,
                "control_name": "gamma",
                "control_value": float(index),
                "rho_model": 0.1,
                "train_source_fingerprint": "source",
                "train_pipeline_fingerprint": "pipeline",
            },
        )

    with pytest.raises(ValueError, match="different VG beta modes"):
        synth_eval_runner._aggregate(tmp_path, run_dirs, run_dirs)


def test_rolling_checkpoint_restores_trainer_and_exact_stream_rng(
    monkeypatch, tmp_path
) -> None:
    config = _small_config(monkeypatch)
    spec = next(spec for spec in build_specs(config) if spec.method == "topk")

    def make_synthetic() -> SyntheticModel:
        return SyntheticModel(
            SyntheticModelConfig(
                num_features=6,
                hidden_dim=3,
                firing_probability=ConstantFiringProbabilityConfig(0.4),
                mean_firing_magnitudes=1.0,
                std_firing_magnitudes=0.0,
                bias=False,
                seed=None,
            )
        )

    with temporary_seed_for_device(spec.init_seed, "cpu"):
        model = build_model(config, spec)
        synthetic = make_synthetic()
    trainer = _trainer(config, spec, model, synthetic, "cpu")
    assert trainer.cfg.logger.log_to_wandb is False
    resume_path = tmp_path / "resume.pt"
    with temporary_seed_for_device(spec.train_stream_seed, "cpu"):
        output = trainer.step(next(trainer.data_provider))
        trainer.n_training_steps += 1
        saved_model = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        _save_resume_checkpoint(
            resume_path,
            trainer=trainer,
            run_fingerprint="fingerprint",
            history=[{"step": 0, "loss": 1.0}],
            last_loss=float(output.loss.detach()),
            elapsed_seconds=3.5,
            device="cpu",
        )
        expected_next_batch = next(trainer.data_provider).clone()

    with temporary_seed_for_device(spec.init_seed, "cpu"):
        resumed_model = build_model(config, spec)
        resumed_synthetic = make_synthetic()
    resumed_trainer = _trainer(
        config, spec, resumed_model, resumed_synthetic, "cpu"
    )
    history, last_loss, elapsed, cpu_rng, device_rng = _load_resume_checkpoint(
        resume_path,
        trainer=resumed_trainer,
        run_fingerprint="fingerprint",
        device="cpu",
    )

    assert history == [{"step": 0, "loss": 1.0}]
    assert last_loss == pytest.approx(float(output.loss.detach()))
    assert elapsed == pytest.approx(3.5)
    assert device_rng is None
    assert resumed_trainer.n_training_steps == 1
    assert resumed_trainer.n_training_samples == 4
    for name, value in saved_model.items():
        assert torch.equal(resumed_model.state_dict()[name], value)
    with temporary_seed_for_device(
        spec.train_stream_seed, "cpu", cpu_rng_state=cpu_rng
    ):
        actual_next_batch = next(resumed_trainer.data_provider)
    assert torch.equal(actual_next_batch, expected_next_batch)
    with pytest.raises(ValueError, match="cannot change RNG device type"):
        _load_resume_checkpoint(
            resume_path,
            trainer=resumed_trainer,
            run_fingerprint="fingerprint",
            device="cuda:0",
        )


def test_l1_resume_is_bit_exact_during_coefficient_warmup(
    monkeypatch, tmp_path
) -> None:
    config = _small_config(monkeypatch)
    spec = next(spec for spec in build_specs(config) if spec.method == "l1")

    def make_synthetic() -> SyntheticModel:
        return SyntheticModel(
            SyntheticModelConfig(
                num_features=6,
                hidden_dim=3,
                firing_probability=ConstantFiringProbabilityConfig(0.4),
                mean_firing_magnitudes=1.0,
                std_firing_magnitudes=0.0,
                bias=False,
                seed=None,
            )
        )

    def fresh_trainer():
        with temporary_seed_for_device(spec.init_seed, "cpu"):
            model = build_model(config, spec)
            synthetic = make_synthetic()
        return _trainer(config, spec, model, synthetic, "cpu")

    uninterrupted = fresh_trainer()
    with temporary_seed_for_device(spec.train_stream_seed, "cpu"):
        while uninterrupted.n_training_samples < config.data.n_train:
            uninterrupted.maybe_reset_sparsity()
            uninterrupted.step(next(uninterrupted.data_provider))
            uninterrupted.n_training_steps += 1

    partial = fresh_trainer()
    resume_path = tmp_path / "l1_resume.pt"
    with temporary_seed_for_device(spec.train_stream_seed, "cpu"):
        partial.maybe_reset_sparsity()
        first_output = partial.step(next(partial.data_provider))
        partial.n_training_steps += 1
        _save_resume_checkpoint(
            resume_path,
            trainer=partial,
            run_fingerprint="l1",
            history=[],
            last_loss=float(first_output.loss.detach()),
            elapsed_seconds=0.0,
            device="cpu",
        )

    resumed = fresh_trainer()
    _, _, _, cpu_rng, _ = _load_resume_checkpoint(
        resume_path,
        trainer=resumed,
        run_fingerprint="l1",
        device="cpu",
    )
    with temporary_seed_for_device(
        spec.train_stream_seed, "cpu", cpu_rng_state=cpu_rng
    ):
        while resumed.n_training_samples < config.data.n_train:
            resumed.maybe_reset_sparsity()
            resumed.step(next(resumed.data_provider))
            resumed.n_training_steps += 1

    assert resumed.coefficient_schedulers.keys() == uninterrupted.coefficient_schedulers.keys()
    coefficient_name = next(iter(resumed.coefficient_schedulers))
    assert resumed.coefficient_schedulers[coefficient_name].value == pytest.approx(
        uninterrupted.coefficient_schedulers[coefficient_name].value
    )
    for name, value in uninterrupted.sae.state_dict().items():
        assert torch.equal(resumed.sae.state_dict()[name], value), name


def test_streaming_evaluator_matches_official_core_metrics_and_caps_cache(
    monkeypatch,
) -> None:
    config = _small_config(monkeypatch)
    config.data = SynthSAEBenchDataConfig(
        input_dim=3,
        ground_truth_num_features=6,
        sae_width=5,
        n_train=12,
        n_test=8,
    )
    spec = next(spec for spec in build_specs(config) if spec.method == "topk")
    with temporary_seed_for_device(spec.init_seed, "cpu"):
        model = build_model(config, spec)
        synthetic = SyntheticModel(
            SyntheticModelConfig(
                num_features=6,
                hidden_dim=3,
                firing_probability=ConstantFiringProbabilityConfig(0.4),
                mean_firing_magnitudes=1.0,
                std_firing_magnitudes=0.0,
                bias=False,
                seed=None,
            )
        )

    inference = to_inference_sae(model, fold_decoder_norm=True)
    with temporary_seed_for_device(spec.eval_stream_seed, "cpu"):
        official = eval_sae_on_synthetic_data(
            inference,
            synthetic.feature_dict,
            synthetic.activation_generator,
            num_samples=config.data.n_test,
            batch_size=config.training.batch_size,
        )
    row, cache = evaluate_model(model, synthetic, config, spec)

    assert row["n_evaluation_samples"] == 8
    assert row["sae_l0"] == pytest.approx(official.sae_l0)
    assert row["true_l0"] == pytest.approx(official.true_l0)
    assert row["explained_variance"] == pytest.approx(
        official.explained_variance, abs=1.0e-6
    )
    assert row["mcc"] == pytest.approx(official.mcc, abs=1.0e-6)
    assert row["uniqueness"] == pytest.approx(official.uniqueness, abs=1.0e-6)
    assert row["classification_precision"] == pytest.approx(
        official.classification.precision, abs=1.0e-6
    )
    assert row["classification_recall"] == pytest.approx(
        official.classification.recall, abs=1.0e-6
    )
    assert row["classification_f1"] == pytest.approx(
        official.classification.f1_score, abs=1.0e-6
    )
    assert row["classification_accuracy"] == pytest.approx(
        official.classification.accuracy, abs=1.0e-6
    )
    assert cache["mask"].shape == (config.training.heatmap_samples, 5)
    assert cache["true_support"].shape == cache["mask"].shape
    assert cache["matched_true_idx"].shape == (5,)
    assert np.asarray(cache["preview_sample_count"]).item() == 3


def test_vg_streaming_eval_records_hard_expected_and_posterior_diagnostics(
    monkeypatch,
) -> None:
    config = _small_config(monkeypatch)
    spec = next(spec for spec in build_specs(config) if spec.method == "vgsae")
    with temporary_seed_for_device(spec.init_seed, "cpu"):
        model = build_model(config, spec)
        synthetic = SyntheticModel(
            SyntheticModelConfig(
                num_features=6,
                hidden_dim=3,
                firing_probability=ConstantFiringProbabilityConfig(0.4),
                mean_firing_magnitudes=1.0,
                std_firing_magnitudes=0.0,
                bias=False,
                seed=None,
            )
        )

    row, cache = evaluate_model(model, synthetic, config, spec)

    assert row["vg_expected_l0"] == pytest.approx(row["expected_l0"])
    assert row["vg_expected_to_hard_l0_ratio"] >= 0.0
    assert row["vg_posterior_probability_q10"] <= row[
        "vg_posterior_probability_q90"
    ]
    assert cache["posterior_probability"].shape == (
        config.training.heatmap_samples,
        config.data.sae_width,
    )
