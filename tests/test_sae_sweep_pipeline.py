import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.create_experiment_notebooks as notebook_generator
from src.sae_model import VGSAEConfig, VariationalGarroteSAE
from src.sae_sweep import (
    METHOD_ORDER,
    RunSpec,
    SweepConfig,
    SyntheticDataConfig,
    TrainingConfig,
    build_model,
    build_specs,
    default_sweep_dir,
    default_sweep_config,
    load_checkpoint,
    save_checkpoint,
    sweep_experiment_id,
)
from src.sae_train import fit_sae
from runs._sweep_io import aggregate_csv, write_json, write_rows
from runs.run_CustomData_sweep import (
    _preflight_wandb,
    _run_metadata,
    _wandb_run,
    configured_sweep,
    parse_args,
    selected_specs,
)
from runs.run_CustomData_sweep_eval import (
    _aggregate,
    _checkpoint_kinds,
    _training_ready,
    evaluate_one,
    parse_args as parse_eval_args,
)


def test_default_sweep_uses_current_full_grid_and_paired_initialization() -> None:
    config = default_sweep_config()
    specs = build_specs(config)
    counts = {method: sum(spec.method == method for spec in specs) for method in METHOD_ORDER}

    assert config.data.input_dim == 128
    assert config.data.ground_truth_num_features == 1024
    assert config.data.sae_width == 1024
    assert config.data.n_train == 8196
    assert config.data.n_test == 1024
    assert config.data.support_density == pytest.approx(0.01)
    assert config.data.frequency_skew == pytest.approx(0.5)
    assert config.data.amplitude_mode == "exponential"
    assert config.training.beta_mode == "profiled"
    assert len(specs) == 273
    assert counts == {
        "vgsae": 33,
        "l1": 16,
        "topk": 128,
        "batchtopk": 41,
        "jumprelu": 32,
        "gated": 23,
    }
    for method_index, method in enumerate(METHOD_ORDER):
        assert {spec.init_seed for spec in specs if spec.method == method} == {
            100_000 + method_index
        }


def test_config_derived_experiment_id_and_default_directory(tmp_path: Path) -> None:
    config = SweepConfig(
        data=SyntheticDataConfig(
            input_dim=32,
            ground_truth_num_features=256,
            sae_width=128,
            support_density=0.1,
        ),
        seeds=[1],
    )

    experiment_id = sweep_experiment_id(config)

    assert experiment_id == "stage1_beta_profiled_din32_gt256_sae128_sd010_seed1"
    assert default_sweep_dir(tmp_path, config) == (
        tmp_path / "outputs" / "runs" / experiment_id
    )


@pytest.mark.parametrize(
    ("density", "token"), [(0.1, "010"), (0.05, "005"), (0.125, "0125")]
)
def test_experiment_id_density_and_multiple_seed_tokens(
    density: float, token: str
) -> None:
    config = SweepConfig(
        data=SyntheticDataConfig(
            input_dim=32,
            ground_truth_num_features=256,
            sae_width=128,
            support_density=density,
        ),
        seeds=[0, 2, 5],
    )

    assert sweep_experiment_id(config).endswith(f"_sd{token}_seeds0-2-5")


def test_current_full_and_fast_experiment_ids_are_distinct() -> None:
    assert sweep_experiment_id(default_sweep_config()) == (
        "stage1_beta_profiled_din128_gt1024_sae1024_sd001_seed0"
    )
    assert sweep_experiment_id(default_sweep_config(fast=True)) == (
        "stage1_fast_beta_profiled_din128_gt1024_sae1024_sd001_seed0"
    )


@pytest.mark.parametrize(
    ("amplitude_mode", "frequency_skew", "expected_tokens", "absent_tokens"),
    [
        ("constant", 0.5, ("_ampconstant",), ("_fs0",)),
        ("uniform", 0.5, ("_ampuniform",), ("_fs0",)),
        ("exponential", 0.0, ("_fs0",), ("_ampconstant", "_ampuniform")),
        ("constant", 0.0, ("_ampconstant", "_fs0"), ()),
        ("uniform", 0.0, ("_ampuniform", "_fs0"), ()),
    ],
)
def test_ablation_axes_are_encoded_in_experiment_id(
    amplitude_mode: str,
    frequency_skew: float,
    expected_tokens: tuple[str, ...],
    absent_tokens: tuple[str, ...],
) -> None:
    config = default_sweep_config(fast=True)
    config.data.amplitude_mode = amplitude_mode
    config.data.frequency_skew = frequency_skew

    experiment_id = sweep_experiment_id(config)

    assert all(token in experiment_id for token in expected_tokens)
    assert all(token not in experiment_id for token in absent_tokens)


def test_wandb_credentials_are_loaded_only_from_local_environment(
    tmp_path: Path, monkeypatch
) -> None:
    from runs.run_CustomData_sweep import _load_project_env

    env_path = tmp_path / ".env"
    env_path.write_text("WANDB_API_KEY=from-file\n")
    monkeypatch.setattr("runs.run_CustomData_sweep.PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    _load_project_env()
    assert os.environ["WANDB_API_KEY"] == "from-file"

    monkeypatch.setenv("WANDB_API_KEY", "from-shell")
    _load_project_env()
    assert os.environ["WANDB_API_KEY"] == "from-shell"


def test_wandb_preflight_verifies_credentials_before_workers(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("wandb.login", lambda **kwargs: calls.append(kwargs) or True)

    _preflight_wandb()

    assert calls == [{"verify": True, "force": True}]


def test_wandb_preflight_hides_sdk_error_details(monkeypatch) -> None:
    def fail_login(**_kwargs) -> bool:
        raise ValueError("sensitive SDK detail")

    monkeypatch.setattr("wandb.login", fail_login)

    with pytest.raises(RuntimeError, match="authentication preflight failed") as error:
        _preflight_wandb()

    assert "sensitive SDK detail" not in str(error.value)


def test_wandb_is_forced_online_with_filterable_sweep_identity(
    tmp_path: Path, monkeypatch
) -> None:
    config = SweepConfig(
        experiment_name="stage1_custom_baseline_fast",
        data=SyntheticDataConfig(
            input_dim=3,
            ground_truth_num_features=7,
            sae_width=5,
            n_train=8,
            n_test=4,
            support_density=0.2,
            frequency_skew=0.0,
            amplitude_mode="uniform",
            amplitude_scale=1.7,
        ),
        training=TrainingConfig(train_steps=2, history_every=1),
        seeds=[2],
        methods=["vgsae"],
        controls={"vgsae": [0.5]},
    )
    spec = build_specs(config)[0]
    sweep_dir = tmp_path / "custom-output"
    (sweep_dir / "manifest.json").parent.mkdir(parents=True)
    (sweep_dir / "manifest.json").write_text("{}")
    run_dir = sweep_dir / "runs" / "vgsae" / spec.run_id
    captured = {}
    sentinel = object()
    monkeypatch.setattr(
        "wandb.init", lambda **kwargs: captured.update(kwargs) or sentinel
    )

    result = _wandb_run({"sweep_config": config.to_dict()}, spec, run_dir)

    exp_id = sweep_experiment_id(config)
    assert result is sentinel
    assert captured["group"] == "custom-output"
    assert captured["job_type"] == "vgsae"
    assert captured["mode"] == "online"
    assert captured["force"] is True
    assert captured["tags"] == [
        "stage:stage1_custom_baseline_fast",
        "method:vgsae",
        "beta_mode:profiled",
        "amplitude_mode:uniform",
        "frequency_skew:0",
    ]
    assert captured["config"]["exp_id"] == exp_id
    assert captured["config"]["stage"] == "stage1_custom_baseline_fast"
    assert captured["config"]["sweep_root"] == "custom-output"
    assert captured["config"]["method"] == "vgsae"
    assert captured["config"]["beta_mode"] == "profiled"
    assert captured["config"]["amplitude_mode"] == "uniform"
    assert captured["config"]["amplitude_scale"] == pytest.approx(1.7)
    assert captured["config"]["frequency_skew"] == pytest.approx(0.0)
    metadata = _run_metadata(
        config,
        spec,
        "cpu",
        {"source_fingerprint": "source", "pipeline_fingerprint": "pipeline"},
    )
    assert metadata["exp_id"] == exp_id
    assert metadata["beta_mode"] == "profiled"
    assert metadata["amplitude_mode"] == "uniform"
    assert metadata["amplitude_scale"] == pytest.approx(1.7)
    assert metadata["frequency_skew"] == pytest.approx(0.0)


@pytest.mark.parametrize("option", ["--no-wandb", "--wandb-mode=offline"])
def test_custom_sweep_rejects_wandb_bypass_options(
    option: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_CustomData_sweep.py", option])

    with pytest.raises(SystemExit):
        parse_args()


def test_custom_sweep_cli_accepts_beta_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_CustomData_sweep.py", "--beta-mode", "learned"],
    )

    assert parse_args().beta_mode == "learned"


def test_custom_sweep_cli_accepts_amplitude_and_frequency_ablation_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_CustomData_sweep.py",
            "--amplitude-mode",
            "constant",
            "--frequency-skew",
            "0",
        ],
    )

    args = parse_args()

    assert args.amplitude_mode == "constant"
    assert args.frequency_skew == pytest.approx(0.0)


def test_custom_sweep_cli_rejects_fixed_beta_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_CustomData_sweep.py", "--beta-mode", "fixed"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_custom_eval_cli_selects_mode_and_rejects_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_CustomData_sweep_eval.py", "--beta-mode", "learned"],
    )
    assert parse_eval_args().beta_mode == "learned"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_CustomData_sweep_eval.py", "--beta-mode", "fixed"],
    )
    with pytest.raises(SystemExit):
        parse_eval_args()


def test_custom_wandb_rejects_run_without_sweep_manifest(tmp_path: Path) -> None:
    config = default_sweep_config(fast=True)
    spec = build_specs(config)[0]

    with pytest.raises(ValueError, match="Cannot identify sweep root"):
        _wandb_run(
            {"sweep_config": config.to_dict()},
            spec,
            tmp_path / "runs" / spec.method / spec.run_id,
        )


def test_eval_defaults_to_both_checkpoint_kinds() -> None:
    assert _checkpoint_kinds(None) == ("last", "best")
    assert _checkpoint_kinds("last") == ("last",)
    assert _checkpoint_kinds("best") == ("best",)


def test_sweep_config_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = SweepConfig(
        experiment_name="test",
        data=SyntheticDataConfig(input_dim=3, n_features=4, n_train=8, n_test=4),
        training=TrainingConfig(train_steps=2, history_every=1),
        seeds=[2],
        methods=["vgsae"],
        controls={"vgsae": [0.5]},
    )
    restored = SweepConfig.from_dict(config.to_dict())
    spec = build_specs(restored)[0]
    model = VariationalGarroteSAE(
        VGSAEConfig(input_dim=3, n_latents=4, lambda_sparsity=0.5)
    )
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        config=restored,
        spec=spec,
        checkpoint_kind="last",
        step=1,
        loss=1.25,
        metadata={"train_device": "cpu"},
    )

    loaded, payload = load_checkpoint(checkpoint)
    assert payload["checkpoint_kind"] == "last"
    assert payload["metadata"] == {"train_device": "cpu"}
    assert RunSpec.from_dict(payload["run_spec"]) == spec
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)


def test_stage1_beta_mode_roundtrip_and_model_construction() -> None:
    config = SweepConfig(
        data=SyntheticDataConfig(
            input_dim=3,
            ground_truth_num_features=7,
            sae_width=5,
            n_train=8,
            n_test=4,
            support_density=0.2,
        ),
        training=TrainingConfig(
            train_steps=2,
            history_every=1,
            beta_mode="learned",
        ),
        methods=["vgsae"],
        controls={"vgsae": [0.5]},
    )

    restored = SweepConfig.from_dict(config.to_dict())
    model = build_model(restored, build_specs(restored)[0])

    assert restored.training.beta_mode == "learned"
    assert model.config.beta_mode == "learned"
    assert model.log_beta is not None
    assert sweep_experiment_id(restored).startswith("stage1_beta_learned_")


def test_stage1_rejects_unsupported_beta_mode() -> None:
    config = default_sweep_config(fast=True).to_dict()
    config["training"]["beta_mode"] = "fixed"

    with pytest.raises(ValueError, match="beta_mode"):
        SweepConfig.from_dict(config)

    with pytest.raises(ValueError, match="profiled or learned"):
        TrainingConfig(beta_mode="fixed")  # type: ignore[arg-type]


def test_stage1_eval_summary_and_csv_retain_beta_mode(tmp_path: Path) -> None:
    config = SweepConfig(
        data=SyntheticDataConfig(
            input_dim=3,
            ground_truth_num_features=7,
            sae_width=5,
            n_train=8,
            n_test=4,
            support_density=0.2,
        ),
        training=TrainingConfig(
            train_steps=2,
            history_every=1,
            beta_mode="learned",
        ),
        methods=["vgsae"],
        controls={"vgsae": [0.5]},
    )
    spec = build_specs(config)[0]
    run_dir = tmp_path / "runs" / "vgsae" / spec.run_id
    write_json(
        run_dir / "config.json",
        {"sweep_config": config.to_dict(), "spec": spec.to_dict()},
    )
    write_json(
        run_dir / "eval" / "last" / "metrics.json",
        {
            "run_id": spec.run_id,
            "seed": spec.seed,
            "method": spec.method,
            "method_label": "VG-SAE",
            "control_name": spec.control_name,
            "control_value": spec.control_value,
            "input_dim": config.data.input_dim,
            "ground_truth_num_features": config.data.ground_truth_num_features,
            "sae_width": config.data.sae_width,
            "support_density": config.data.support_density,
            "beta_mode": config.training.beta_mode,
            "rho_model": 0.1,
            "train_source_fingerprint": "train-source",
            "train_pipeline_fingerprint": "train-pipeline",
        },
    )
    write_json(
        run_dir / "eval" / "last" / "status.json",
        {
            "eval_fingerprint": "eval-pipeline",
            "eval_provenance": {"source_fingerprint": "eval-source"},
        },
    )
    write_rows(
        run_dir / "training_history.csv",
        [
            {
                "method": spec.method,
                "run_id": spec.run_id,
                "step": 0,
                "beta_mode": config.training.beta_mode,
            }
        ],
    )

    _aggregate(tmp_path, [run_dir], [run_dir], "last")

    summary = json.loads((tmp_path / "summary" / "last" / "summary.json").read_text())
    assert summary["beta_mode"] == "learned"
    assert "beta_mode" in (
        tmp_path / "summary" / "last" / "final_metrics.csv"
    ).read_text().splitlines()[0]
    assert "beta_mode" in (
        tmp_path / "summary" / "last" / "final_metrics_seed_mean.csv"
    ).read_text().splitlines()[0]


def test_stage1_aggregation_rejects_unsupported_beta_mode(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "vgsae" / "vg"
    write_json(
        run_dir / "eval" / "last" / "metrics.json",
        {
            "beta_mode": "fixed",
            "train_source_fingerprint": "source",
            "train_pipeline_fingerprint": "pipeline",
        },
    )
    with pytest.raises(ValueError, match="unsupported beta_mode.*fixed"):
        _aggregate(tmp_path, [run_dir], [run_dir], "last")

    write_rows(run_dir / "training_history.csv", [{"beta_mode": "fixed"}])
    with pytest.raises(ValueError, match="unsupported beta_mode.*fixed"):
        aggregate_csv(
            [run_dir],
            Path("training_history.csv"),
            tmp_path / "summary" / "training_curves.csv",
        )


@pytest.mark.parametrize("beta_mode", ["profiled", "learned"])
def test_stage1_eval_records_checkpoint_beta_precision(
    tmp_path: Path, beta_mode: str
) -> None:
    config = SweepConfig(
        data=SyntheticDataConfig(
            input_dim=3,
            ground_truth_num_features=7,
            sae_width=5,
            n_train=8,
            n_test=4,
            support_density=0.2,
        ),
        training=TrainingConfig(
            train_steps=1,
            history_every=1,
            beta_mode=beta_mode,  # type: ignore[arg-type]
        ),
        methods=["vgsae"],
        controls={"vgsae": [0.5]},
    )
    spec = build_specs(config)[0]
    model = build_model(config, spec)
    run_dir = tmp_path / "runs" / "vgsae" / spec.run_id
    provenance = {
        "source_fingerprint": "source",
        "pipeline_fingerprint": "pipeline",
    }
    bundle = {
        "fingerprint": "fingerprint",
        "sweep_config": config.to_dict(),
        "spec": spec.to_dict(),
    }
    write_json(run_dir / "config.json", bundle)
    write_json(
        run_dir / "train_status.json",
        {"state": "complete", "fingerprint": "fingerprint"},
    )
    write_rows(run_dir / "training_history.csv", [{"step": 0}])
    write_json(run_dir / "training_summary.json", {})
    save_checkpoint(
        run_dir / "checkpoints" / "last.pt",
        model=model,
        config=config,
        spec=spec,
        checkpoint_kind="last",
        step=0,
        loss=1.0,
        metadata={
            "train_fingerprint": "fingerprint",
            "train_device": "cpu",
            "train_provenance": provenance,
        },
    )

    evaluate_one(run_dir, "last", "cpu", True)

    metrics = json.loads(
        (run_dir / "eval" / "last" / "metrics.json").read_text()
    )
    assert metrics["beta_mode"] == beta_mode
    assert metrics["final_beta_precision"] > 0.0


def test_fit_sae_tracks_best_snapshot_and_streams_history() -> None:
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(12, 3, generator=generator)
    model = VariationalGarroteSAE(VGSAEConfig(input_dim=3, n_latents=5))
    callbacks = []
    result = fit_sae(
        model,
        x,
        max_steps=3,
        batch_size=6,
        history_every=1,
        seed=7,
        history_callback=callbacks.append,
    )

    assert callbacks == result.history
    assert result.best_loss == min(row["loss"] for row in result.history)
    assert result.best_step in {0, 1, 2}
    assert result.best_state_dict is not None
    assert all(
        not value.is_cuda
        for value in result.best_state_dict.values()
        if isinstance(value, torch.Tensor)
    )


def test_notebook_10_is_plot_only_and_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(notebook_generator, "NOTEBOOK_DIR", tmp_path)
    notebook_generator.write_notebooks()
    notebook = json.loads(
        (tmp_path / "10_exp07_parallel_sweep_results.ipynb").read_text()
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_CustomData_sweep.py" in source
    assert "run_CustomData_sweep_eval.py" in source
    assert "fit_sae(" not in source
    assert "evaluate_model(" not in source
    assert 'VGSAE_CHECKPOINT_KIND", "last"' in source
    assert 'VGSAE_DENSITY_MODE", "reported"' in source
    assert '"target_model_density_empirical"' in source
    assert "density_mode=DENSITY_MODE" in source
    assert "plot_vg_posterior_diagnostics" in source
    assert "VGSAE_BASELINE_SWEEP_DIR" in source
    assert "load_comparison_results" in source
    assert "run_roots=RUN_ROOTS" in source
    assert "vg_posterior_columns.issubset(final_df.columns)" in source
    for figure_name in (
        "data_overview.png",
        "reconstruction_metrics.png",
        "recovery_metrics.png",
        "support_metrics.png",
        "sparsity_diagnostics.png",
        "training_curves.png",
        "mask_heatmaps.png",
        "vg_posterior_diagnostics.png",
        "stage1_style_reconstruction_metrics.png",
        "stage1_style_recovery_metrics.png",
        "stage1_style_support_metrics.png",
        "stage1_style_sparsity_diagnostics.png",
    ):
        assert figure_name in source
    assert 'metric_suite="stage1"' in source
    assert "Stage 2's independently best-matched ground-truth feature" in source
    assert "Mean Correlation Coefficient, not Matthews correlation" in source
    assert "Expected L0 / $d_\\mathrm{sae}$" in source
    assert "do not reproduce Stage 1's rectangular-Hungarian-union semantics" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-10-cell-{index}", "exec")
    vg_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "vg_posterior_columns =" in "".join(cell.get("source", []))
    )
    exec(vg_cell, {"final_df": SimpleNamespace(columns=set())})

    class _Methods:
        def __eq__(self, value: str):
            assert value == "vgsae"
            return self

        @staticmethod
        def any() -> bool:
            return True

    class _SynthFrame:
        columns = {"vg_expected_explained_variance", "vg_expected_l0"}

        def __getitem__(self, key: str):
            assert key == "method"
            return _Methods()

    calls = []

    def fake_vg_plot(frame, **kwargs):
        calls.append((frame, kwargs))

    exec(
        vg_cell,
        {
            "FIGURE_DIR": tmp_path,
            "final_df": _SynthFrame(),
            "TARGET_MODEL_DENSITY": 0.1,
            "DENSITY_MODE": "hard",
            "plot_context": {"sae_width": 8},
            "plot_vg_posterior_diagnostics": fake_vg_plot,
            "plt": SimpleNamespace(show=lambda: None),
        },
    )
    assert len(calls) == 1
    assert isinstance(calls[0][0], _SynthFrame)
    assert calls[0][1] == {
        "target_model_density": 0.1,
        "sae_width": 8,
        "output_path": tmp_path / "vg_posterior_diagnostics.png",
        "density_mode": "hard",
    }


def test_method_filter_selects_tasks_without_narrowing_saved_config() -> None:
    args = SimpleNamespace(
        config=None,
        fast_dev_run=True,
        methods="vgsae",
        seeds=None,
        train_steps=None,
        history_every=None,
        beta_mode=None,
        amplitude_mode=None,
        frequency_skew=None,
    )
    config = configured_sweep(args)
    specs = selected_specs(config, args.methods)

    assert config.methods == list(METHOD_ORDER)
    assert len(specs) == 2
    assert {spec.method for spec in specs} == {"vgsae"}


def test_stage1_cli_overrides_data_axes_control_and_seed() -> None:
    args = SimpleNamespace(
        config=None,
        fast_dev_run=True,
        input_dim=3,
        ground_truth_num_features=7,
        sae_width=5,
        support_density=0.2,
        amplitude_mode="uniform",
        frequency_skew=0.0,
        seed=4,
        seeds=None,
        sparsity_controls=["vgsae=-1,0,1", "topk=1,3"],
        train_steps=None,
        history_every=None,
        beta_mode="learned",
    )

    config = configured_sweep(args)

    assert config.data.input_dim == 3
    assert config.data.ground_truth_num_features == 7
    assert config.data.sae_width == 5
    assert config.data.support_density == 0.2
    assert config.data.amplitude_mode == "uniform"
    assert config.data.frequency_skew == pytest.approx(0.0)
    assert config.seeds == [4]
    assert config.controls["vgsae"] == [-1.0, 0.0, 1.0]
    assert config.controls["topk"] == [1, 3]
    assert config.training.beta_mode == "learned"


def test_default_topk_grids_follow_overridden_sae_width() -> None:
    args = SimpleNamespace(
        config=None,
        fast_dev_run=False,
        input_dim=None,
        ground_truth_num_features=None,
        sae_width=5,
        support_density=None,
        seed=None,
        seeds=None,
        sparsity_controls=None,
        train_steps=None,
        history_every=None,
        beta_mode=None,
        amplitude_mode=None,
        frequency_skew=None,
    )

    config = configured_sweep(args)

    assert config.controls["topk"] == [1, 2, 3, 4, 5]
    assert config.controls["batchtopk"][-1] == 5.0


def test_evaluation_rejects_non_complete_training_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_json(run_dir / "config.json", {"fingerprint": "current"})
    write_json(
        run_dir / "train_status.json",
        {"state": "complete", "fingerprint": "current"},
    )
    checkpoint = run_dir / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    assert not _training_ready(run_dir, "last")
    (run_dir / "training_history.csv").write_text("step\n0\n")
    write_json(run_dir / "training_summary.json", {})
    assert _training_ready(run_dir, "last")

    write_json(
        run_dir / "train_status.json",
        {"state": "failed", "fingerprint": "current"},
    )
    assert not _training_ready(run_dir, "last")
