import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.sae_model import VGSAEConfig, VariationalGarroteSAE
from src.sae_sweep import (
    METHOD_ORDER,
    RunSpec,
    SweepConfig,
    SyntheticDataConfig,
    TrainingConfig,
    build_specs,
    default_sweep_dir,
    default_sweep_config,
    load_checkpoint,
    save_checkpoint,
    sweep_experiment_id,
)
from src.sae_train import fit_sae
from runs._sweep_io import write_json
from runs.run_saes_sweep import (
    _preflight_wandb,
    _run_metadata,
    _wandb_run,
    configured_sweep,
    selected_specs,
)
from runs.run_saes_sweep_eval import _checkpoint_kinds, _training_ready


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

    assert experiment_id == "stage1_din32_gt256_sae128_sd010_seed1"
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
        "stage1_din128_gt1024_sae1024_sd001_seed0"
    )
    assert sweep_experiment_id(default_sweep_config(fast=True)) == (
        "stage1_fast_din128_gt1024_sae1024_sd001_seed0"
    )


def test_wandb_credentials_are_loaded_only_from_local_environment(
    tmp_path: Path, monkeypatch
) -> None:
    from runs.run_saes_sweep import _load_project_env

    env_path = tmp_path / ".env"
    env_path.write_text("WANDB_API_KEY=from-file\n")
    monkeypatch.setattr("runs.run_saes_sweep.PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    _load_project_env()
    assert os.environ["WANDB_API_KEY"] == "from-file"

    monkeypatch.setenv("WANDB_API_KEY", "from-shell")
    _load_project_env()
    assert os.environ["WANDB_API_KEY"] == "from-shell"


def test_wandb_preflight_verifies_credentials_before_workers(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("wandb.login", lambda **kwargs: calls.append(kwargs) or True)

    _preflight_wandb("online")

    assert calls == [{"verify": True, "force": True}]


def test_wandb_preflight_hides_sdk_error_details(monkeypatch) -> None:
    def fail_login(**_kwargs) -> bool:
        raise ValueError("sensitive SDK detail")

    monkeypatch.setattr("wandb.login", fail_login)

    with pytest.raises(RuntimeError, match="authentication preflight failed") as error:
        _preflight_wandb("online")

    assert "sensitive SDK detail" not in str(error.value)


def test_wandb_logs_filterable_experiment_id(
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

    result = _wandb_run(
        {"sweep_config": config.to_dict()}, spec, run_dir, "offline"
    )

    exp_id = sweep_experiment_id(config)
    assert result is sentinel
    assert captured["group"] == "custom-output"
    assert captured["config"]["exp_id"] == exp_id
    metadata = _run_metadata(
        config,
        spec,
        "cpu",
        {"source_fingerprint": "source", "pipeline_fingerprint": "pipeline"},
    )
    assert metadata["exp_id"] == exp_id


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


def test_notebook_10_is_plot_only_and_compiles() -> None:
    notebook = json.loads(Path("notebooks/10_exp07_parallel_sweep_results.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_saes_sweep.py" in source
    assert "run_saes_sweep_eval.py" in source
    assert "fit_sae(" not in source
    assert "evaluate_model(" not in source
    assert "VGSAE_CHECKPOINT_KIND', 'last'" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-10-cell-{index}", "exec")


def test_method_filter_selects_tasks_without_narrowing_saved_config() -> None:
    args = SimpleNamespace(
        config=None,
        fast_dev_run=True,
        methods="vgsae",
        seeds=None,
        train_steps=None,
        history_every=None,
    )
    config = configured_sweep(args)
    specs = selected_specs(config, args.methods)

    assert config.methods == list(METHOD_ORDER)
    assert len(specs) == 2
    assert {spec.method for spec in specs} == {"vgsae"}


def test_stage1_cli_overrides_dimensions_density_control_and_seed() -> None:
    args = SimpleNamespace(
        config=None,
        fast_dev_run=True,
        input_dim=3,
        ground_truth_num_features=7,
        sae_width=5,
        support_density=0.2,
        seed=4,
        seeds=None,
        sparsity_controls=["vgsae=-1,0,1", "topk=1,3"],
        train_steps=None,
        history_every=None,
    )

    config = configured_sweep(args)

    assert config.data.input_dim == 3
    assert config.data.ground_truth_num_features == 7
    assert config.data.sae_width == 5
    assert config.data.support_density == 0.2
    assert config.seeds == [4]
    assert config.controls["vgsae"] == [-1.0, 0.0, 1.0]
    assert config.controls["topk"] == [1, 3]


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
