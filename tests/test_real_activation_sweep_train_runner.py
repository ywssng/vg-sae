from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import runs.run_RealActivation_sweep as runner
from src.real_activation_sweep import (
    GEMMA_LAYER12_TRAIN_TOKENS,
    PAPER_BATCHTOPK_K_GRIDS,
    RealActivationSweepConfig,
    build_specs,
    default_sweep_config,
)


def _args(**overrides) -> Namespace:
    values = {
        "config": None,
        "targets": "all",
        "output_root": None,
        "output_dir": None,
        "fast_dev_run": False,
        "methods": "all",
        "seed": None,
        "seeds": None,
        "sparsity_controls": None,
        "training_tokens": None,
        "eval_tokens": None,
        "downstream_eval_tokens": None,
        "batch_size": None,
        "history_every": None,
        "resume_every": None,
        "devices": "cpu",
        "max_per_device": 1,
        "force": False,
        "skip_eval": False,
        "worker": False,
        "run_dir": None,
        "device": "cpu",
    }
    values.update(overrides)
    return Namespace(**values)


def test_configured_sweeps_keep_target_specific_protocols() -> None:
    configs = runner.configured_sweeps(_args())
    assert [config.data.target_name for config in configs] == [
        "gemma-2-2b-layer5",
        "gemma-2-2b-layer12",
        "llama-3.2-1b-layer7",
    ]
    layer12 = configs[1]
    assert layer12.data.n_train_tokens == GEMMA_LAYER12_TRAIN_TOKENS
    assert layer12.controls["batchtopk"] == list(
        PAPER_BATCHTOPK_K_GRIDS["gemma-2-2b-layer12"]
    )
    assert layer12.training.beta_mode == "learned"
    assert layer12.seeds == [0]
    assert configs[0].seeds == configs[2].seeds == [0, 1, 2]


def test_fast_dev_run_is_one_complete_context_batch_per_selected_method() -> None:
    config = runner.configured_sweeps(
        _args(
            targets="gemma-2-2b-layer5",
            fast_dev_run=True,
            methods="vgsae,jumprelu",
        )
    )[0]
    assert config.data.n_train_tokens == 4_096
    assert config.data.eval_token_offset == 4_096
    assert config.data.n_eval_tokens == 4_096
    assert config.seeds == [0]
    assert config.methods == ["vgsae", "l1", "batchtopk", "jumprelu"]
    assert {name: len(values) for name, values in config.controls.items()} == {
        "vgsae": 1,
        "l1": 1,
        "batchtopk": 1,
        "jumprelu": 1,
    }
    assert [spec.method for spec in runner.selected_specs(config, "vgsae,jumprelu")] == [
        "vgsae",
        "jumprelu",
    ]


def test_control_override_validates_batchtopk_integrality() -> None:
    with pytest.raises(ValueError, match="integers"):
        runner.configured_sweeps(
            _args(
                targets="llama-3.2-1b-layer7",
                sparsity_controls=["batchtopk=10.5"],
            )
        )


def test_prepare_runs_records_pinned_activation_identity(tmp_path: Path) -> None:
    config = default_sweep_config("llama-3.2-1b-layer7")
    config.seeds = [0]
    config.methods = ["batchtopk"]
    config.controls = {"batchtopk": [10]}
    sweep_dir = tmp_path / "sweep"
    tasks, run_dirs = runner.prepare_runs(
        sweep_dir, config, build_specs(config), force=False
    )
    assert len(tasks) == len(run_dirs) == 1
    manifest = runner.read_json(sweep_dir / "manifest.json")
    bundle = runner.read_json(run_dirs[0] / "config.json")
    assert manifest["activation_identity"]["model_revision"] == (
        config.data.model_revision
    )
    assert bundle["activation_identity"]["dataset_revision"] == (
        config.data.dataset_revision
    )
    assert bundle["spec"]["method"] == "batchtopk"
    assert runner.read_json(run_dirs[0] / "train_status.json")["state"] == "queued"


def test_completed_manifest_training_runs_keeps_prior_incremental_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = tmp_path / "runs" / "batchtopk" / "prior"
    selected = tmp_path / "runs" / "vgsae" / "selected"
    incomplete = tmp_path / "runs" / "l1" / "incomplete"
    runner.write_json(
        tmp_path / "manifest.json",
        {
            "runs": [
                {"relative_dir": str(path.relative_to(tmp_path))}
                for path in (prior, selected, incomplete)
            ]
        },
    )
    for path in (prior, selected, incomplete):
        runner.write_json(path / "config.json", {"fingerprint": path.name})
    monkeypatch.setattr(
        runner,
        "_is_complete",
        lambda run_dir, fingerprint: (
            run_dir != incomplete and fingerprint == run_dir.name
        ),
    )

    assert runner._completed_manifest_training_runs(tmp_path) == [prior, selected]


def test_main_launches_eval_after_all_training_artifacts_are_complete(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_sweep_config("llama-3.2-1b-layer7")
    sweep_dir = tmp_path / "sweep"
    run_dir = sweep_dir / "runs" / "vgsae" / "one"
    launched: list[tuple[Path, str]] = []

    monkeypatch.setattr(runner, "_load_project_env", lambda: None)
    monkeypatch.setattr(runner, "configured_sweeps", lambda args: [config])
    monkeypatch.setattr(
        runner,
        "prepare_runs",
        lambda *args, **kwargs: ([], [run_dir]),
    )
    monkeypatch.setattr(runner, "_sweep_dir", lambda *args, **kwargs: sweep_dir)
    monkeypatch.setattr(runner, "_is_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        runner,
        "aggregate_csv",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "_completed_manifest_training_runs",
        lambda _path: [run_dir],
    )
    monkeypatch.setattr(
        runner,
        "read_json",
        lambda path: {"fingerprint": "ready"},
    )
    monkeypatch.setattr(
        runner,
        "_launch_evaluation",
        lambda path, **kwargs: launched.append((path, kwargs["methods"])) or 0,
    )

    assert runner.main(_args(targets="llama-3.2-1b-layer7")) == 0
    assert launched == [(sweep_dir, "all")]


def test_main_incremental_subset_ignores_unselected_incomplete_manifest_runs(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_sweep_config("llama-3.2-1b-layer7")
    sweep_dir = tmp_path / "sweep"
    selected_spec = next(
        spec for spec in build_specs(config) if spec.method == "vgsae"
    )
    selected = sweep_dir / "runs" / "vgsae" / selected_spec.run_id
    unrelated = sweep_dir / "runs" / "l1" / "incomplete"
    launched: list[str] = []

    monkeypatch.setattr(runner, "_load_project_env", lambda: None)
    monkeypatch.setattr(runner, "configured_sweeps", lambda args: [config])
    monkeypatch.setattr(
        runner,
        "prepare_runs",
        lambda *args, **kwargs: ([], [selected, unrelated]),
    )
    monkeypatch.setattr(runner, "_sweep_dir", lambda *args, **kwargs: sweep_dir)
    monkeypatch.setattr(
        runner,
        "_is_complete",
        lambda run_dir, _fingerprint: run_dir == selected,
    )
    monkeypatch.setattr(runner, "aggregate_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_completed_manifest_training_runs",
        lambda _path: [selected],
    )
    monkeypatch.setattr(runner, "read_json", lambda path: {"fingerprint": "ready"})
    monkeypatch.setattr(
        runner,
        "_launch_evaluation",
        lambda _path, **kwargs: launched.append(kwargs["methods"]) or 0,
    )

    assert runner.main(_args(methods="vgsae")) == 0
    assert launched == ["vgsae"]
