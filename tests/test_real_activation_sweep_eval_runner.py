from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("sae_lens")

from runs._sweep_io import write_json, write_rows
import runs.run_RealActivation_sweep_eval as runner
from src.real_activation_sweep import (
    RealActivationRunSpec,
    RealActivationSweepConfig,
    RealActivationTrainingConfig,
    target_data_config,
)


def _small_config(*, method: str = "l1") -> RealActivationSweepConfig:
    return RealActivationSweepConfig(
        data=target_data_config(
            "gemma-2-2b-layer5",
            n_train_tokens=4_096,
            n_eval_tokens=4_096,
            n_downstream_eval_tokens=2_048,
        ),
        training=RealActivationTrainingConfig(
            batch_size=4_096,
            preview_tokens=7,
            autocast_data=False,
        ),
        methods=[method],
        controls={method: [1.25]},
    )


def _spec(method: str = "l1", *, seed: int = 0) -> RealActivationRunSpec:
    return RealActivationRunSpec(
        method=method,
        control_name="l1_coefficient" if method == "l1" else "lambda_sparsity",
        control_value=1.25,
        seed=seed,
        init_seed=50_000 + seed,
        calibration_seed=20_000 + seed,
        train_stream_seed=30_000 + seed,
        eval_stream_seed=40_000 + seed,
    )


def test_downstream_eval_uses_same_heldout_prefix_and_weighted_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    inference = SimpleNamespace(eval=lambda: None)
    monkeypatch.setattr(runner, "to_inference_sae", lambda *_args, **_kwargs: inference)
    monkeypatch.setattr(runner, "model_input_device", lambda _model: torch.device("cpu"))
    token_batches = [
        torch.zeros(1, 1_024, dtype=torch.long),
        torch.ones(1, 1_024, dtype=torch.long),
    ]
    calls: list[dict[str, object]] = []

    def fake_iter(data, **kwargs):
        assert data is config.data
        assert kwargs["start_token"] == config.data.eval_token_offset
        assert kwargs["n_tokens"] == 2_048
        yield from token_batches

    values = iter(
        [
            {
                "ce_loss_with_sae": torch.tensor([2.0, 4.0]),
                "ce_loss_without_sae": torch.tensor([1.0, 3.0]),
                "ce_loss_with_ablation": torch.tensor([5.0, 7.0]),
                "kl_div_with_sae": torch.tensor([1.0, 3.0]),
                "kl_div_with_ablation": torch.tensor([2.0, 6.0]),
            },
            {
                "ce_loss_with_sae": torch.tensor([6.0]),
                "ce_loss_without_sae": torch.tensor([5.0]),
                "ce_loss_with_ablation": torch.tensor([9.0]),
                "kl_div_with_sae": torch.tensor([5.0]),
                "kl_div_with_ablation": torch.tensor([10.0]),
            },
        ]
    )

    def fake_recons(*args, **kwargs):
        calls.append(kwargs)
        assert args[0] is inference
        return next(values)

    monkeypatch.setattr(runner, "iter_token_batches", fake_iter)
    monkeypatch.setattr(runner, "get_recons_loss", fake_recons)

    metrics = runner._evaluate_downstream(object(), object(), config)

    assert metrics["ce_loss_with_sae"] == pytest.approx(4.0)
    assert metrics["ce_loss_without_sae"] == pytest.approx(3.0)
    assert metrics["ce_loss_with_ablation"] == pytest.approx(7.0)
    assert metrics["ce_loss_score"] == pytest.approx(0.75)
    assert metrics["kl_div_with_sae"] == pytest.approx(3.0)
    assert metrics["kl_div_with_ablation"] == pytest.approx(6.0)
    assert metrics["kl_div_score"] == pytest.approx(0.5)
    assert metrics["n_downstream_evaluation_tokens"] == 2_048
    assert all(call["model_kwargs"] == {"prepend_bos": False} for call in calls)
    assert all(call["hook_name"] == config.data.hook_name for call in calls)


def test_normalized_scores_return_none_for_zero_reference_gap() -> None:
    assert runner._safe_normalized_score(1.0, 0.0) is None
    assert runner._safe_normalized_score(1.0, float("nan")) is None
    assert runner._safe_normalized_score(3.0, 4.0) == pytest.approx(0.75)


def test_evaluate_one_records_exact_budgets_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    spec = _spec()
    run_dir = tmp_path / spec.run_id
    checkpoint = run_dir / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    bundle = {
        "fingerprint": "train-fingerprint",
        "sweep_config": config.to_dict(),
        "spec": spec.to_dict(),
    }
    write_json(run_dir / "config.json", bundle)
    write_json(
        run_dir / "train_status.json",
        {"state": "complete", "fingerprint": "train-fingerprint"},
    )
    write_rows(run_dir / "training_history.csv", [{"step": 0}])
    write_json(run_dir / "training_summary.json", {"loss": 1.0})

    training_model = SimpleNamespace(core=None)
    payload = {
        "checkpoint_kind": "last",
        "run_spec": spec.to_dict(),
        "sweep_config": config.to_dict(),
        "step": 3,
        "n_training_tokens": 4_096,
        "loss": 0.25,
        "metadata": {
            "train_fingerprint": "train-fingerprint",
            "train_device": "cuda:7",
            "train_provenance": {
                "source_fingerprint": "train-source",
                "pipeline_fingerprint": "train-pipeline",
            },
        },
    }
    monkeypatch.setattr(runner, "activate_worker_device", lambda _device: None)
    monkeypatch.setattr(
        runner,
        "_stage3_runtime_provenance",
        lambda: {
            "source_fingerprint": "eval-source",
            "pipeline_fingerprint": "eval-pipeline",
        },
    )
    monkeypatch.setattr(runner, "load_checkpoint", lambda *_args: (training_model, payload))
    language_model = object()
    monkeypatch.setattr(runner, "load_real_language_model", lambda *_args: language_model)
    provider = object()
    provider_call: dict[str, object] = {}

    def fake_provider(data, model, **kwargs):
        assert data is config.data or data == config.data
        assert model is language_model
        provider_call.update(kwargs)
        return provider

    monkeypatch.setattr(runner, "make_live_activation_provider", fake_provider)
    eval_call: dict[str, object] = {}

    def fake_evaluate(model, actual_provider, actual_config, actual_spec, **kwargs):
        assert model is training_model
        assert actual_provider is provider
        assert actual_config == config
        assert actual_spec == spec
        eval_call.update(kwargs)
        return (
            {
                "run_id": spec.run_id,
                "method": spec.method,
                "method_label": "L1/ReLU SAE",
                "beta_mode": "learned",
                "control_name": spec.control_name,
                "control_value": spec.control_value,
                "seed": spec.seed,
                "rho_model": 0.01,
                "target_name": config.data.target_name,
                "model_id": config.data.model_id,
                "model_revision": config.data.model_revision,
                "layer": config.data.layer,
                "hook_name": config.data.hook_name,
            },
            {"input": np.zeros((1, config.data.input_dim), dtype=np.float32)},
        )

    monkeypatch.setattr(runner, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(
        runner,
        "_evaluate_downstream",
        lambda *_args: {"ce_loss_score": 0.8, "kl_div_score": 0.7},
    )

    runner.evaluate_one(
        run_dir,
        "cpu",
        True,
        decoder_pairwise_block_size=17,
    )

    assert provider_call["start_token"] == config.data.eval_token_offset
    assert provider_call["total_tokens"] == config.data.n_eval_tokens
    assert provider_call["mix_fraction"] == 0.0
    assert eval_call["n_eval_tokens"] == config.data.n_eval_tokens
    assert eval_call["preview_tokens"] == config.training.preview_tokens
    assert eval_call["decoder_pairwise_block_size"] == 17
    metrics = json.loads((run_dir / "eval" / "last" / "metrics.json").read_text())
    assert metrics["checkpoint_training_tokens"] == 4_096
    assert metrics["train_source_fingerprint"] == "train-source"
    assert metrics["eval_pipeline_fingerprint"] == "eval-pipeline"
    assert metrics["ce_loss_score"] == pytest.approx(0.8)
    status = json.loads((run_dir / "eval" / "last" / "status.json").read_text())
    assert status["state"] == "complete"
    assert status["n_evaluation_tokens"] == config.data.n_eval_tokens
    assert (run_dir / "eval" / "last" / "cache.npz").exists()
    assert runner._is_complete(run_dir, "eval-pipeline")


def test_aggregate_writes_seed_means_curves_and_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dirs: list[Path] = []
    for seed, l0 in ((0, 10.0), (1, 14.0)):
        run_dir = tmp_path / "runs" / "l1" / f"seed-{seed}"
        run_dirs.append(run_dir)
        write_json(
            run_dir / "eval" / "last" / "metrics.json",
            {
                "run_id": run_dir.name,
                "target_name": "gemma-2-2b-layer5",
                "model_id": "google/gemma-2-2b",
                "model_revision": "revision",
                "layer": 5,
                "hook_name": "model.layers.5",
                "method": "l1",
                "method_label": "L1/ReLU SAE",
                "beta_mode": "learned",
                "control_name": "l1_coefficient",
                "control_value": 1.25,
                "seed": seed,
                "rho_model": l0 / 32_768,
                "average_l0": l0,
                "explained_variance": 0.5 + seed * 0.1,
                "train_source_fingerprint": "train-source",
                "train_pipeline_fingerprint": "train-pipeline",
                "eval_pipeline_fingerprint": "eval-pipeline",
            },
        )
        write_json(
            run_dir / "eval" / "last" / "status.json",
            {
                "eval_provenance": {"source_fingerprint": "eval-source"},
            },
        )
        write_rows(
            run_dir / "training_history.csv",
            [{"method": "l1", "run_id": run_dir.name, "step": 0, "loss": 1.0}],
        )

    figure = tmp_path / "summary" / "last" / "figures" / "reconstruction.png"
    monkeypatch.setattr(runner, "plot_all", lambda _root: [figure])

    runner._aggregate(tmp_path, run_dirs, run_dirs)

    mean_csv = tmp_path / "summary" / "last" / "final_metrics_seed_mean.csv"
    assert mean_csv.exists()
    text = mean_csv.read_text()
    assert "n_seeds" in text
    assert ",2," in text or text.rstrip().endswith(",2")
    summary = json.loads((tmp_path / "summary" / "last" / "summary.json").read_text())
    assert summary["n_evaluated_runs"] == 2
    assert summary["methods"] == ["l1"]
    assert summary["figures"] == [
        "summary/last/figures/reconstruction.png"
    ]
    assert (tmp_path / "summary" / "training_curves.csv").exists()


def test_completed_run_dirs_merge_prior_incremental_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = tmp_path / "runs" / "batchtopk" / "prior"
    selected = tmp_path / "runs" / "vgsae" / "selected"
    incomplete = tmp_path / "runs" / "l1" / "incomplete"
    monkeypatch.setattr(
        runner,
        "manifest_run_dirs",
        lambda _sweep_dir: [prior, selected, incomplete],
    )
    monkeypatch.setattr(
        runner,
        "_is_complete",
        lambda run_dir, fingerprint: (
            fingerprint == "eval-fingerprint" and run_dir != incomplete
        ),
    )

    assert runner._completed_run_dirs(tmp_path, "eval-fingerprint") == [
        prior,
        selected,
    ]


def test_eval_scheduler_defaults_to_one_worker_per_device() -> None:
    assert runner.DEFAULT_MAX_PER_DEVICE == 1
