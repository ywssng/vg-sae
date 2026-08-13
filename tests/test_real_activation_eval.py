from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sae_lens.saes.standard_sae import (
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)

from src.real_activation_eval import evaluate_model
from src.saelens_vg import VGTrainingSAE, VGTrainingSAEConfig


class _RecordingProvider:
    def __init__(self, batches: list[torch.Tensor]) -> None:
        self.batches = batches
        self.calls = 0

    def __iter__(self) -> _RecordingProvider:
        return self

    def __next__(self) -> torch.Tensor:
        if self.calls == len(self.batches):
            raise StopIteration
        batch = self.batches[self.calls]
        self.calls += 1
        return batch


def _identity_config(*, n_eval: int, preview_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(n_eval=n_eval),
        training=SimpleNamespace(preview_tokens=preview_tokens),
    )


def test_streaming_real_eval_truncates_exactly_and_matches_hand_metrics() -> None:
    model = StandardTrainingSAE(
        StandardTrainingSAEConfig(
            d_in=2,
            d_sae=2,
            l1_coefficient=0.0,
            decoder_init_norm=None,
        )
    )
    with torch.no_grad():
        model.W_enc.copy_(torch.eye(2))
        # Identical unit decoder rows make the pairwise absolute cosine exactly 1.
        model.W_dec.copy_(torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        model.b_enc.zero_()
        model.b_dec.zero_()

    provider = _RecordingProvider(
        [
            torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
            torch.tensor([[1.0, 1.0], [3.0, 4.0], [100.0, 100.0]]),
            torch.full((2, 2), 1_000.0),
        ]
    )
    config = _identity_config(n_eval=4, preview_tokens=2)
    spec = SimpleNamespace(
        run_id="gemma-layer-12-l1",
        method="l1",
        method_label="L1/ReLU SAE",
        model_id="google/gemma-2-2b",
        layer=12,
        hook_name="blocks.12.hook_resid_post",
        control_name="l1_coefficient",
        control_value=1.0e-3,
        seed=3,
        init_seed=4,
        eval_stream_seed=5,
    )

    row, cache = evaluate_model(model, provider, config, spec)

    # Only the first two rows of the oversized final batch count; the provider's
    # third batch is never requested.
    assert provider.calls == 2
    assert row["n_evaluation_tokens"] == 4
    assert row["n_evaluation_samples"] == 4
    assert row["reconstruction_mse"] == pytest.approx(42.0 / 8.0)
    assert row["reconstruction_error"] == pytest.approx(math.sqrt(42.0 / 32.0))
    assert row["explained_variance"] == pytest.approx(
        1.0 - (42.0 / 4.0) / (8.0 - 4.625)
    )
    assert row["reconstruction_cosine"] == pytest.approx(
        (1.0 + 0.0 + 1.0 / math.sqrt(2.0) + 3.0 / 5.0) / 4.0
    )
    assert row["average_l0"] == pytest.approx(1.5)
    assert row["l1"] == pytest.approx(3.0)
    assert row["rho_model"] == pytest.approx(0.75)
    assert row["dead_latents"] == 0
    assert row["dead_fraction"] == pytest.approx(0.0)
    assert row["input_l2_mean"] == pytest.approx(
        (1.0 + 2.0 + math.sqrt(2.0) + 5.0) / 4.0
    )
    assert row["output_l2_mean"] == pytest.approx(3.0)
    assert row["l2_ratio"] == pytest.approx(
        (1.0 + 1.0 + math.sqrt(2.0) + 7.0 / 5.0) / 4.0
    )
    assert row["relative_reconstruction_bias"] == pytest.approx(58.0 / 24.0)
    assert row["decoder_pairwise_cosine_similarity"] == pytest.approx(1.0)

    assert row["run_id"] == "gemma-layer-12-l1"
    assert row["method"] == "l1"
    assert row["model_id"] == "google/gemma-2-2b"
    assert row["layer"] == 12
    assert row["hook_name"] == "blocks.12.hook_resid_post"
    assert row["control_name"] == "l1_coefficient"
    assert row["control_value"] == pytest.approx(1.0e-3)
    assert row["seed"] == 3

    assert row["ground_truth_available"] is False
    for metric in (
        "support_f1",
        "decoder_recovery_cosine",
        "mcc",
        "uniqueness",
        "classification_accuracy",
        "generalization_error",
    ):
        assert row[metric] is None
    assert row["vg_expected_l0"] is None
    assert row["expected_l0"] == pytest.approx(row["average_l0"])
    # This rejects NaN/Infinity and verifies the metric row is JSON serializable.
    json.dumps(row, allow_nan=False)

    assert np.asarray(cache["preview_token_count"]).item() == 2
    assert cache["input"].shape == (2, 2)
    assert cache["h"].shape == (2, 2)
    assert cache["mask"].dtype == np.bool_
    np.testing.assert_allclose(cache["input"], [[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(cache["reconstruction"], [[1.0, 0.0], [2.0, 0.0]])


def test_vg_real_eval_reports_hard_expected_and_posterior_metrics() -> None:
    model = VGTrainingSAE(
        VGTrainingSAEConfig(
            d_in=2,
            d_sae=2,
            beta_mode="learned",
            decoder_bias=False,
        )
    )
    with torch.no_grad():
        model.core.gate_encoder.weight.zero_()
        model.core.gate_encoder.bias.copy_(
            torch.logit(torch.tensor([0.25, 0.75]))
        )
        model.core.amplitude_encoder.weight.zero_()
        model.core.amplitude_encoder.bias.copy_(
            torch.log(torch.expm1(torch.tensor([2.0, 4.0])))
        )
        model.core.decoder.weight.copy_(torch.eye(2))

    provider = _RecordingProvider(
        [torch.tensor([[0.0, 4.0]]), torch.tensor([[0.0, 2.0], [99.0, 99.0]])]
    )
    row, cache = evaluate_model(
        model,
        provider,
        n_eval_tokens=2,
        preview_tokens=2,
        identity={
            "run_id": "llama-layer-8-vg",
            "method": "vgsae",
            "model": "meta-llama/Llama-3.2-1B",
            "layer": 8,
            "hook_name": "blocks.8.hook_resid_post",
            "control_name": "lambda_sparsity",
            "control_value": 0.5,
            "seed": 11,
            "beta_mode": "learned",
        },
    )

    assert provider.calls == 2
    assert row["method"] == "vgsae"
    assert row["model_id"] == "meta-llama/Llama-3.2-1B"
    assert row["beta_mode"] == "learned"
    assert row["average_l0"] == pytest.approx(1.0)
    assert row["l1"] == pytest.approx(4.0)
    assert row["rho_model"] == pytest.approx(0.5)
    assert row["dead_latents"] == 1
    assert row["reconstruction_mse"] == pytest.approx(1.0)
    assert row["reconstruction_error"] == pytest.approx(math.sqrt(4.0 / 20.0))
    assert row["explained_variance"] == pytest.approx(-1.0)
    assert row["reconstruction_cosine"] == pytest.approx(1.0)
    assert row["relative_reconstruction_bias"] == pytest.approx(4.0 / 3.0)
    assert row["decoder_pairwise_cosine_similarity"] == pytest.approx(0.0)

    assert row["vg_expected_reconstruction_mse"] == pytest.approx(0.625)
    assert row["vg_expected_relative_error"] == pytest.approx(
        math.sqrt(2.5 / 20.0)
    )
    assert row["vg_expected_explained_variance"] == pytest.approx(-0.25)
    assert row["vg_expected_hard_ev_gap"] == pytest.approx(0.75)
    assert row["vg_expected_l0"] == pytest.approx(1.0)
    assert row["expected_l0"] == pytest.approx(1.0)
    assert row["vg_expected_density"] == pytest.approx(0.5)
    assert row["vg_posterior_rho"] == pytest.approx(0.5)
    assert row["vg_posterior_variance"] == pytest.approx(0.1875)
    assert row["vg_expected_to_hard_l0_ratio"] == pytest.approx(1.0)
    assert row["vg_posterior_probability_q10"] == pytest.approx(0.25)
    assert row["vg_posterior_probability_q25"] == pytest.approx(0.25)
    assert row["vg_posterior_probability_q50"] == pytest.approx(0.5)
    assert row["vg_posterior_probability_q75"] == pytest.approx(0.75)
    assert row["vg_posterior_probability_q90"] == pytest.approx(0.75)
    assert row["vg_posterior_probability_quantile_sample_count"] == 4

    np.testing.assert_allclose(cache["h"], [[0.0, 4.0], [0.0, 4.0]])
    np.testing.assert_allclose(
        cache["reconstruction"], [[0.0, 4.0], [0.0, 4.0]]
    )
    np.testing.assert_allclose(
        cache["posterior_probability"], [[0.25, 0.75], [0.25, 0.75]]
    )
    np.testing.assert_allclose(cache["expected_h"], [[0.5, 3.0], [0.5, 3.0]])
    np.testing.assert_allclose(
        cache["expected_reconstruction"], [[0.5, 3.0], [0.5, 3.0]]
    )
    json.dumps(row, allow_nan=False)


def test_real_eval_rejects_provider_shorter_than_exact_budget() -> None:
    model = StandardTrainingSAE(
        StandardTrainingSAEConfig(d_in=2, d_sae=2, l1_coefficient=0.0)
    )
    provider = _RecordingProvider([torch.ones(2, 2)])

    with pytest.raises(ValueError, match="exhausted after 2 tokens"):
        evaluate_model(
            model,
            provider,
            n_eval_tokens=3,
            preview_tokens=1,
        )
