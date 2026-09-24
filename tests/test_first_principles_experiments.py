"""Regression checks for campaign evaluation, independent of training outcomes."""
import math

import torch

from scripts.run_first_principles import (
    joint_metrics,
    known_dictionary,
    profile_batch_diagnostics,
    sample_data,
)
from src.sae_model import VGSAEConfig, VariationalGarroteSAE


def test_variance_ablation_is_evaluated_with_full_stochastic_risk() -> None:
    model = VariationalGarroteSAE(VGSAEConfig(
        input_dim=2, n_latents=2, use_variance_term=False,
        decoder_bias=False, beta_mode="learned",
    ))
    with torch.no_grad():
        model.decoder.weight.copy_(torch.eye(2))
        model.gate_encoder.weight.zero_()
        model.gate_encoder.bias.zero_()
        model.amplitude_encoder.weight.zero_()
        model.amplitude_encoder.bias.fill_(math.log(math.expm1(1)))
    data = sample_data(torch.eye(2), 8, 92, .5, 0.)
    metrics = joint_metrics(model, data, "cpu")
    # Two independent fair unit-amplitude selectors: total variance=.5,
    # so per-coordinate stochastic risk exceeds mean risk by .25.
    assert abs(metrics["full_stochastic_mse"] - metrics["mean_mse"] - .25) < 1e-6
    assert metrics["optimized_variance_energy"] == 0
    assert abs(metrics["full_variance_energy"] - .25) < 1e-6
    assert metrics["expected_mask_count"] == 1
    assert metrics["hard_l0"] == 0  # Strict m>.5, never count soft means as hard.


def test_joint_matching_keeps_nonnegative_feature_sign_and_all_atoms() -> None:
    model = VariationalGarroteSAE(VGSAEConfig(input_dim=2, n_latents=2))
    with torch.no_grad():
        model.decoder.weight.copy_(-torch.eye(2))
    data = sample_data(torch.eye(2), 8, 12, .5, .1)
    metrics = joint_metrics(model, data, "cpu")
    assert metrics["signed_dictionary_cosine"] <= 0
    assert metrics["recovered_fraction_095"] == 0


def test_independent_split_streams_keep_dictionary_and_reproduce() -> None:
    d = known_dictionary(6, .7)
    train = sample_data(d, 32, 240111, .25, .1)
    repeat = sample_data(d, 32, 240111, .25, .1)
    test = sample_data(d, 32, 240113, .25, .1)
    assert torch.equal(train["x"], repeat["x"])
    assert not torch.equal(train["x"], test["x"])
    assert torch.equal(train["dictionary"], test["dictionary"])
    assert torch.allclose(train["clean"], train["z"] @ d.T)


def test_profile_jensen_gap_uses_same_checkpoint_and_leaves_it_unchanged() -> None:
    model = VariationalGarroteSAE(VGSAEConfig(input_dim=2, n_latents=2)).double()
    x = torch.tensor([[0.,0.],[1.,1.],[3.,3.],[9.,9.]],dtype=torch.float64)
    before = {name: tensor.clone() for name,tensor in model.state_dict().items()}
    rows = profile_batch_diagnostics(model, x, [1,2,4])
    assert all(row["jensen_gap"] >= -1e-10 for row in rows)
    assert abs(rows[-1]["jensen_gap"]) < 1e-10
    assert rows[0]["beta_std"] > 0
    for name,tensor in model.state_dict().items():
        assert torch.equal(before[name], tensor)
