"""Closed-form and exhaustive-mask checks for the constructed replication pilot."""

import itertools
import math

import pytest
import torch

from scripts.idea_discovery_replication_pilot import build_replication_case


@pytest.mark.parametrize("width", [4, 16, 64, 256])
@pytest.mark.parametrize("prior_rule", ["fixed_gamma", "fixed_expected_count"])
@pytest.mark.parametrize("beta_mode", ["learned", "profiled"])
def test_actual_free_energy_and_hard_inference_match_replication_closed_form(
    width: int, prior_rule: str, beta_mode: str
) -> None:
    model, x, pi = build_replication_case(width, prior_rule, beta_mode)
    result = model.free_energy(x)
    variance = (1 - pi) / (2 * width * pi)
    expected_loss = (
        2 * variance - 4 * math.log(2 / (2 * math.pi))
        if beta_mode == "learned"
        else 4 * math.log(2 * variance / 8)
    )
    assert x.dtype == torch.float64
    assert model.decoder.weight.dtype == torch.float64
    assert torch.equal(model.decoder_column_sqnorms(), torch.ones(width, dtype=torch.float64))
    assert result["x_hat"].detach().numpy() == pytest.approx(x.numpy(), abs=1e-10, rel=1e-10)
    assert float(result["recon"].detach()) == pytest.approx(0, abs=1e-20)
    assert float(result["variance"].detach()) == pytest.approx(variance, abs=1e-10, rel=1e-10)
    assert float((result["prior"] - result["entropy"]).detach()) == pytest.approx(0, abs=1e-10)
    assert float(result["loss"].detach()) == pytest.approx(expected_loss, abs=1e-10, rel=1e-10)
    assert float(result["beta_eff"].detach()) == pytest.approx(
        2 if beta_mode == "learned" else 8 / (2 * variance), abs=1e-10, rel=1e-10
    )
    mask, _, hard_code = model.encode_inference(x)
    hard_output = model.decode(hard_code)
    assert torch.count_nonzero(mask) == 0
    assert torch.count_nonzero(hard_output) == 0
    assert float((hard_output - x).square().sum().detach()) == 1


@pytest.mark.parametrize("prior_rule", ["fixed_gamma", "fixed_expected_count"])
def test_energy_matches_exhaustive_bernoulli_mask_expectation(prior_rule: str) -> None:
    model, x, pi = build_replication_case(4, prior_rule, "learned")
    _, amplitude, _ = model.encode(x)
    masks = torch.tensor(list(itertools.product([0.0, 1.0], repeat=4)), dtype=torch.float64)
    probabilities = (pi ** masks * (1 - pi) ** (1 - masks)).prod(dim=1)
    decoded = model.decode(masks * amplitude)
    direct_energy = (probabilities * 0.5 * (decoded - x).square().sum(dim=1)).sum()
    direct_mean = (probabilities[:, None] * decoded).sum(dim=0, keepdim=True)
    result = model.free_energy(x)
    assert float(probabilities.sum()) == pytest.approx(1, abs=1e-14)
    assert torch.allclose(result["energy"], direct_energy, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result["x_hat"], direct_mean, atol=1e-12, rtol=1e-12)


def test_fixed_gamma_profiled_loss_decreases_with_replication() -> None:
    losses, variances = [], []
    for width in (4, 16, 64, 256):
        model, x, _ = build_replication_case(width, "fixed_gamma", "profiled")
        result = model.free_energy(x)
        losses.append(float(result["loss"].detach()))
        variances.append(float(result["variance"].detach()))
    for first, second in zip(losses, losses[1:]):
        assert second - first == pytest.approx(-4 * math.log(4), abs=1e-10, rel=1e-10)
    for first, second in zip(variances, variances[1:]):
        assert second / first == pytest.approx(0.25, abs=1e-12, rel=1e-12)


def test_matched_prior_count_removes_this_inverse_width_energy_decrease() -> None:
    previous_variance = 0.0
    for width in (4, 16, 64, 256):
        model, x, _ = build_replication_case(width, "fixed_expected_count", "profiled")
        result = model.free_energy(x)
        variance = float(result["variance"].detach())
        assert float(result["sparsity"].detach()) == pytest.approx(1, abs=1e-12)
        assert variance == pytest.approx(0.5 * (1 - 1 / width), abs=1e-12)
        assert previous_variance < variance < 0.5
        previous_variance = variance
