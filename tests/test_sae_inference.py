"""Independent small-system checks for conditional support inference."""
import itertools
import math

import pytest
import torch

from src.sae_inference import (
    best_found_mean_field,
    enumerate_support_posterior,
    factorized_free_energy,
    mean_field_fixed_point,
)


DTYPE = torch.float64


def _tiny_case():
    x = torch.tensor([[0.3, -0.4], [1.1, 0.2]], dtype=DTYPE)
    d = torch.tensor([[1.0, 0.6, -0.3], [0.0, 0.8, 0.5]], dtype=DTYPE)
    a = torch.tensor([[0.7, 1.2, -0.8], [1.5, 0.0, 0.4]], dtype=DTYPE)
    return x, d, a


def _direct_distribution(x, d, a, beta, gamma):
    # A separate probability-distribution construction, including both
    # normalizers, avoids repeating the production energy implementation.
    states = torch.tensor(list(itertools.product([0.0, 1.0], repeat=d.shape[1])), dtype=x.dtype)
    signal = (a[:, None, :] * states) @ d.T
    prior = torch.distributions.Bernoulli(logits=torch.full_like(states, -gamma))
    likelihood = torch.distributions.Normal(signal, 1 / math.sqrt(beta))
    joint = likelihood.log_prob(x[:, None, :]).sum(-1) + prior.log_prob(states).sum(-1)
    return states, joint


def test_orthogonal_exact_factorization_with_signed_samplewise_amplitudes():
    x = torch.tensor([[0.2, -1.4, 0.7], [1.3, 0.1, -0.9]], dtype=DTYPE)
    d = torch.diag(torch.tensor([1.0, 0.6, -1.3], dtype=DTYPE))
    a = torch.tensor([[1.2, -0.7, 0.0], [0.5, 1.4, -0.9]], dtype=DTYPE)
    beta, gamma = 4.3, 0.8
    expected = torch.sigmoid(beta * a * (x @ d) - 0.5 * beta * a.square() * d.square().sum(0) - gamma)
    exact = enumerate_support_posterior(x, d, a, beta=beta, gamma=gamma)
    torch.testing.assert_close(exact["m"], expected, atol=1e-14, rtol=1e-13)
    off_diagonal = exact["covariance"] - torch.diag_embed(exact["covariance"].diagonal(dim1=-2, dim2=-1))
    assert off_diagonal.abs().max() < 1e-14
    free_energy = factorized_free_energy(x, d, a, expected, beta=beta, gamma=gamma)
    torch.testing.assert_close(free_energy, -exact["log_evidence"], atol=1e-13, rtol=1e-13)
    mf = best_found_mean_field(x, d, a, beta=beta, gamma=gamma, starts=[torch.full_like(a, 0.2)])
    torch.testing.assert_close(mf["m"], expected, atol=1e-13, rtol=1e-13)
    assert (mf["sweeps"] == 1).all()
    assert mf["converged"].all()


def test_free_energy_and_gradient_equal_direct_enumerated_expectation():
    x, d, a = _tiny_case()
    beta, gamma = 3.7, -0.4
    m = torch.tensor([[0.2, 0.6, 0.4], [0.7, 0.3, 0.8]], dtype=DTYPE, requires_grad=True)
    states, log_joint = _direct_distribution(x, d, a, beta, gamma)
    log_q = torch.distributions.Bernoulli(probs=m[:, None, :]).log_prob(states).sum(-1)
    q = log_q.exp()
    direct_f = (q * (log_q - log_joint)).sum(-1)
    analytic_f = factorized_free_energy(x, d, a, m, beta=beta, gamma=gamma)
    torch.testing.assert_close(analytic_f, direct_f, atol=2e-14, rtol=1e-13)
    direct_gradient = torch.autograd.grad(direct_f.sum(), m, retain_graph=True)[0]
    analytic_gradient = torch.autograd.grad(analytic_f.sum(), m)[0]
    torch.testing.assert_close(analytic_gradient, direct_gradient, atol=2e-14, rtol=1e-13)
    exact = enumerate_support_posterior(x, d, a, beta=beta, gamma=gamma)
    torch.testing.assert_close(exact["log_evidence"], torch.logsumexp(log_joint, -1))
    direct_kl = (q * (log_q - log_joint + exact["log_evidence"][:, None])).sum(-1)
    torch.testing.assert_close(analytic_f + exact["log_evidence"], direct_kl)
    assert (direct_kl >= 0).all()


def test_field_response_is_negative_support_count_variance():
    x, d, a = _tiny_case()
    beta, gamma, eps = 5.0, 0.7, 1e-5
    exact = enumerate_support_posterior(x, d, a, beta=beta, gamma=gamma)
    count = exact["states"].sum(-1)
    mean_count = (exact["p"] * count).sum(-1)
    variance = (exact["p"] * (count - mean_count[:, None]).square()).sum(-1)
    plus = enumerate_support_posterior(x, d, a, beta=beta, gamma=gamma + eps)["m"].sum(-1)
    minus = enumerate_support_posterior(x, d, a, beta=beta, gamma=gamma - eps)["m"].sum(-1)
    torch.testing.assert_close((plus - minus) / (2 * eps), -variance, atol=2e-10, rtol=1e-8)


def test_coherent_atoms_have_negative_conditional_support_covariance():
    d = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=DTYPE)
    x = torch.tensor([[1.0, 0.0]], dtype=DTYPE)
    a = torch.ones(2, dtype=DTYPE)
    exact = enumerate_support_posterior(x, d, a, beta=40.0, gamma=0.0)
    torch.testing.assert_close(exact["m"], torch.full((1, 2), 0.5, dtype=DTYPE))
    assert exact["covariance"][0, 0, 1] < -0.249
    factorized_variance = (exact["m"] * (1 - exact["m"]) * a.square() * d.square().sum(0)).sum(-1)
    assert exact["signal_variance"].item() < 1e-7
    assert factorized_variance.item() > 0.49


@pytest.mark.parametrize("gamma", [-1000.0, 1000.0])
def test_endpoint_probabilities_and_extreme_finite_prior_costs_are_stable(gamma):
    x, d, a = _tiny_case()
    endpoint_m = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=DTYPE)
    exact = enumerate_support_posterior(x, d, a, beta=2.0, gamma=gamma)
    energy = factorized_free_energy(x, d, a, endpoint_m, beta=2.0, gamma=gamma)
    assert torch.isfinite(energy).all()
    for name in ("log_joint", "log_p", "p", "m", "signal_variance", "covariance"):
        assert torch.isfinite(exact[name]).all()
    torch.testing.assert_close(exact["p"].sum(-1), torch.ones(len(x), dtype=DTYPE))


def test_broadcast_amplitudes_agree_with_repeated_samplewise_input():
    x, d, _ = _tiny_case()
    amplitude = torch.tensor([0.6, -1.0, 1.8], dtype=DTYPE)
    broad = enumerate_support_posterior(x, d, amplitude, beta=2.0, gamma=0.4)
    rows = enumerate_support_posterior(x, d, amplitude.repeat(len(x), 1), beta=2.0, gamma=0.4)
    for key in broad:
        torch.testing.assert_close(broad[key], rows[key])


def test_coordinate_descent_and_samplewise_best_start_preserve_inputs():
    x, d, a = _tiny_case()
    starts = [torch.full_like(a, p) for p in (0.02, 0.5, 0.98)]
    originals = [value.clone() for value in (x, d, a, *starts)]
    settings = dict(beta=5.0, gamma=0.8, starts=starts, tolerance=1e-11)
    previous = best_found_mean_field(x, d, a, max_sweeps=0, **settings)
    for sweeps in (1, 2, 5, 100):
        current = best_found_mean_field(x, d, a, max_sweeps=sweeps, **settings)
        assert (current["all_start_free_energy"] <= previous["all_start_free_energy"] + 2e-14).all()
        previous = current
    result = current
    assert result["converged"].all()
    assert result["residual"].max() <= settings["tolerance"]
    torch.testing.assert_close(result["free_energy"], result["all_start_free_energy"].min(-1).values)
    assert (result["free_energy"] <= result["initial_start_free_energy"].min(-1).values + 2e-14).all()
    fixed = mean_field_fixed_point(x, d, a, result["m"], beta=5.0, gamma=0.8)
    torch.testing.assert_close(result["residual"], (result["m"] - fixed).abs().amax(-1))
    for actual, original in zip((x, d, a, *starts), originals, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


def test_invalid_probabilities_and_precision_fail_explicitly():
    x, d, a = _tiny_case()
    with pytest.raises(ValueError, match="beta"):
        enumerate_support_posterior(x, d, a, beta=0.0, gamma=0.0)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        factorized_free_energy(x, d, a, torch.full_like(a, 1.1), beta=1.0, gamma=0.0)
    with pytest.raises(ValueError, match="at least one"):
        best_found_mean_field(x, d, a, beta=1.0, gamma=0.0, starts=[])
