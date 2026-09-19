"""Independent deterministic witnesses for the known-model posterior pilot."""
import math

import torch

from scripts.idea_discovery_posterior_pilot import (
    BETA, GAMMA, PI, coordinate_meanfield, exact_posterior,
    make_oracle_model, meanfield_energy,
)


def test_orthogonal_enumeration_matches_analytic_posterior():
    x = torch.tensor([[.1, .6, -.3], [.9, .2, 1.2]], dtype=torch.float64)
    d = torch.eye(3, dtype=torch.float64)
    exact = exact_posterior(x, d)
    expected = torch.sigmoid(BETA * x - .5 * BETA - GAMMA)
    torch.testing.assert_close(exact["m"], expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(exact["p"].sum(-1), torch.ones(2, dtype=x.dtype))


def test_vg_energy_matches_independent_support_enumeration():
    d = torch.tensor([[1., .95], [0., math.sqrt(1 - .95**2)]], dtype=torch.float64)
    x = torch.tensor([[.2, .5], [.8, -.1]], dtype=torch.float64)
    model = make_oracle_model(d)
    logits = torch.logit(torch.tensor([.23, .67], dtype=x.dtype))
    with torch.no_grad():
        model.gate_encoder.weight.zero_()
        model.gate_encoder.bias.copy_(logits)
    terms = model.free_energy(x)
    m = terms["m"]
    exact = exact_posterior(x, d)
    states = exact["states"]
    log_q = (states[None] * m[:, None].log() + (1 - states[None]) * torch.log1p(-m[:, None])).sum(-1)
    enumerated_f = (log_q.exp() * (log_q - exact["log_joint"])).sum(-1)
    torch.testing.assert_close(terms["loss"], enumerated_f.mean(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(meanfield_energy(x, d, m), enumerated_f, atol=1e-12, rtol=1e-12)


def test_coordinate_updates_recover_orthogonal_exact_posterior():
    x = torch.tensor([[.4, .7], [1.1, -.2]], dtype=torch.float64)
    d = torch.eye(2, dtype=x.dtype)
    refined, residual = coordinate_meanfield(x, d, torch.full_like(x, PI), sweeps=1)
    exact = exact_posterior(x, d)
    torch.testing.assert_close(refined, exact["m"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(meanfield_energy(x, d, refined) + exact["log_evidence"], torch.zeros(2, dtype=x.dtype), atol=1e-12, rtol=0)
    assert residual.max() < 1e-12


def test_coherent_coordinate_update_does_not_increase_energy():
    d = torch.tensor([[1., .95], [0., math.sqrt(1 - .95**2)]], dtype=torch.float64)
    x = torch.tensor([[.5, .2], [1.2, -.1]], dtype=torch.float64)
    initial = torch.tensor([[.2, .4], [.8, .1]], dtype=torch.float64)
    refined, _ = coordinate_meanfield(x, d, initial, sweeps=1)
    assert torch.all(meanfield_energy(x, d, refined) <= meanfield_energy(x, d, initial) + 1e-12)
