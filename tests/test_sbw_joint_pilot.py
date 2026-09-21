"""Deterministic checks for conditional support inference and P3 isolation."""
import math

import torch

from scripts.run_sbw_joint_pilot import (
    code_metrics, coordinate_logits, exact_posterior, make_model, mf_statistics,
    product_free_energy, product_risk, solve_mean_field, state_risks, states_for,
    training_loss,
)
from scripts.sbw_pilot_utils import geometry_metrics, orthogonal_dictionary, sample_toy


def tiny_case():
    generator = torch.Generator().manual_seed(71)
    y = torch.randn(4, 4, generator=generator, dtype=torch.float64)*.3
    decoder = torch.randn(4, 3, generator=generator, dtype=torch.float64)*.3
    a = .5 + torch.rand(4, 3, generator=generator, dtype=torch.float64)*.3
    return y, a, decoder


def test_product_variance_equals_enumerated_risk_and_complete_free_energy():
    y, a, decoder = tiny_case()
    states = states_for(3, a)
    m = torch.tensor([[.2, .5, .8]], dtype=torch.float64).expand_as(a)
    q = (m[:, None]**states * (1-m[:, None])**(1-states)).prod(-1)
    risks = state_risks(y, a, decoder, states)
    direct = sum(q[:, i]*(y-(a*state)@decoder.T).square().sum(-1) for i, state in enumerate(states))
    torch.testing.assert_close((q*risks).sum(-1), direct, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(product_risk(y, a, decoder, m), direct, atol=1e-12, rtol=1e-12)
    beta, gamma = 2., 1.3
    full = (q*(.5*beta*risks + gamma*states.sum(-1) + q.log())).sum(-1)
    full += 3*torch.nn.functional.softplus(a.new_tensor(-gamma)) - 2*math.log(beta/(2*math.pi))
    torch.testing.assert_close(product_free_energy(y, a, decoder, m, beta, gamma), full, atol=1e-12, rtol=1e-12)


def test_all_32_support_states_have_correct_risks_and_map_tie_order():
    generator = torch.Generator().manual_seed(81)
    y = torch.randn(3, 20, generator=generator, dtype=torch.float64)
    a = torch.rand(3, 5, generator=generator, dtype=torch.float64)
    decoder = torch.randn(20, 5, generator=generator, dtype=torch.float64)
    out = exact_posterior(y, a, decoder, 10., 4.)
    assert out["q"].shape == (3, 32)
    direct = torch.zeros(3, dtype=torch.float64)
    for index in range(32):
        mask = torch.tensor([(index >> j) & 1 for j in range(5)], dtype=torch.float64)
        residual = y-(mask*a)@decoder.T
        direct += out["q"][:, index]*residual.square().sum(1)
    torch.testing.assert_close(out["risk"], direct, atol=1e-12, rtol=1e-12)
    tied = exact_posterior(torch.zeros_like(y), torch.zeros_like(a), decoder, 10., 0.)
    assert tied["map"].count_nonzero() == 0


def test_orthogonal_exact_posterior_factorizes():
    y, a, _ = tiny_case()
    decoder = torch.eye(4, dtype=torch.float64)[:, :3]
    exact = exact_posterior(y, a, decoder, 10., 4.)
    logits = 10*a*y[:, :3] - 5*a.square()-4
    m = logits.sigmoid()
    q_product = (m[:, None]**exact["states"] * (1-m[:, None])**(1-exact["states"])).prod(-1)
    torch.testing.assert_close(exact["m"], m, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(exact["q"], q_product, atol=1e-12, rtol=1e-12)
    solved = solve_mean_field(y, a, decoder, 10., 4.)
    assert solved["converged"].all()
    torch.testing.assert_close(solved["m"], exact["m"], atol=1e-12, rtol=1e-12)


def test_exact_log_partition_gradient_equals_envelope():
    tensors = tuple(t.requires_grad_() for t in tiny_case())
    exact = exact_posterior(*tensors, 2., .4)
    logz_grad = torch.autograd.grad(exact["free_energy"].sum(), tensors, retain_graph=True)
    q = exact["q"].detach()
    envelope = (q*exact["energy"]).sum() + torch.xlogy(q, q).sum()
    envelope_grad = torch.autograd.grad(envelope, tensors)
    for actual, expected in zip(logz_grad, envelope_grad):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_coordinate_descent_monotonic_stationary_and_gradient_finite_difference():
    y, a, decoder = tiny_case()
    solved = solve_mean_field(y, a, decoder, 2., .4, keep_history=True,
                             iterate_tol=1e-12, stationarity_tol=1e-12)
    assert solved["converged"].all()
    for before, after in zip(solved["history"], solved["history"][1:]):
        assert bool((after-before <= 1e-12).all())
    m = solved["m"].detach().requires_grad_()
    value = product_free_energy(y, a, decoder, m, 2., .4).sum()
    grad, = torch.autograd.grad(value, m)
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=2e-10, rtol=0)
    linear, off = mf_statistics(y, a, decoder, 2., .4)
    torch.testing.assert_close(m.logit(), coordinate_logits(m, linear, off), atol=2e-10, rtol=0)
    probe = torch.full_like(m, .3, requires_grad=True)
    analytic, = torch.autograd.grad(product_free_energy(y, a, decoder, probe, 2., .4).sum(), probe)
    delta = torch.zeros_like(probe)
    delta[1, 2] = 1e-6
    finite = (product_free_energy(y, a, decoder, probe.detach()+delta, 2., .4).sum()
              -product_free_energy(y, a, decoder, probe.detach()-delta, 2., .4).sum())/2e-6
    torch.testing.assert_close(analytic[1, 2], finite, atol=1e-8, rtol=1e-8)


def test_converged_mean_field_envelope_matches_resolved_finite_difference():
    y, a, decoder = tiny_case()
    decoder.requires_grad_()
    solved = solve_mean_field(y, a, decoder, 2., .4, iterate_tol=1e-12, stationarity_tol=1e-12)
    assert solved["converged"].all()
    grad, = torch.autograd.grad(product_free_energy(y, a, decoder, solved["m"], 2., .4).sum(), decoder)
    values = []
    for sign in (-1, 1):
        changed = decoder.detach().clone()
        changed[1, 2] += sign*1e-6
        solution = solve_mean_field(y, a, changed, 2., .4, iterate_tol=1e-12, stationarity_tol=1e-12)
        assert solution["converged"].all()
        values.append(solution["objective"].sum())
    torch.testing.assert_close(grad[1, 2], (values[1]-values[0])/2e-6, atol=1e-8, rtol=1e-8)


def test_known_dictionary_and_code_identity_metrics():
    dictionary = orthogonal_dictionary(210)
    data = sample_toy(dictionary, 512, 10210, correlation=-.4)
    geom, (li, ti, _) = geometry_metrics(dictionary, dictionary)
    metrics = code_metrics(data.z, data.x, data, li, ti)
    assert abs(geom["matched_positive_cosine"]-1) < 1e-12
    assert geom["mixing_energy"] < 1e-12
    assert metrics["support_f1"] == metrics["support_macro_f1"] == 1
    assert metrics["hard_latent_relative_error"] == 0


def test_fixed_beta_shared_initialization_and_normalized_product_core_equality():
    models = [make_model(arm, 4., 210) for arm in ("amortized", "optimized_mf", "exact32")]
    for model in models:
        assert not model.log_beta.requires_grad
        torch.testing.assert_close(model.decoder.weight, models[0].decoder.weight)
        torch.testing.assert_close(model.amplitude_encoder.weight, models[0].amplitude_encoder.weight)
        assert model.pre_bias.count_nonzero() == 0
    x = torch.randn(8, 20, generator=torch.Generator().manual_seed(6))
    for arm, model in zip(("amortized", "optimized_mf", "exact32"), models):
        loss, _ = training_loss(model, arm, x)
        loss.backward()
        assert model.log_beta.grad is None
    model = models[0].double()
    x = x.double()
    m, a, _ = model.encode(x)
    ours = product_free_energy(x-model.pre_bias, a, model.decoder.weight, m, float(model.log_beta.exp()), 4.)
    torch.testing.assert_close(ours.mean(), model.free_energy(x)["loss"], atol=1e-10, rtol=1e-10)
