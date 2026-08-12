import math

import pytest
import torch
import torch.nn.functional as F

from src.sae_model import VGSAEConfig, VariationalGarroteSAE


def _constant_energy_model(**kwargs: object) -> VariationalGarroteSAE:
    model = VariationalGarroteSAE(
        VGSAEConfig(
            input_dim=2,
            n_latents=1,
            use_variance_term=False,
            use_entropy_term=False,
            **kwargs,
        )
    )
    with torch.no_grad():
        model.decoder.weight.zero_()
        model.pre_bias.zero_()
        model.gate_encoder.weight.zero_()
        model.gate_encoder.bias.zero_()
    return model


def test_learned_beta_uses_full_gaussian_and_normalized_signed_prior() -> None:
    gamma, beta = -0.7, 2.0
    model = _constant_energy_model(beta=beta, beta_mode="learned", lambda_sparsity=gamma)
    x = torch.tensor([[1.0, 2.0], [2.0, 0.0]])
    energy = 0.5 * x.pow(2).sum(dim=1)
    prior = gamma * 0.5 + F.softplus(torch.tensor(-gamma))
    expected = (beta * energy - torch.log(torch.tensor(beta / (2 * math.pi))) + prior).mean()
    assert torch.allclose(model.free_energy(x)["loss"], expected)


def test_learned_beta_gradient_and_stationary_point() -> None:
    x = torch.tensor([[1.0, 2.0], [2.0, 0.0]])
    energy_mean = 0.5 * x.pow(2).sum(dim=1).mean()
    model = _constant_energy_model(beta=1.3, beta_mode="learned", lambda_sparsity=0.0)
    loss = model.free_energy(x)["loss"]
    (gradient,) = torch.autograd.grad(loss, model.log_beta)
    assert torch.allclose(gradient, 1.3 * energy_mean - 1.0)

    stationary = _constant_energy_model(
        beta=float(1.0 / energy_mean), beta_mode="learned", lambda_sparsity=0.0
    )
    stationary.free_energy(x)["loss"].backward()
    assert stationary.log_beta.grad == pytest.approx(0.0, abs=1e-6)


def test_profiled_gradient_matches_learned_at_profiled_beta() -> None:
    torch.manual_seed(4)
    x = torch.randn(5, 3)
    profiled = VariationalGarroteSAE(
        VGSAEConfig(3, 4, beta_mode="profiled", lambda_sparsity=-0.2)
    )
    profiled_output = profiled.free_energy(x)
    beta_star = float(profiled_output["beta_eff"])
    grad_profiled = torch.autograd.grad(
        profiled_output["loss"], profiled.amplitude_encoder.weight
    )[0]

    learned = VariationalGarroteSAE(
        VGSAEConfig(3, 4, beta=beta_star, beta_mode="learned", lambda_sparsity=-0.2)
    )
    learned.load_state_dict(profiled.state_dict(), strict=False)
    grad_learned = torch.autograd.grad(
        learned.free_energy(x)["loss"], learned.amplitude_encoder.weight
    )[0]
    assert torch.allclose(grad_profiled, grad_learned, atol=2e-6, rtol=2e-5)


def test_fixed_mode_and_legacy_trace_beta_are_rejected() -> None:
    with pytest.raises(ValueError, match="profiled, learned"):
        _constant_energy_model(beta_mode="fixed")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="trace_beta"):
        _constant_energy_model(trace_beta=True)

    model = _constant_energy_model()
    with pytest.raises(ValueError, match="profiled, learned"):
        model.free_energy(torch.ones(1, 2), beta_mode="fixed")  # type: ignore[arg-type]


def test_learned_mode_requires_trainable_beta() -> None:
    profiled = _constant_energy_model(beta_mode="profiled")
    with pytest.raises(ValueError, match="requires"):
        profiled.free_energy(torch.ones(1, 2), beta_mode="learned")


def test_beta_mode_override_and_loss_epsilon_are_validated() -> None:
    model = _constant_energy_model()
    with pytest.raises(ValueError, match="beta_mode"):
        model.free_energy(torch.ones(1, 2), beta_mode="typo")  # type: ignore[arg-type]
    for loss_eps in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            _constant_energy_model(loss_eps=loss_eps)
