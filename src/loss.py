"""Reduced free-energy objective for Variational Garrote."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import VariationalGarrote


@dataclass
class VGLossTerms:
    """Named components of the beta-eliminated VG free energy."""

    free_energy: torch.Tensor
    energy: torch.Tensor
    reconstruction_sum: torch.Tensor
    variance_sum: torch.Tensor
    entropy: torch.Tensor
    sparsity_penalty: torch.Tensor
    beta_estimate: torch.Tensor
    rho_model: torch.Tensor


def _prepare_targets(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"Expected y with shape (n_samples,), got {tuple(y.shape)}.")
    return y


def energy_components(
    m: torch.Tensor,
    w: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute E(m, w) and its reconstruction/variance components."""
    y = _prepare_targets(y)
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (n_samples, n_features), got {tuple(x.shape)}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")
    if x.shape[1] != m.shape[0] or m.shape != w.shape:
        raise ValueError("m, w, and x feature dimensions must match.")

    residual = y - x @ (m * w)
    reconstruction_sum = residual.pow(2).sum()
    x_square_sum = x.pow(2).sum(dim=0)
    variance_sum = (m * (1.0 - m) * w.pow(2) * x_square_sum).sum()
    energy = 0.5 * (reconstruction_sum + variance_sum)
    return energy, reconstruction_sum, variance_sum


def energy_term(
    m: torch.Tensor,
    w: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Paper data energy E(m, w)."""
    energy, _, _ = energy_components(m=m, w=w, x=x, y=y)
    return energy


def vg_loss_terms(
    model: VariationalGarrote,
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float | None = None,
) -> VGLossTerms:
    """Compute the prior-corrected beta-eliminated VG objective.

    The printed reduced equation contains `- gamma * sum(m)`, but the selector
    prior P(s_i) proportional to exp(-gamma s_i) and the paper prose imply a
    positive penalty in the negative log posterior. This implementation uses
    `+ gamma * sum(m)` so larger positive gamma encourages sparsity.
    """
    y = _prepare_targets(y)
    m = model.mask()
    w = model.weight
    gamma_value = float(model.config.gamma if gamma is None else gamma)
    eps = max(float(model.config.loss_eps), torch.finfo(x.dtype).eps)

    energy, reconstruction_sum, variance_sum = energy_components(m=m, w=w, x=x, y=y)
    safe_energy = energy.clamp_min(eps)
    m_safe = m.clamp(eps, 1.0 - eps)

    entropy = (-m_safe * torch.log(m_safe) - (1.0 - m_safe) * torch.log1p(-m_safe)).sum()
    sparsity_penalty = gamma_value * m.sum()
    free_energy = 0.5 * x.shape[0] * torch.log(safe_energy) - entropy + sparsity_penalty
    beta_estimate = x.new_tensor(x.shape[0] / (2.0 * float(safe_energy.detach().cpu())))

    return VGLossTerms(
        free_energy=free_energy,
        energy=energy,
        reconstruction_sum=reconstruction_sum,
        variance_sum=variance_sum,
        entropy=entropy,
        sparsity_penalty=sparsity_penalty,
        beta_estimate=beta_estimate,
        rho_model=m.mean(),
    )


def vg_free_energy(
    model: VariationalGarrote,
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float | None = None,
) -> torch.Tensor:
    return vg_loss_terms(model=model, x=x, y=y, gamma=gamma).free_energy


def free_energy_loss(
    model: VariationalGarrote,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compatibility wrapper returning `(loss, scalar_components)`."""
    terms = vg_loss_terms(model=model, x=x, y=y)
    return terms.free_energy, {
        "E": float(terms.energy.detach().cpu()),
        "reconstruction_sum": float(terms.reconstruction_sum.detach().cpu()),
        "variance_sum": float(terms.variance_sum.detach().cpu()),
        "entropy": float(terms.entropy.detach().cpu()),
        "sparsity_reg": float(terms.sparsity_penalty.detach().cpu()),
        "beta_est": float(terms.beta_estimate.detach().cpu()),
        "rho_model": float(terms.rho_model.detach().cpu()),
    }
