"""Conditional Bernoulli support inference for small SAE diagnostics.

The model conditions on a dictionary D and point amplitudes a, and uses
``x | s, a, D ~ N(D(a*s), beta**-1 I)`` with independent support prior
``P(s_j=1) = sigmoid(-gamma)``. Input-dependent amplitudes do not turn this
conditional diagnostic into a normalized generative model over x.

This generalizes the unit-amplitude math in idea_discovery_posterior_pilot.
Enumeration is an exponential, tiny-system reference, not a scalable method.
Coordinate mean-field is deterministic and returns the best solution found
among supplied starts; it does not certify the global variational optimum.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _inputs(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    amplitudes: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    if x.ndim != 2 or dictionary.ndim != 2:
        raise ValueError("x and dictionary must have shapes (batch, d) and (d, K)")
    if x.shape[0] == 0 or x.shape[1] == 0 or dictionary.shape[1] == 0:
        raise ValueError("batch, observation dimension, and support width must be nonzero")
    if x.shape[1] != dictionary.shape[0]:
        raise ValueError("x and dictionary observation dimensions differ")
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(gamma):
        raise ValueError("beta must be finite and positive; gamma must be finite")
    for name, tensor in (("x", x), ("dictionary", dictionary), ("amplitudes", amplitudes)):
        if not tensor.is_floating_point() or tensor.dtype != x.dtype or tensor.device != x.device:
            raise ValueError(f"{name} must use the same floating dtype and device as x")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
    if amplitudes.ndim not in (1, 2) or amplitudes.shape[-1] != dictionary.shape[1]:
        raise ValueError("amplitudes must have shape (K,), (1, K), or (batch, K)")
    try:
        return torch.broadcast_to(amplitudes, (x.shape[0], dictionary.shape[1]))
    except RuntimeError as error:
        raise ValueError("amplitudes cannot broadcast to (batch, K)") from error


def _probabilities(m: torch.Tensor, shape: torch.Size, x: torch.Tensor) -> torch.Tensor:
    if m.dtype != x.dtype or m.device != x.device:
        raise ValueError("support probabilities must use the dtype and device of x")
    if m.ndim not in (1, 2) or m.shape[-1] != shape[-1]:
        raise ValueError("support probabilities must have shape (K,), (1, K), or (batch, K)")
    try:
        m = torch.broadcast_to(m, shape)
    except RuntimeError as error:
        raise ValueError("support probabilities cannot broadcast to (batch, K)") from error
    if not torch.isfinite(m).all() or (m < 0).any() or (m > 1).any():
        raise ValueError("support probabilities must be finite and within [0, 1]")
    return m


def _free_energy(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    amplitudes: torch.Tensor,
    m: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    squared_error = (x - (amplitudes * m) @ dictionary.T).square().sum(-1)
    variance = (m * (1 - m) * amplitudes.square() * dictionary.square().sum(0)).sum(-1)
    prior_cost = (m * _softplus(gamma) + (1 - m) * _softplus(-gamma)).sum(-1)
    negative_entropy = (
        torch.special.xlogy(m, m) + torch.special.xlogy(1 - m, 1 - m)
    ).sum(-1)
    gaussian_normalizer = -0.5 * x.shape[1] * math.log(beta / (2 * math.pi))
    return 0.5 * beta * (squared_error + variance) + gaussian_normalizer + prior_cost + negative_entropy


@torch.no_grad()
def enumerate_support_posterior(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    amplitudes: torch.Tensor,
    *,
    beta: float,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Enumerate a normalized conditional posterior with all inputs fixed.

    Returns states (2**K, K), log_joint/log_p/p (batch, 2**K), m (batch, K),
    log_evidence/signal_variance (batch,), and covariance (batch, K, K).
    Signal variance is E[||D(a*s) - E[D(a*s)]||²], including covariances.
    Finite gamma values remain stable even when sigmoid(-gamma) rounds to an
    endpoint. Memory and computation grow exponentially in K.
    """
    amplitudes = _inputs(x, dictionary, amplitudes, beta, gamma)
    k = dictionary.shape[1]
    if k > 20:
        raise ValueError("exact enumeration is limited to tiny systems (K <= 20)")
    indices = torch.arange(2**k, device=x.device, dtype=torch.long)
    bits = torch.arange(k, device=x.device, dtype=torch.long)
    states = ((indices[:, None] >> bits) & 1).to(x.dtype)
    signal = (amplitudes[:, None, :] * states[None, :, :]) @ dictionary.T
    squared_error = (x[:, None, :] - signal).square().sum(-1)
    log_prior = -(states * _softplus(gamma) + (1 - states) * _softplus(-gamma)).sum(-1)
    log_joint = (
        -0.5 * beta * squared_error
        + 0.5 * x.shape[1] * math.log(beta / (2 * math.pi))
        + log_prior
    )
    log_evidence = torch.logsumexp(log_joint, -1)
    log_p = log_joint - log_evidence[:, None]
    p = log_p.exp()
    m = p @ states
    signal_mean = (amplitudes * m) @ dictionary.T
    signal_variance = (p * (signal - signal_mean[:, None, :]).square().sum(-1)).sum(-1)
    second_moment = torch.einsum("bs,si,sj->bij", p, states, states)
    covariance = second_moment - m[:, :, None] * m[:, None, :]
    return dict(
        states=states, log_joint=log_joint, log_evidence=log_evidence,
        log_p=log_p, p=p, m=m, signal_variance=signal_variance,
        covariance=covariance,
    )


def factorized_free_energy(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    amplitudes: torch.Tensor,
    m: torch.Tensor,
    *,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """Full normalized negative ELBO for product Bernoulli q, per sample.

    No gradients are taken or parameters changed internally. Unlike the
    inference solvers, this expression preserves autograd for validation or
    gate training. At m=0 or 1 its value is finite, with 0*log(0)=0; interior
    probabilities should be used when requesting finite entropy gradients.
    """
    amplitudes = _inputs(x, dictionary, amplitudes, beta, gamma)
    m = _probabilities(m, amplitudes.shape, x)
    return _free_energy(x, dictionary, amplitudes, m, beta, gamma)


def _fixed_point(
    projected: torch.Tensor,
    gram: torch.Tensor,
    amplitudes: torch.Tensor,
    m: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    diagonal = gram.diagonal()
    code = amplitudes * m
    excluding_self = projected - code @ gram + code * diagonal
    return torch.sigmoid(
        beta * amplitudes * excluding_self - 0.5 * beta * amplitudes.square() * diagonal - gamma
    )


@torch.no_grad()
def mean_field_fixed_point(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    amplitudes: torch.Tensor,
    m: torch.Tensor,
    *,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """Simultaneous coordinate-optimal probabilities at a fixed input m.

    This map is useful for stationarity residuals. Simultaneously applying it
    is not the coordinate descent algorithm and need not decrease free energy.
    """
    amplitudes = _inputs(x, dictionary, amplitudes, beta, gamma)
    m = _probabilities(m, amplitudes.shape, x)
    return _fixed_point(x @ dictionary, dictionary.T @ dictionary, amplitudes, m, beta, gamma)


@torch.no_grad()
def best_found_mean_field(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    amplitudes: torch.Tensor,
    *,
    beta: float,
    gamma: float,
    starts: Sequence[torch.Tensor],
    max_sweeps: int = 100,
    tolerance: float = 1e-9,
) -> dict[str, torch.Tensor]:
    """Cyclic coordinate descent, selecting the lowest energy start per sample.

    All inputs remain fixed and unmodified. No truth labels enter inference.
    A sweep updates coordinates in index order. Each sample stops when its
    max absolute fixed-point residual <= tolerance or at max_sweeps.

    Returned m has shape (batch, K); free_energy, residual, selected_start,
    sweeps, and converged have shape (batch,). Initial/start diagnostics use
    shape (batch, number_of_starts). Starts may be (K,), (1, K), or (batch, K).
    Including an encoder's probabilities among starts permits a gate-only
    refinement comparison, but does not certify the exact amortization gap.
    """
    amplitudes = _inputs(x, dictionary, amplitudes, beta, gamma)
    if not starts:
        raise ValueError("at least one mean-field start is required")
    if not isinstance(max_sweeps, int) or max_sweeps < 0:
        raise ValueError("max_sweeps must be a nonnegative integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    gram = dictionary.T @ dictionary
    projected = x @ dictionary
    diagonal = gram.diagonal()
    candidates, energies, residuals, counts, initial_energies = [], [], [], [], []
    for start in starts:
        m = _probabilities(start, amplitudes.shape, x).clone()
        initial_energies.append(_free_energy(x, dictionary, amplitudes, m, beta, gamma))
        residual = (m - _fixed_point(projected, gram, amplitudes, m, beta, gamma)).abs().amax(-1)
        sweeps = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        active = residual > tolerance
        for _ in range(max_sweeps):
            if not active.any():
                break
            for j in range(dictionary.shape[1]):
                excluding_j = projected[:, j] - (amplitudes * m) @ gram[:, j] + amplitudes[:, j] * m[:, j] * diagonal[j]
                updated = torch.sigmoid(
                    beta * amplitudes[:, j] * excluding_j
                    - 0.5 * beta * amplitudes[:, j].square() * diagonal[j] - gamma
                )
                m[:, j] = torch.where(active, updated, m[:, j])
            sweeps += active.to(torch.long)
            residual = (m - _fixed_point(projected, gram, amplitudes, m, beta, gamma)).abs().amax(-1)
            active = residual > tolerance
        candidates.append(m)
        energies.append(_free_energy(x, dictionary, amplitudes, m, beta, gamma))
        residuals.append(residual)
        counts.append(sweeps)
    all_energy = torch.stack(energies, 1)
    all_residual = torch.stack(residuals, 1)
    all_sweeps = torch.stack(counts, 1)
    chosen = all_energy.argmin(1)
    rows = torch.arange(x.shape[0], device=x.device)
    return dict(
        m=torch.stack(candidates, 1)[rows, chosen],
        free_energy=all_energy[rows, chosen],
        residual=all_residual[rows, chosen],
        selected_start=chosen,
        sweeps=all_sweeps[rows, chosen],
        converged=all_residual[rows, chosen] <= tolerance,
        initial_start_free_energy=torch.stack(initial_energies, 1),
        all_start_free_energy=all_energy,
        all_start_residual=all_residual,
        all_start_sweeps=all_sweeps,
    )
