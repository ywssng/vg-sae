"""Diagnostics for VG-SAE synthetic and activation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .sae_model import VariationalGarroteSAE


@dataclass
class SAEObservables:
    mse: float
    rho: float
    entropy: float
    v_eff: float
    dead_fraction: float
    interference_energy: float
    variance_energy: float


@torch.no_grad()
def vg_sae_observables(
    model: VariationalGarroteSAE,
    x: torch.Tensor,
    dead_threshold: float = 1.0e-6,
) -> SAEObservables:
    model.eval()
    output = model(x)
    m = output["m"]
    a = output["a"]
    h = output["h"]
    x_hat = output["x_hat"]
    eps = max(float(model.config.loss_eps), torch.finfo(x.dtype).eps)
    m_safe = m.clamp(eps, 1.0 - eps)
    entropy = (-m_safe * torch.log(m_safe) - (1.0 - m_safe) * torch.log1p(-m_safe)).mean()
    decoder_norm_square = model.decoder_column_sqnorms()
    variance_energy = 0.5 * (m * (1.0 - m) * a.pow(2) * decoder_norm_square).sum(dim=1).mean()

    gram_square = (model.decoder.weight.T @ model.decoder.weight).pow(2)
    gram_square.fill_diagonal_(0.0)
    interference = 0.5 * (h @ gram_square * h).sum(dim=1).mean()

    return SAEObservables(
        mse=float((x - x_hat).pow(2).mean().detach().cpu()),
        rho=float(m.mean().detach().cpu()),
        entropy=float(entropy.detach().cpu()),
        v_eff=float((m * (1.0 - m)).mean().detach().cpu()),
        dead_fraction=float((h.mean(dim=0) <= dead_threshold).to(torch.float32).mean().detach().cpu()),
        interference_energy=float(interference.detach().cpu()),
        variance_energy=float(variance_energy.detach().cpu()),
    )


@torch.no_grad()
def feature_uncertainty(model: VariationalGarroteSAE, x: torch.Tensor) -> torch.Tensor:
    m = model(x)["m"]
    return (m * (1.0 - m)).mean(dim=0)


def susceptibility(lambda_values: np.ndarray, rho_values: np.ndarray) -> np.ndarray:
    lambdas = np.asarray(lambda_values, dtype=np.float64)
    rhos = np.asarray(rho_values, dtype=np.float64)
    if lambdas.ndim != 1 or rhos.ndim != 1 or lambdas.shape != rhos.shape:
        raise ValueError("lambda_values and rho_values must be one-dimensional arrays with matching shape.")
    if lambdas.shape[0] < 2:
        return np.zeros_like(rhos)
    order = np.argsort(lambdas)
    chi_sorted = -np.gradient(rhos[order], lambdas[order])
    chi = np.empty_like(chi_sorted)
    chi[order] = chi_sorted
    return chi


def decoder_cosine_matrix(learned_dictionary: torch.Tensor, true_dictionary: torch.Tensor) -> np.ndarray:
    learned = learned_dictionary.detach().cpu()
    true = true_dictionary.detach().cpu()
    learned = learned / learned.norm(dim=0, keepdim=True).clamp_min(1.0e-12)
    true = true / true.norm(dim=0, keepdim=True).clamp_min(1.0e-12)
    return torch.abs(learned.T @ true).numpy()


def decoder_recovery_cosine(learned_dictionary: torch.Tensor, true_dictionary: torch.Tensor) -> float:
    cosines = decoder_cosine_matrix(learned_dictionary, true_dictionary)
    try:
        from scipy.optimize import linear_sum_assignment

        row_idx, col_idx = linear_sum_assignment(-cosines)
        return float(cosines[row_idx, col_idx].mean()) if row_idx.size else 0.0
    except Exception:
        return float(cosines.max(axis=1).mean()) if cosines.size else 0.0


@torch.no_grad()
def support_precision_recall(
    model: VariationalGarroteSAE,
    x: torch.Tensor,
    true_support: torch.Tensor,
    true_dictionary: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Compare matched learned support to true support for synthetic data."""
    m = model(x)["m"].detach().cpu()
    true_support_cpu = true_support.detach().cpu()
    cosines = decoder_cosine_matrix(model.decoder.weight, true_dictionary)
    try:
        from scipy.optimize import linear_sum_assignment

        learned_idx, true_idx = linear_sum_assignment(-cosines)
    except Exception:
        learned_idx = np.arange(cosines.shape[0])
        true_idx = cosines.argmax(axis=1)

    pred = m[:, learned_idx] > threshold
    target = true_support_cpu[:, true_idx] > 0.5
    tp = torch.logical_and(pred, target).sum().item()
    fp = torch.logical_and(pred, torch.logical_not(target)).sum().item()
    fn = torch.logical_and(torch.logical_not(pred), target).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(precision), float(recall)


@torch.no_grad()
def amplitude_shrinkage(
    model: VariationalGarroteSAE,
    x: torch.Tensor,
    true_z: torch.Tensor,
    true_dictionary: torch.Tensor,
    eps: float = 1.0e-8,
) -> float:
    output = model(x)
    h = output["h"].detach().cpu()
    true_z_cpu = true_z.detach().cpu()
    cosines = decoder_cosine_matrix(model.decoder.weight, true_dictionary)
    try:
        from scipy.optimize import linear_sum_assignment

        learned_idx, true_idx = linear_sum_assignment(-cosines)
    except Exception:
        learned_idx = np.arange(cosines.shape[0])
        true_idx = cosines.argmax(axis=1)
    target = true_z_cpu[:, true_idx]
    active = target > eps
    if not active.any():
        return 0.0
    ratio = h[:, learned_idx][active] / target[active].clamp_min(eps)
    return float(ratio.mean().item())
