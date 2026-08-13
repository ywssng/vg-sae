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
def decoder_pairwise_cosine_similarity(
    decoder_atoms: torch.Tensor,
    *,
    block_size: int = 256,
) -> float:
    """Return Eq. (4)'s mean absolute cosine over decoder-atom pairs.

    ``decoder_atoms`` must use one row per SAE latent.  Blocks keep the exact
    computation practical for wide decoders without materializing the full
    square Gram matrix and its upper-triangle mask at once.
    """

    if decoder_atoms.ndim != 2:
        raise ValueError("decoder_atoms must be a two-dimensional tensor.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    width = int(decoder_atoms.shape[0])
    if width < 2:
        raise ValueError("decoder pairwise cosine requires at least two atoms.")
    atoms = decoder_atoms.detach().float()
    norms = atoms.norm(dim=1, keepdim=True)
    if not torch.isfinite(atoms).all() or not torch.isfinite(norms).all():
        raise ValueError("decoder atoms and their L2 norms must be finite.")
    # This matches the paper's Appendix-M F.normalize convention: an exactly
    # zero decoder row stays zero and contributes zero similarity to its pairs.
    atoms = torch.nn.functional.normalize(atoms, p=2.0, dim=1)
    pair_sum = torch.zeros((), dtype=torch.float64, device=atoms.device)
    for start in range(0, width, block_size):
        stop = min(start + block_size, width)
        within = atoms[start:stop] @ atoms[start:stop].T
        pair_sum += within.triu(diagonal=1).abs().sum(dtype=torch.float64)
        if stop < width:
            across = atoms[start:stop] @ atoms[stop:].T
            pair_sum += across.abs().sum(dtype=torch.float64)
    pair_count = width * (width - 1) // 2
    return float((pair_sum / pair_count).cpu())


def decoder_atoms_from_model(model: torch.nn.Module) -> torch.Tensor:
    """Return decoder atoms row-wise for local VG and SAELens SAE models."""

    if isinstance(model, VariationalGarroteSAE):
        return model.decoder.weight.T
    atoms = getattr(model, "W_dec", None)
    if isinstance(atoms, torch.Tensor):
        return atoms
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


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
