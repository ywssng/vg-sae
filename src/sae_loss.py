"""Losses for VG-SAE and SAE baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .sae_model import GatedSAE, L1ReLUSAE, TopKSAE, VariationalGarroteSAE


@dataclass
class VGSAELossTerms:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    reconstruction_mse: torch.Tensor
    variance_loss: torch.Tensor
    prior_loss: torch.Tensor
    entropy_loss: torch.Tensor
    entropy: torch.Tensor
    beta: torch.Tensor
    rho: torch.Tensor
    v_eff: torch.Tensor


@dataclass
class BaselineSAELossTerms:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    reconstruction_mse: torch.Tensor
    sparsity_loss: torch.Tensor
    rho: torch.Tensor


def bernoulli_kl_from_lambda(
    m: torch.Tensor,
    lambda_sparsity: float | torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """KL(Bern(m) || Bern(pi)) with pi = sigmoid(-lambda)."""
    m_safe = m.clamp(eps, 1.0 - eps)
    lam = torch.as_tensor(lambda_sparsity, dtype=m.dtype, device=m.device)
    negative_entropy = m_safe * torch.log(m_safe) + (1.0 - m_safe) * torch.log1p(-m_safe)
    prior_cross_entropy = m_safe * F.softplus(lam) + (1.0 - m_safe) * F.softplus(-lam)
    return negative_entropy + prior_cross_entropy


def vg_sae_loss_terms(model: VariationalGarroteSAE, x: torch.Tensor) -> VGSAELossTerms:
    output = model.free_energy(x)
    m = output["m"]
    entropy_weight = float(model.config.entropy_weight) if model.config.use_entropy_term else 0.0
    entropy_loss = -entropy_weight * output["entropy"]

    return VGSAELossTerms(
        loss=output["loss"],
        reconstruction_loss=output["recon"],
        reconstruction_mse=2.0 * output["recon"] / float(model.config.input_dim),
        variance_loss=output["variance"],
        prior_loss=output["prior"],
        entropy_loss=entropy_loss,
        entropy=output["entropy"],
        beta=output["beta_eff"],
        rho=m.mean(),
        v_eff=(m * (1.0 - m)).mean(),
    )


def l1_sae_loss_terms(model: L1ReLUSAE, x: torch.Tensor) -> BaselineSAELossTerms:
    output = model(x)
    residual_square = (x - output["x_hat"]).pow(2).sum(dim=1)
    reconstruction_loss = 0.5 * residual_square.mean()
    h = output["h"]
    sparsity_loss = float(model.config.l1_coefficient) * h.abs().sum(dim=1).mean()
    return BaselineSAELossTerms(
        loss=reconstruction_loss + sparsity_loss,
        reconstruction_loss=reconstruction_loss,
        reconstruction_mse=residual_square.mean() / float(model.config.input_dim),
        sparsity_loss=sparsity_loss,
        rho=(h > 0.0).to(x.dtype).mean(),
    )


def topk_sae_loss_terms(model: TopKSAE, x: torch.Tensor) -> BaselineSAELossTerms:
    output = model(x)
    residual_square = (x - output["x_hat"]).pow(2).sum(dim=1)
    reconstruction_loss = 0.5 * residual_square.mean()
    mask = output["mask"]
    return BaselineSAELossTerms(
        loss=reconstruction_loss,
        reconstruction_loss=reconstruction_loss,
        reconstruction_mse=residual_square.mean() / float(model.config.input_dim),
        sparsity_loss=x.new_zeros(()),
        rho=mask.mean(),
    )


def gated_sae_loss_terms(model: GatedSAE, x: torch.Tensor) -> BaselineSAELossTerms:
    output = model(x)
    residual_square = (x - output["x_hat"]).pow(2).sum(dim=1)
    reconstruction_loss = 0.5 * residual_square.mean()
    gate = output["gate"]
    sparsity_loss = float(model.config.l1_coefficient) * gate.sum(dim=1).mean()
    return BaselineSAELossTerms(
        loss=reconstruction_loss + sparsity_loss,
        reconstruction_loss=reconstruction_loss,
        reconstruction_mse=residual_square.mean() / float(model.config.input_dim),
        sparsity_loss=sparsity_loss,
        rho=gate.mean(),
    )


def sae_loss_terms(model: torch.nn.Module, x: torch.Tensor) -> VGSAELossTerms | BaselineSAELossTerms:
    if isinstance(model, VariationalGarroteSAE):
        return vg_sae_loss_terms(model, x)
    if isinstance(model, L1ReLUSAE):
        return l1_sae_loss_terms(model, x)
    if isinstance(model, TopKSAE):
        return topk_sae_loss_terms(model, x)
    if isinstance(model, GatedSAE):
        return gated_sae_loss_terms(model, x)
    raise TypeError(f"Unsupported SAE model type: {type(model).__name__}")
