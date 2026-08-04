"""Losses for VG-SAE and SAE baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .sae_model import (
    BatchTopKSAE,
    GatedSAE,
    JumpReLUSAE,
    L1ReLUSAE,
    Step,
    TopKSAE,
    VariationalGarroteSAE,
)


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
    auxiliary_loss: torch.Tensor
    rho: torch.Tensor
    feature_acts: torch.Tensor


def bernoulli_kl_from_lambda(
    m: torch.Tensor,
    lambda_sparsity: float | torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """KL(Bern(m) || Bern(pi)) with pi = sigmoid(-lambda)."""
    requested_eps = float(eps)
    if not 0.0 < requested_eps < 0.5:
        raise ValueError("eps must be finite and lie in (0, 0.5).")
    eps_value = max(requested_eps, float(torch.finfo(m.dtype).eps))
    m_safe = m.clamp(eps_value, 1.0 - eps_value)
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
    reconstruction_loss = residual_square.mean()
    h = output["h"]
    sparsity_loss = float(model.config.l1_coefficient) * (
        h.abs() * model.decoder_column_norms()
    ).sum(dim=1).mean()
    return BaselineSAELossTerms(
        loss=reconstruction_loss + sparsity_loss,
        reconstruction_loss=reconstruction_loss,
        reconstruction_mse=residual_square.mean() / float(model.config.input_dim),
        sparsity_loss=sparsity_loss,
        auxiliary_loss=x.new_zeros(()),
        rho=(h > 0.0).to(x.dtype).mean(),
        feature_acts=h,
    )


def topk_sae_loss_terms(
    model: TopKSAE,
    x: torch.Tensor,
    dead_feature_mask: torch.Tensor | None = None,
) -> BaselineSAELossTerms:
    output = model(x)
    residual_square = (x - output["x_hat"]).pow(2).sum(dim=1)
    reconstruction_loss = residual_square.mean()
    auxiliary_loss = model.auxiliary_loss(x, output, dead_feature_mask)
    h = output["h"]
    return BaselineSAELossTerms(
        loss=reconstruction_loss + auxiliary_loss,
        reconstruction_loss=reconstruction_loss,
        reconstruction_mse=residual_square.mean() / float(model.config.input_dim),
        sparsity_loss=x.new_zeros(()),
        auxiliary_loss=auxiliary_loss,
        rho=(h > 0).to(x).mean(),
        feature_acts=h,
    )


def gated_sae_loss_terms(model: GatedSAE, x: torch.Tensor) -> BaselineSAELossTerms:
    output = model(x)
    residual_square = (x - output["x_hat"]).pow(2).sum(dim=1)
    reconstruction_loss = residual_square.mean()
    gate_magnitudes = F.relu(output["pi_gate"])
    sparsity_loss = float(model.config.l1_coefficient) * (
        gate_magnitudes * model.decoder_column_norms()
    ).sum(dim=1).mean()
    gate_hat = model.decode(gate_magnitudes)
    auxiliary_loss = (x - gate_hat).pow(2).sum(dim=1).mean()
    return BaselineSAELossTerms(
        loss=reconstruction_loss + sparsity_loss + auxiliary_loss,
        reconstruction_loss=reconstruction_loss,
        reconstruction_mse=residual_square.mean() / float(model.config.input_dim),
        sparsity_loss=sparsity_loss,
        auxiliary_loss=auxiliary_loss,
        rho=output["mask"].mean(),
        feature_acts=output["h"],
    )


def jumprelu_sae_loss_terms(
    model: JumpReLUSAE,
    x: torch.Tensor,
    dead_feature_mask: torch.Tensor | None = None,
) -> BaselineSAELossTerms:
    output = model(x)
    residual_square = (x - output["x_hat"]).pow(2).sum(dim=1)
    reconstruction_loss = residual_square.mean()
    cfg, h, hidden_pre = model.config, output["h"], output["hidden_pre"]
    if cfg.jumprelu_sparsity_loss_mode == "step":
        count = Step.apply(hidden_pre, model.threshold, cfg.jumprelu_bandwidth).sum(dim=1)
    elif cfg.jumprelu_sparsity_loss_mode == "tanh":
        count = torch.tanh(
            cfg.jumprelu_tanh_scale * h * model.decoder_column_norms()
        ).sum(dim=1)
    else:  # defensive against a mutated config
        raise ValueError(f"Invalid sparsity loss mode: {cfg.jumprelu_sparsity_loss_mode}")
    sparsity_loss = float(cfg.l0_coefficient) * count.mean()
    auxiliary_loss = x.new_zeros(())
    if cfg.pre_act_loss_coefficient is not None and dead_feature_mask is not None:
        dead = dead_feature_mask.to(device=x.device, dtype=x.dtype)
        pre_act = (
            F.relu(model.threshold - hidden_pre) * dead * model.decoder_column_norms()
        ).sum(dim=1).mean()
        auxiliary_loss = float(cfg.pre_act_loss_coefficient) * pre_act
    return BaselineSAELossTerms(
        loss=reconstruction_loss + sparsity_loss + auxiliary_loss,
        reconstruction_loss=reconstruction_loss,
        reconstruction_mse=residual_square.mean() / float(cfg.input_dim),
        sparsity_loss=sparsity_loss,
        auxiliary_loss=auxiliary_loss,
        rho=(h > 0).to(x).mean(),
        feature_acts=h,
    )


def sae_loss_terms(
    model: torch.nn.Module,
    x: torch.Tensor,
    dead_feature_mask: torch.Tensor | None = None,
) -> VGSAELossTerms | BaselineSAELossTerms:
    if isinstance(model, VariationalGarroteSAE):
        return vg_sae_loss_terms(model, x)
    if isinstance(model, L1ReLUSAE):
        return l1_sae_loss_terms(model, x)
    if isinstance(model, BatchTopKSAE):
        return topk_sae_loss_terms(model, x, dead_feature_mask)
    if isinstance(model, TopKSAE):
        return topk_sae_loss_terms(model, x, dead_feature_mask)
    if isinstance(model, JumpReLUSAE):
        return jumprelu_sae_loss_terms(model, x, dead_feature_mask)
    if isinstance(model, GatedSAE):
        return gated_sae_loss_terms(model, x)
    raise TypeError(f"Unsupported SAE model type: {type(model).__name__}")
