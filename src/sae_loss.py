"""Loss reporting for VG-SAE and official SAELens baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE
from sae_lens.saes.sae import (
    TrainCoefficientConfig,
    TrainingSAE,
    TrainStepInput,
)

from .sae_model import VariationalGarroteSAE


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
    details: dict[str, torch.Tensor] = field(default_factory=dict)


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
    return VGSAELossTerms(
        loss=output["loss"],
        reconstruction_loss=output["recon"],
        reconstruction_mse=2.0 * output["recon"] / float(model.config.input_dim),
        variance_loss=output["variance"],
        prior_loss=output["prior"],
        entropy_loss=-entropy_weight * output["entropy"],
        entropy=output["entropy"],
        beta=output["beta_eff"],
        rho=m.mean(),
        v_eff=(m * (1.0 - m)).mean(),
    )


def _final_coefficients(model: TrainingSAE) -> dict[str, float]:
    return {
        name: float(value.value if isinstance(value, TrainCoefficientConfig) else value)
        for name, value in model.get_coefficients().items()
    }


def saelens_sae_loss_terms(
    model: TrainingSAE,
    x: torch.Tensor,
    dead_feature_mask: torch.Tensor | None = None,
    *,
    coefficients: dict[str, float] | None = None,
    n_training_steps: int = 0,
    update_state: bool = False,
) -> BaselineSAELossTerms:
    """Expose an official training forward pass through project metric names.

    Read-only BatchTopK reporting deliberately bypasses only its EMA update; its
    encoding and losses still dispatch to the official architecture methods.
    """

    step_input = TrainStepInput(
        sae_in=x,
        coefficients=coefficients or _final_coefficients(model),
        dead_neuron_mask=dead_feature_mask,
        n_training_steps=n_training_steps,
        is_logging_step=False,
    )
    if isinstance(model, BatchTopKTrainingSAE) and not update_state:
        output = TrainingSAE.training_forward_pass(model, step_input)
    else:
        output = model.training_forward_pass(step_input)

    feature_acts = (
        output.feature_acts.to_dense()
        if output.feature_acts.is_sparse
        else output.feature_acts
    )
    zero = x.new_zeros(())
    reconstruction_loss = output.losses.get("mse_loss")
    if reconstruction_loss is None:
        reconstruction_loss = (output.sae_in - output.sae_out).pow(2).sum(dim=-1).mean()
        sparsity_loss = zero
        auxiliary_loss = output.loss - reconstruction_loss
    else:
        sparsity_loss = sum(
            (
                output.losses[name]
                for name in ("l0_loss", "l1_loss")
                if name in output.losses
            ),
            zero,
        )
        auxiliary_loss = sum(
            (
                value
                for name, value in output.losses.items()
                if name not in {"mse_loss", "l0_loss", "l1_loss"}
            ),
            zero,
        )
    return BaselineSAELossTerms(
        loss=output.loss,
        reconstruction_loss=reconstruction_loss,
        reconstruction_mse=reconstruction_loss / float(model.cfg.d_in),
        sparsity_loss=sparsity_loss,
        auxiliary_loss=auxiliary_loss,
        rho=feature_acts.bool().to(x.dtype).mean(),
        feature_acts=feature_acts,
        details={**output.losses, **output.metrics},
    )


def sae_loss_terms(
    model: torch.nn.Module,
    x: torch.Tensor,
    dead_feature_mask: torch.Tensor | None = None,
) -> VGSAELossTerms | BaselineSAELossTerms:
    if isinstance(model, VariationalGarroteSAE):
        return vg_sae_loss_terms(model, x)
    if isinstance(model, TrainingSAE):
        return saelens_sae_loss_terms(model, x, dead_feature_mask)
    raise TypeError(f"Unsupported SAE model type: {type(model).__name__}")
