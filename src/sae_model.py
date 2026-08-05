"""Sparse autoencoder models for VG-SAE experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _dtype_from_value
from .sae_baselines import (
    BatchTopKSAE,
    BatchTopKSAEConfig,
    GatedSAE,
    GatedSAEConfig,
    JumpReLU,
    JumpReLUSAE,
    JumpReLUSAEConfig,
    L1ReLUSAE,
    L1SAEConfig,
    Step,
    TopKSAE,
    TopKSAEConfig,
)


@dataclass
class VGSAEConfig:
    """Configuration for the amortized vector-output VG-SAE."""

    input_dim: int
    n_latents: int

    # VG objective. ``lambda_sparsity`` is the signed gamma sparsity field.
    beta: float = 1.0
    lambda_sparsity: float = 1.0
    use_variance_term: bool = True
    use_entropy_term: bool = True
    entropy_weight: float = 1.0
    beta_mode: Literal["profiled", "fixed", "learned"] | None = None
    trace_beta: bool | None = True

    # Architecture.
    decoder_bias: bool = True
    tie_encoder_init: bool = True
    gate_bias_init: float = -2.0
    amplitude_bias_init: float = 0.0
    nonnegative_amplitudes: bool = True

    # Numerics / inference.
    normalize_decoder: bool = True
    loss_eps: float = 1.0e-8
    inference_threshold: float = 0.5
    dtype: torch.dtype | str = torch.float32

    @property
    def torch_dtype(self) -> torch.dtype:
        return _dtype_from_value(self.dtype)

    def validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.n_latents <= 0:
            raise ValueError("n_latents must be positive.")
        if not math.isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("beta must be positive.")
        if not math.isfinite(self.lambda_sparsity):
            raise ValueError("lambda_sparsity/gamma must be finite.")
        if self.beta_mode not in {None, "profiled", "fixed", "learned"}:
            raise ValueError("beta_mode must be one of: profiled, fixed, learned.")
        if not math.isfinite(self.entropy_weight) or self.entropy_weight < 0.0:
            raise ValueError("entropy_weight must be non-negative.")
        if not math.isfinite(self.loss_eps) or self.loss_eps <= 0.0:
            raise ValueError("loss_eps must be positive and finite.")
        if not (0.0 <= self.inference_threshold <= 1.0):
            raise ValueError("inference_threshold must lie in [0, 1].")


class VariationalGarroteSAE(nn.Module):
    """Amortized VG sparse autoencoder for vector-valued reconstructions.

    For each input vector, the encoder parameterizes a mean-field Bernoulli
    posterior over sample-specific support variables and deterministic latent
    amplitudes. The decoder columns play the role of vector-valued regression
    weights in the paper's scalar-output VG objective.

    Public API is intentionally close to the supplied implementation:
    ``encode`` returns ``(m, a, h)``, ``forward`` returns a reconstruction dict,
    and ``free_energy`` returns loss components.
    """

    def __init__(self, config: VGSAEConfig):
        super().__init__()
        config.validate()
        self.config = config

        d, L, dtype = config.input_dim, config.n_latents, config.torch_dtype
        self.gate_encoder = nn.Linear(d, L, dtype=dtype)
        self.amplitude_encoder = nn.Linear(d, L, dtype=dtype)
        self.decoder = nn.Linear(L, d, bias=False, dtype=dtype)
        self.pre_bias: Optional[nn.Parameter]
        self.pre_bias = nn.Parameter(torch.zeros(d, dtype=dtype)) if config.decoder_bias else None
        mode = self._resolve_beta_mode()
        self.log_beta = (
            nn.Parameter(torch.tensor(math.log(config.beta), dtype=dtype))
            if mode == "learned"
            else None
        )
        self.reset_parameters()

    def _resolve_beta_mode(
        self,
        beta_mode: Literal["profiled", "fixed", "learned"] | None = None,
        trace_beta: bool | None = None,
    ) -> Literal["profiled", "fixed", "learned"]:
        if beta_mode is not None:
            mode = beta_mode
        elif self.config.beta_mode is not None:
            mode = self.config.beta_mode
        else:
            trace = self.config.trace_beta if trace_beta is None else trace_beta
            mode = "profiled" if trace is not False else "fixed"
        if mode not in {"profiled", "fixed", "learned"}:
            raise ValueError("beta_mode must be one of: profiled, fixed, learned.")
        return mode

    def reset_parameters(self) -> None:
        cfg = self.config
        nn.init.normal_(self.decoder.weight, mean=0.0, std=cfg.input_dim ** -0.5)
        if cfg.normalize_decoder:
            self.normalize_decoder_columns()

        if cfg.tie_encoder_init:
            with torch.no_grad():
                self.amplitude_encoder.weight.copy_(self.decoder.weight.t())
        else:
            nn.init.kaiming_uniform_(self.amplitude_encoder.weight, a=5 ** 0.5)

        nn.init.kaiming_uniform_(self.gate_encoder.weight, a=5 ** 0.5)
        nn.init.constant_(self.gate_encoder.bias, float(cfg.gate_bias_init))
        nn.init.constant_(self.amplitude_encoder.bias, float(cfg.amplitude_bias_init))

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (batch, input_dim), got {tuple(x.shape)}.")
        if x.shape[1] != self.config.input_dim:
            raise ValueError(f"Expected input_dim={self.config.input_dim}, got {x.shape[1]}.")

    def _center(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.pre_bias if self.pre_bias is not None else x

    def _uncenter(self, x_centered: torch.Tensor) -> torch.Tensor:
        return x_centered + self.pre_bias if self.pre_bias is not None else x_centered

    def _positive_eps(self, t: torch.Tensor) -> float:
        # Use a dtype-safe lower bound for log/division, while preserving user intent.
        return max(float(self.config.loss_eps), float(torch.finfo(t.dtype).tiny))

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        eps = max(float(self.config.loss_eps), float(torch.finfo(self.decoder.weight.dtype).tiny))
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(eps)
        self.decoder.weight.div_(norms)

    @torch.no_grad()
    def remove_decoder_parallel_grad(self) -> None:
        """Project out gradient components parallel to unit-norm decoder atoms.

        Call this before optimizer.step() when also calling normalize_decoder_columns().
        """
        if self.decoder.weight.grad is None:
            return
        W = self.decoder.weight
        G = self.decoder.weight.grad
        proj = (G * W).sum(dim=0, keepdim=True) * W
        G.sub_(proj)

    def decoder_column_sqnorms(self) -> torch.Tensor:
        return self.decoder.weight.pow(2).sum(dim=0)

    def _encode_centered(
        self, x_centered: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logits = self.gate_encoder(x_centered)
        m = torch.sigmoid(gate_logits)

        amp_pre = self.amplitude_encoder(x_centered)
        a = F.softplus(amp_pre) if self.config.nonnegative_amplitudes else amp_pre
        h = m * a
        return gate_logits, m, a, h

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_input(x)
        _, m, a, h = self._encode_centered(self._center(x))
        return m, a, h

    def decode_centered(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self._uncenter(self.decode_centered(h))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        m, a, h = self.encode(x)
        x_hat = self.decode(h)
        return {"x_hat": x_hat, "m": m, "a": a, "h": h}

    @staticmethod
    def bernoulli_entropy_from_logits(logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Stable Bernoulli entropy H(sigmoid(logits)) per latent.

        H = softplus(logit) - sigmoid(logit) * logit.
        This avoids the float32 bug where ``clamp(max=1-1e-8)`` fails because
        1 - 1e-8 rounds to 1.0.
        """
        return F.softplus(logits) - probs * logits

    def free_energy(
        self,
        x: torch.Tensor,
        *,
        beta: Optional[float] = None,
        lambda_sparsity: Optional[float] = None,
        entropy_weight: Optional[float] = None,
        use_entropy_term: Optional[bool] = None,
        beta_mode: Literal["profiled", "fixed", "learned"] | None = None,
        trace_beta: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the VG-SAE negative variational log posterior.

        ``beta`` overrides the constant precision in fixed mode. ``beta_mode``
        overrides the configured mode; the legacy ``trace_beta`` switch is used
        only when neither explicit mode is set.
        """
        self._check_input(x)
        cfg = self.config
        gamma_value = cfg.lambda_sparsity if lambda_sparsity is None else float(lambda_sparsity)
        entropy_coeff = cfg.entropy_weight if entropy_weight is None else float(entropy_weight)
        entropy_on = cfg.use_entropy_term if use_entropy_term is None else bool(use_entropy_term)
        mode = self._resolve_beta_mode(beta_mode, trace_beta)

        if not math.isfinite(gamma_value):
            raise ValueError("lambda_sparsity/gamma must be finite.")
        if not math.isfinite(entropy_coeff) or entropy_coeff < 0.0:
            raise ValueError("entropy_weight must be non-negative.")
        if not entropy_on:
            entropy_coeff = 0.0

        B, d = x.shape
        x_centered = self._center(x)
        gate_logits, m, a, h = self._encode_centered(x_centered)
        x_hat_centered = self.decode_centered(h)
        x_hat = self._uncenter(x_hat_centered)

        # E_q[1/2 ||x - sum_j s_j a_j D_j||^2]
        recon = 0.5 * (x_centered - x_hat_centered).pow(2).sum(dim=1)

        if cfg.use_variance_term:
            col_sq = self.decoder_column_sqnorms()
            variance = 0.5 * (m * (1.0 - m) * a.pow(2) * col_sq).sum(dim=1)
        else:
            variance = torch.zeros_like(recon)
        energy = recon + variance

        l0_surrogate = m.sum(dim=1)
        prior_normalizer = cfg.n_latents * F.softplus(x.new_tensor(-gamma_value))
        prior = gamma_value * l0_surrogate + prior_normalizer
        entropy = self.bernoulli_entropy_from_logits(gate_logits, m).sum(dim=1)

        if mode == "profiled":
            # Gaussian NLL: beta * E_tot - (M/2) log beta.  beta* = M/(2E_tot).
            # Constants independent of parameters are intentionally omitted.
            E_tot = energy.sum().clamp_min(self._positive_eps(energy))
            M = B * d
            scaled_E = (2.0 * E_tot / float(M)).clamp_min(self._positive_eps(energy))
            loss_total = 0.5 * M * scaled_E.log() + prior.sum() - entropy_coeff * entropy.sum()
            loss = loss_total / B
            beta_eff = torch.as_tensor(0.5 * M, device=x.device, dtype=x.dtype) / E_tot.detach()
        else:
            if mode == "learned":
                if self.log_beta is None:
                    raise ValueError(
                        "learned beta mode requires a model configured with beta_mode='learned'."
                    )
                if beta is not None:
                    raise ValueError(
                        "beta cannot override the trainable precision in learned mode."
                    )
                beta_eff = self.log_beta.exp()
            else:
                beta_value = cfg.beta if beta is None else float(beta)
                if not math.isfinite(beta_value) or beta_value <= 0.0:
                    raise ValueError("beta must be positive and finite.")
                beta_eff = x.new_tensor(beta_value)
            gaussian_normalizer = -0.5 * d * torch.log(beta_eff / (2.0 * math.pi))
            per_sample = beta_eff * energy + gaussian_normalizer + prior - entropy_coeff * entropy
            loss = per_sample.mean()

        return {
            "loss": loss,
            "recon": recon.mean(),
            "variance": variance.mean(),
            "energy": energy.mean(),
            "sparsity": l0_surrogate.mean(),
            "prior": prior.mean(),
            "entropy": entropy.mean(),
            "beta_eff": beta_eff,
            "x_hat": x_hat,
            "m": m,
            "a": a,
            "h": h,
            "gate_logits": gate_logits,
        }

    @torch.no_grad()
    def encode_inference(self, x: torch.Tensor, threshold: Optional[float] = None):
        self._check_input(x)
        tau = self.config.inference_threshold if threshold is None else float(threshold)
        if not (0.0 <= tau <= 1.0):
            raise ValueError("threshold must lie in [0, 1].")
        m, a, _ = self.encode(x)
        s = (m > tau).to(dtype=a.dtype)
        return s, a, s * a


__all__ = [
    "BatchTopKSAE",
    "BatchTopKSAEConfig",
    "GatedSAE",
    "GatedSAEConfig",
    "JumpReLU",
    "JumpReLUSAE",
    "JumpReLUSAEConfig",
    "L1ReLUSAE",
    "L1SAEConfig",
    "Step",
    "TopKSAE",
    "TopKSAEConfig",
    "VGSAEConfig",
    "VariationalGarroteSAE",
]
