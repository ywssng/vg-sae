"""Sparse autoencoder models for VG-SAE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _dtype_from_value


@dataclass
class SAEConfig:
    """Shared configuration for small sparse-autoencoder experiments."""

    input_dim: int
    n_latents: int
    decoder_bias: bool = True
    dtype: torch.dtype | str = torch.float32

    @property
    def torch_dtype(self) -> torch.dtype:
        return _dtype_from_value(self.dtype)


@dataclass
class VGSAEConfig:
    """Configuration for the amortized vector-output VG-SAE."""

    input_dim: int
    n_latents: int

    # VG objective. ``lambda_sparsity`` is the positive gamma penalty.
    beta: float = 1.0
    lambda_sparsity: float = 1.0
    use_variance_term: bool = True
    use_entropy_term: bool = True
    entropy_weight: float = 1.0
    trace_beta: bool = True

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
        if self.beta <= 0.0:
            raise ValueError("beta must be positive.")
        if self.lambda_sparsity < 0.0:
            raise ValueError("lambda_sparsity/gamma must be non-negative for the Soh prior exp(-gamma s).")
        if self.entropy_weight < 0.0:
            raise ValueError("entropy_weight must be non-negative.")
        if self.loss_eps <= 0.0:
            raise ValueError("loss_eps must be positive.")
        if not (0.0 <= self.inference_threshold <= 1.0):
            raise ValueError("inference_threshold must lie in [0, 1].")


@dataclass
class L1SAEConfig(SAEConfig):
    l1_coefficient: float = 1.0e-3


@dataclass
class TopKSAEConfig(SAEConfig):
    k: int = 4


@dataclass
class GatedSAEConfig(SAEConfig):
    l1_coefficient: float = 1.0e-3
    gate_bias_init: float = -2.0


class UnitNormDecoderMixin:
    """Mixin for models with decoder weight shaped (input_dim, n_latents)."""

    decoder: nn.Linear

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        weight = self.decoder.weight.data
        weight.div_(weight.norm(dim=0, keepdim=True).clamp_min(1.0e-12))

    def decoder_column_norms(self) -> torch.Tensor:
        return self.decoder.weight.norm(dim=0)


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
        # config.validate()
        self.config = config

        d, L, dtype = config.input_dim, config.n_latents, config.torch_dtype
        self.gate_encoder = nn.Linear(d, L, dtype=dtype)
        self.amplitude_encoder = nn.Linear(d, L, dtype=dtype)
        self.decoder = nn.Linear(L, d, bias=False, dtype=dtype)
        self.pre_bias: Optional[nn.Parameter]
        self.pre_bias = nn.Parameter(torch.zeros(d, dtype=dtype)) if config.decoder_bias else None
        self.reset_parameters()

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
        trace_beta: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the VG-SAE negative variational log posterior.

        Optional keyword arguments override config values for schedules without
        mutating the dataclass.
        """
        self._check_input(x)
        cfg = self.config
        beta_value = cfg.beta if beta is None else float(beta)
        gamma_value = cfg.lambda_sparsity if lambda_sparsity is None else float(lambda_sparsity)
        entropy_coeff = cfg.entropy_weight if entropy_weight is None else float(entropy_weight)
        entropy_on = cfg.use_entropy_term if use_entropy_term is None else bool(use_entropy_term)
        trace = cfg.trace_beta if trace_beta is None else bool(trace_beta)

        if beta_value <= 0.0:
            raise ValueError("beta must be positive.")
        # if gamma_value < 0.0:
            # raise ValueError("lambda_sparsity/gamma must be non-negative.")
        if entropy_coeff < 0.0:
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
        prior = gamma_value * l0_surrogate
        entropy = self.bernoulli_entropy_from_logits(gate_logits, m).sum(dim=1)

        if trace:
            # Gaussian NLL: beta * E_tot - (M/2) log beta.  beta* = M/(2E_tot).
            # Constants independent of parameters are intentionally omitted.
            E_tot = energy.sum().clamp_min(self._positive_eps(energy))
            M = B * d
            scaled_E = (2.0 * E_tot / float(M)).clamp_min(self._positive_eps(energy))
            loss_total = 0.5 * M * scaled_E.log() + prior.sum() - entropy_coeff * entropy.sum()
            loss = loss_total / B
            beta_eff = torch.as_tensor(0.5 * M, device=x.device, dtype=x.dtype) / E_tot.detach()
        else:
            per_sample = beta_value * energy + prior - entropy_coeff * entropy
            loss = per_sample.mean()
            beta_eff = torch.as_tensor(beta_value, device=x.device, dtype=x.dtype)

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


class L1ReLUSAE(nn.Module, UnitNormDecoderMixin):
    """Standard ReLU SAE with L1 activation penalty."""

    def __init__(self, config: L1SAEConfig):
        super().__init__()
        if config.input_dim <= 0 or config.n_latents <= 0:
            raise ValueError("input_dim and n_latents must be positive.")
        self.config = config
        dtype = config.torch_dtype
        self.encoder = nn.Linear(config.input_dim, config.n_latents, dtype=dtype)
        self.decoder = nn.Linear(config.n_latents, config.input_dim, bias=config.decoder_bias, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=self.config.input_dim ** -0.5)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder_columns()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {"x_hat": self.decoder(h), "h": h}


class TopKSAE(nn.Module, UnitNormDecoderMixin):
    """TopK SAE with exactly k selected latents per sample."""

    def __init__(self, config: TopKSAEConfig):
        super().__init__()
        if config.input_dim <= 0 or config.n_latents <= 0:
            raise ValueError("input_dim and n_latents must be positive.")
        if not 0 < config.k <= config.n_latents:
            raise ValueError("k must satisfy 0 < k <= n_latents.")
        self.config = config
        dtype = config.torch_dtype
        self.encoder = nn.Linear(config.input_dim, config.n_latents, dtype=dtype)
        self.decoder = nn.Linear(config.n_latents, config.input_dim, bias=config.decoder_bias, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=self.config.input_dim ** -0.5)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder_columns()

    def encode_with_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = F.relu(self.encoder(x))
        _, indices = torch.topk(acts, k=self.config.k, dim=1)
        mask = torch.zeros_like(acts)
        mask.scatter_(1, indices, 1.0)
        return mask * acts, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encode_with_mask(x)
        return h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, mask = self.encode_with_mask(x)
        return {"x_hat": self.decoder(h), "h": h, "mask": mask}


class GatedSAE(nn.Module, UnitNormDecoderMixin):
    """Deterministic sigmoid-gated SAE baseline."""

    def __init__(self, config: GatedSAEConfig):
        super().__init__()
        if config.input_dim <= 0 or config.n_latents <= 0:
            raise ValueError("input_dim and n_latents must be positive.")
        self.config = config
        dtype = config.torch_dtype
        self.gate_encoder = nn.Linear(config.input_dim, config.n_latents, dtype=dtype)
        self.magnitude_encoder = nn.Linear(config.input_dim, config.n_latents, dtype=dtype)
        self.decoder = nn.Linear(config.n_latents, config.input_dim, bias=config.decoder_bias, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.gate_encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.magnitude_encoder.weight, a=5**0.5)
        nn.init.constant_(self.gate_encoder.bias, float(self.config.gate_bias_init))
        nn.init.zeros_(self.magnitude_encoder.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=self.config.input_dim ** -0.5)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder_columns()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate_encoder(x))
        magnitude = F.relu(self.magnitude_encoder(x))
        return gate, gate * magnitude

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        gate, h = self.encode(x)
        return {"x_hat": self.decoder(h), "gate": gate, "h": h}
