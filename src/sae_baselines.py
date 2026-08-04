"""Compact SAELens-aligned sparse-autoencoder baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _dtype_from_value


@dataclass
class SAEConfig:
    input_dim: int
    n_latents: int
    decoder_bias: bool = True
    dtype: torch.dtype | str = torch.float32

    @property
    def torch_dtype(self) -> torch.dtype:
        return _dtype_from_value(self.dtype)

    def validate(self) -> None:
        if self.input_dim <= 0 or self.n_latents <= 0:
            raise ValueError("input_dim and n_latents must be positive.")


@dataclass
class L1SAEConfig(SAEConfig):
    l1_coefficient: float = 1.0e-3


@dataclass
class TopKSAEConfig(SAEConfig):
    k: int = 4
    aux_loss_coefficient: float = 1.0
    rescale_acts_by_decoder_norm: bool = True


@dataclass
class BatchTopKSAEConfig(TopKSAEConfig):
    k: float = 4.0  # type: ignore[assignment]
    topk_threshold_lr: float = 0.01


@dataclass
class JumpReLUSAEConfig(SAEConfig):
    jumprelu_init_threshold: float = 0.01
    jumprelu_bandwidth: float = 0.05
    jumprelu_sparsity_loss_mode: Literal["step", "tanh"] = "step"
    l0_coefficient: float = 1.0
    pre_act_loss_coefficient: float | None = None
    jumprelu_tanh_scale: float = 4.0


@dataclass
class GatedSAEConfig(SAEConfig):
    l1_coefficient: float = 1.0e-3
    gate_bias_init: float = 0.0


class UnitNormDecoderMixin:
    """Local convention: decoder atoms are columns of ``decoder.weight``."""

    decoder: nn.Linear

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        weight = self.decoder.weight
        weight.div_(weight.norm(dim=0, keepdim=True).clamp_min(1.0e-12))

    def decoder_column_norms(self) -> torch.Tensor:
        return self.decoder.weight.norm(dim=0)

    @torch.no_grad()
    def remove_decoder_parallel_grad(self) -> None:
        if self.decoder.weight.grad is None:
            return
        weight, grad = self.decoder.weight, self.decoder.weight.grad
        grad.sub_((grad * weight).sum(dim=0, keepdim=True) * weight)


class CenteredLinearSAE(nn.Module, UnitNormDecoderMixin):
    """Shared SAELens convention: subtract ``b_dec`` before encoding."""

    def __init__(self, config: SAEConfig):
        super().__init__()
        config.validate()
        self.config = config
        dtype = config.torch_dtype
        self.encoder = nn.Linear(config.input_dim, config.n_latents, dtype=dtype)
        self.decoder = nn.Linear(
            config.n_latents, config.input_dim, bias=config.decoder_bias, dtype=dtype
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        self.normalize_decoder_columns()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.t())
        nn.init.zeros_(self.encoder.bias)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

    def _center(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.decoder.bias if self.decoder.bias is not None else x

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._center(x))

    def decode(self, h: torch.Tensor, *, bias: bool = True) -> torch.Tensor:
        return F.linear(h, self.decoder.weight, self.decoder.bias if bias else None)


class L1ReLUSAE(CenteredLinearSAE):
    config: L1SAEConfig

    def __init__(self, config: L1SAEConfig):
        if not math.isfinite(config.l1_coefficient) or config.l1_coefficient < 0:
            raise ValueError("l1_coefficient must be finite and non-negative.")
        super().__init__(config)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.pre_activations(x))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_pre = self.pre_activations(x)
        h = F.relu(hidden_pre)
        return {"x_hat": self.decode(h), "h": h, "hidden_pre": hidden_pre}


class TopKSAE(CenteredLinearSAE):
    """Per-sample TopK over raw preactivations, followed by ReLU."""

    config: TopKSAEConfig

    def __init__(self, config: TopKSAEConfig):
        if not 0 < config.k <= config.n_latents:
            raise ValueError("k must satisfy 0 < k <= n_latents.")
        if not isinstance(config, BatchTopKSAEConfig) and not float(config.k).is_integer():
            raise ValueError("TopK k must be an integer.")
        if not math.isfinite(config.aux_loss_coefficient) or config.aux_loss_coefficient < 0:
            raise ValueError("aux_loss_coefficient must be finite and non-negative.")
        super().__init__(config)

    def _selection_pre(self, hidden_pre: torch.Tensor) -> torch.Tensor:
        if not self.config.rescale_acts_by_decoder_norm:
            return hidden_pre
        return hidden_pre * self.decoder_column_norms()

    def _decode_acts(self, h: torch.Tensor) -> torch.Tensor:
        if not self.config.rescale_acts_by_decoder_norm:
            return h
        return h / self.decoder_column_norms().clamp_min(1.0e-12)

    def _encode_details(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_pre = self._selection_pre(self.pre_activations(x))
        values, indices = hidden_pre.topk(int(self.config.k), dim=-1)
        h = torch.zeros_like(hidden_pre).scatter(-1, indices, values.relu())
        selected = torch.zeros_like(h).scatter(-1, indices, 1.0)
        return h, h.gt(0).to(h), hidden_pre, selected

    def encode_with_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, mask, _, _ = self._encode_details(x)
        return h, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_details(x)[0]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, mask, hidden_pre, selected = self._encode_details(x)
        return {
            "x_hat": self.decode(self._decode_acts(h)),
            "h": h,
            "mask": mask,
            "selection_mask": selected,
            "hidden_pre": hidden_pre,
        }

    def auxiliary_loss(
        self,
        x: torch.Tensor,
        output: dict[str, torch.Tensor],
        dead_feature_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if dead_feature_mask is None:
            return x.new_zeros(())
        dead = dead_feature_mask.to(device=x.device, dtype=torch.bool)
        num_dead, k_aux = int(dead.sum()), x.shape[-1] // 2
        if num_dead == 0 or k_aux == 0:
            return x.new_zeros(())
        scale, k_aux = min(num_dead / k_aux, 1.0), min(k_aux, num_dead)
        candidates = output["hidden_pre"].masked_fill(~dead.unsqueeze(0), -torch.inf)
        topk = candidates.topk(k_aux, dim=-1, sorted=False)
        aux = torch.zeros_like(candidates).scatter(-1, topk.indices, topk.values)
        aux_hat = self.decode(self._decode_acts(aux), bias=False)
        residual = (x - output["x_hat"]).detach()
        sse = (aux_hat - residual).pow(2).sum(dim=-1).mean()
        return float(self.config.aux_loss_coefficient) * scale * sse


class BatchTopKSAE(TopKSAE):
    """Global BatchTopK for training and learned-threshold inference."""

    config: BatchTopKSAEConfig

    def __init__(self, config: BatchTopKSAEConfig):
        if not 0 < config.k <= config.n_latents:
            raise ValueError("k must satisfy 0 < k <= n_latents.")
        if not 0.0 <= config.topk_threshold_lr <= 1.0:
            raise ValueError("topk_threshold_lr must lie in [0, 1].")
        super().__init__(config)
        self.register_buffer("topk_threshold", torch.tensor(0.0, dtype=torch.double))

    def _encode_details(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_pre = self._selection_pre(self.pre_activations(x))
        acts = hidden_pre.relu()
        if self.training:
            flat = acts.flatten()
            count = int(float(self.config.k) * acts.shape[:-1].numel())
            count = min(count, flat.numel())
            topk = flat.topk(count, sorted=False)
            h = torch.zeros_like(flat).scatter(-1, topk.indices, topk.values).view_as(acts)
            selected = torch.zeros_like(flat).scatter(-1, topk.indices, 1.0).view_as(acts)
        else:
            threshold = self.topk_threshold.to(device=acts.device, dtype=acts.dtype)
            h = acts * (acts > threshold)
            selected = (acts > threshold).to(acts)
        return h, h.gt(0).to(h), hidden_pre, selected

    @torch.no_grad()
    def update_topk_threshold(self, feature_acts: torch.Tensor) -> None:
        positive = feature_acts[feature_acts > 0]
        if positive.numel() == 0:
            return
        lr = float(self.config.topk_threshold_lr)
        minimum = positive.min().to(self.topk_threshold)
        self.topk_threshold.mul_(1.0 - lr).add_(minimum, alpha=lr)


def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class Step(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float
    ):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).to(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[no-untyped-def]
        x, threshold = ctx.saved_tensors
        grad_threshold = (
            -rectangle((x - threshold) / ctx.bandwidth) * grad_output / ctx.bandwidth
        ).sum(dim=0)
        return None, grad_threshold, None


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float
    ):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[no-untyped-def]
        x, threshold = ctx.saved_tensors
        x_grad = (x > threshold) * grad_output
        threshold_grad = (
            -threshold
            * rectangle((x - threshold) / ctx.bandwidth)
            * grad_output
            / ctx.bandwidth
        ).sum(dim=0)
        return x_grad, threshold_grad, None


class JumpReLUSAE(CenteredLinearSAE):
    config: JumpReLUSAEConfig

    def __init__(self, config: JumpReLUSAEConfig):
        if not all(
            math.isfinite(value) and value > 0
            for value in (
                config.jumprelu_init_threshold,
                config.jumprelu_bandwidth,
                config.jumprelu_tanh_scale,
            )
        ):
            raise ValueError(
                "JumpReLU threshold, bandwidth, and tanh scale must be positive and finite."
            )
        if config.jumprelu_sparsity_loss_mode not in {"step", "tanh"}:
            raise ValueError("jumprelu_sparsity_loss_mode must be 'step' or 'tanh'.")
        if not math.isfinite(config.l0_coefficient) or config.l0_coefficient < 0:
            raise ValueError("l0_coefficient must be finite and non-negative.")
        if config.pre_act_loss_coefficient is not None and (
            not math.isfinite(config.pre_act_loss_coefficient)
            or config.pre_act_loss_coefficient < 0
        ):
            raise ValueError("pre_act_loss_coefficient must be finite and non-negative.")
        super().__init__(config)
        self.log_threshold = nn.Parameter(
            torch.full(
                (config.n_latents,),
                math.log(config.jumprelu_init_threshold),
                dtype=config.torch_dtype,
            )
        )

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return JumpReLU.apply(
            self.pre_activations(x), self.threshold, self.config.jumprelu_bandwidth
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_pre = self.pre_activations(x)
        h = JumpReLU.apply(hidden_pre, self.threshold, self.config.jumprelu_bandwidth)
        return {
            "x_hat": self.decode(h),
            "h": h,
            "mask": h.gt(0).to(h),
            "hidden_pre": hidden_pre,
            "threshold": self.threshold,
        }


class GatedSAE(CenteredLinearSAE):
    """Hard gate and magnitude paths sharing the SAELens encoder weights."""

    config: GatedSAEConfig

    def __init__(self, config: GatedSAEConfig):
        if not math.isfinite(config.l1_coefficient) or config.l1_coefficient < 0:
            raise ValueError("l1_coefficient must be finite and non-negative.")
        if not math.isfinite(config.gate_bias_init):
            raise ValueError("gate_bias_init must be finite.")
        super().__init__(config)
        self.r_mag = nn.Parameter(torch.zeros(config.n_latents, dtype=config.torch_dtype))
        self.b_mag = nn.Parameter(torch.zeros(config.n_latents, dtype=config.torch_dtype))
        nn.init.constant_(self.encoder.bias, float(config.gate_bias_init))

    @property
    def gate_encoder(self) -> nn.Linear:
        """Compatibility alias for the shared encoder/gating path."""
        return self.encoder

    def encode_full(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centered = self._center(x)
        pi_gate = self.encoder(centered)
        gate = (pi_gate > 0).to(x)
        magnitude_pre = F.linear(
            centered, self.encoder.weight * self.r_mag.exp().unsqueeze(1), self.b_mag
        )
        h = gate * F.relu(magnitude_pre)
        return gate, h, pi_gate, magnitude_pre

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, h, _, _ = self.encode_full(x)
        return gate, h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        gate, h, pi_gate, magnitude_pre = self.encode_full(x)
        return {
            "x_hat": self.decode(h),
            "h": h,
            "gate": gate,
            "mask": h.gt(0).to(h),
            "pi_gate": pi_gate,
            "hidden_pre": magnitude_pre,
        }


__all__ = [
    "BatchTopKSAE",
    "BatchTopKSAEConfig",
    "CenteredLinearSAE",
    "GatedSAE",
    "GatedSAEConfig",
    "JumpReLU",
    "JumpReLUSAE",
    "JumpReLUSAEConfig",
    "L1ReLUSAE",
    "L1SAEConfig",
    "SAEConfig",
    "Step",
    "TopKSAE",
    "TopKSAEConfig",
    "UnitNormDecoderMixin",
]
