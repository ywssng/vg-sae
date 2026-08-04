"""Local L1 SAE plus exact aliases to official SAELens training SAEs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.gated_sae import GatedTrainingSAE, GatedTrainingSAEConfig
from sae_lens.saes.jumprelu_sae import (
    JumpReLU,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    Step,
)
from sae_lens.saes.sae import SAE, TrainingSAE
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

from .model import _dtype_from_value

# Public baseline names are identities, not wrappers or reimplementations.
TopKSAE = TopKTrainingSAE
TopKSAEConfig = TopKTrainingSAEConfig
BatchTopKSAE = BatchTopKTrainingSAE
BatchTopKSAEConfig = BatchTopKTrainingSAEConfig
JumpReLUSAE = JumpReLUTrainingSAE
JumpReLUSAEConfig = JumpReLUTrainingSAEConfig
GatedSAE = GatedTrainingSAE
GatedSAEConfig = GatedTrainingSAEConfig


def to_inference_sae(
    model: TrainingSAE[Any], *, fold_decoder_norm: bool = False
) -> SAE[Any]:
    """Convert with official config/state hooks, then optionally fold decoder norms."""

    state_dict = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.process_state_dict_for_saving_inference(state_dict)
    inference = SAE.from_dict(model.cfg.get_inference_sae_cfg_dict())
    inference.load_state_dict(state_dict)
    inference.eval()
    if fold_decoder_norm:
        inference.fold_W_dec_norm()
    return inference


@dataclass
class SAEConfig:
    """Configuration shared only by project-local SAE implementations."""

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
        weight, grad = self.decoder.weight, self.decoder.weight.grad
        if grad is not None:
            grad.sub_((grad * weight).sum(dim=0, keepdim=True) * weight)


class CenteredLinearSAE(nn.Module, UnitNormDecoderMixin):
    """Linear base retained for the project-local L1/ReLU baseline."""

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
            self.encoder.weight.copy_(self.decoder.weight.T)
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


__all__ = [
    "BatchTopKSAE",
    "BatchTopKSAEConfig",
    "BatchTopKTrainingSAE",
    "BatchTopKTrainingSAEConfig",
    "CenteredLinearSAE",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedTrainingSAE",
    "GatedTrainingSAEConfig",
    "JumpReLU",
    "JumpReLUSAE",
    "JumpReLUSAEConfig",
    "JumpReLUTrainingSAE",
    "JumpReLUTrainingSAEConfig",
    "L1ReLUSAE",
    "L1SAEConfig",
    "SAEConfig",
    "Step",
    "TopKSAE",
    "TopKSAEConfig",
    "TopKTrainingSAE",
    "TopKTrainingSAEConfig",
    "TrainingSAE",
    "UnitNormDecoderMixin",
    "to_inference_sae",
]
