"""Variational Garrote sparse regression model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _dtype_from_value(value: torch.dtype | str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    normalized = value.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype {value!r}; expected 'float32' or 'float64'.")


@dataclass
class VGConfig:
    """Configuration for the Variational Garrote model."""

    n_features: int = 256
    gamma: float = 1.0
    mask_init: float = 0.999
    weight_init_std: float = 1.0
    loss_eps: float = 1.0e-12
    dtype: torch.dtype | str = torch.float32

    @property
    def torch_dtype(self) -> torch.dtype:
        return _dtype_from_value(self.dtype)


class VariationalGarrote(nn.Module):
    """VG linear model with trainable weights and Bernoulli mask probabilities."""

    def __init__(self, config: VGConfig):
        super().__init__()
        self.config = config
        dtype = config.torch_dtype
        eps = max(float(config.loss_eps), torch.finfo(dtype).eps)

        if config.n_features <= 0:
            raise ValueError("n_features must be positive.")
        if not 0.0 < config.mask_init < 1.0:
            raise ValueError("mask_init must be strictly inside (0, 1) for logit initialization.")

        initial_mask = torch.tensor(config.mask_init, dtype=dtype).clamp(eps, 1.0 - eps)
        initial_logit = torch.log(initial_mask / (1.0 - initial_mask))

        self.weight = nn.Parameter(torch.empty(config.n_features, dtype=dtype))
        self.mask_logits = nn.Parameter(torch.full((config.n_features,), float(initial_logit), dtype=dtype))
        self.reset_parameters()

    @property
    def w(self) -> torch.nn.Parameter:
        """Compatibility alias for implementations that call the weights `w`."""
        return self.weight

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=float(self.config.weight_init_std))

    def mask(self) -> torch.Tensor:
        """Return mean-field Bernoulli probabilities m_i in (0, 1)."""
        return torch.sigmoid(self.mask_logits)

    def get_mask(self) -> torch.Tensor:
        return self.mask()

    def effective_weights(self) -> torch.Tensor:
        return self.mask() * self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (n_samples, n_features), got {tuple(x.shape)}.")
        if x.shape[1] != self.config.n_features:
            raise ValueError(f"Expected {self.config.n_features} features, got {x.shape[1]}.")
        return x @ self.effective_weights()

    def rho_model(self) -> torch.Tensor:
        """Paper density rho_model = mean_i m_i."""
        return self.mask().mean()

    def get_sparsity(self) -> float:
        with torch.no_grad():
            return float(self.rho_model().detach().cpu())

    def binary_mask(self, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            return self.mask() > threshold

    def get_binary_mask(self, threshold: float = 0.5) -> torch.Tensor:
        return self.binary_mask(threshold=threshold)
