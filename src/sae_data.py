"""Synthetic sparse-coding data for VG-SAE experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .utils import as_tensor


@dataclass
class SyntheticSparseCodingConfig:
    input_dim: int = 16
    n_features: int = 64
    n_samples: int = 1024
    support_density: float = 0.05
    coherence: float = 0.0
    noise_std: float = 0.03
    frequency_skew: float = 0.0
    amplitude_scale: float = 1.0
    seed: int = 0
    dtype: torch.dtype | str = torch.float32


@dataclass
class SparseCodingTensors:
    x: torch.Tensor
    z: torch.Tensor
    support: torch.Tensor
    dictionary: torch.Tensor
    clean_x: torch.Tensor
    feature_probabilities: torch.Tensor


def _torch_dtype(value: torch.dtype | str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value == "float32":
        return torch.float32
    if value == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype {value!r}; expected 'float32' or 'float64'.")


def make_unit_dictionary(
    input_dim: int,
    n_features: int,
    rng: np.random.Generator,
    coherence: float = 0.0,
) -> np.ndarray:
    """Create a column-normalized dictionary with tunable common-mode coherence."""
    if input_dim <= 0 or n_features <= 0:
        raise ValueError("input_dim and n_features must be positive.")
    if not 0.0 <= coherence < 1.0:
        raise ValueError("coherence must satisfy 0 <= coherence < 1.")

    columns = rng.standard_normal(size=(input_dim, n_features))
    if coherence > 0.0:
        common = rng.standard_normal(size=(input_dim, 1))
        common /= np.linalg.norm(common, axis=0, keepdims=True).clip(1.0e-12)
        columns = np.sqrt(coherence) * common + np.sqrt(1.0 - coherence) * columns
    columns /= np.linalg.norm(columns, axis=0, keepdims=True).clip(1.0e-12)
    return columns


def feature_probabilities(
    n_features: int,
    support_density: float,
    frequency_skew: float = 0.0,
) -> np.ndarray:
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if not 0.0 < support_density < 1.0:
        raise ValueError("support_density must be in (0, 1).")
    ranks = np.arange(1, n_features + 1, dtype=np.float64)
    weights = ranks ** (-float(frequency_skew))
    weights /= weights.mean()
    probabilities = support_density * weights
    return np.clip(probabilities, 0.0, 0.95)


def make_synthetic_sparse_coding(
    config: SyntheticSparseCodingConfig,
    device: torch.device | str = "cpu",
) -> SparseCodingTensors:
    rng = np.random.default_rng(config.seed)
    dtype = _torch_dtype(config.dtype)
    torch_device = torch.device(device)

    dictionary = make_unit_dictionary(
        input_dim=config.input_dim,
        n_features=config.n_features,
        rng=rng,
        coherence=config.coherence,
    )
    probabilities = feature_probabilities(
        n_features=config.n_features,
        support_density=config.support_density,
        frequency_skew=config.frequency_skew,
    )
    support = rng.binomial(1, probabilities[None, :], size=(config.n_samples, config.n_features)).astype(np.float64)
    amplitudes = rng.exponential(scale=config.amplitude_scale, size=(config.n_samples, config.n_features))
    z = support * amplitudes
    clean_x = z @ dictionary.T
    noise = rng.normal(0.0, config.noise_std, size=clean_x.shape)
    x = clean_x + noise

    return SparseCodingTensors(
        x=as_tensor(x, dtype=dtype, device=torch_device),
        z=as_tensor(z, dtype=dtype, device=torch_device),
        support=as_tensor(support, dtype=dtype, device=torch_device),
        dictionary=as_tensor(dictionary, dtype=dtype, device=torch_device),
        clean_x=as_tensor(clean_x, dtype=dtype, device=torch_device),
        feature_probabilities=as_tensor(probabilities, dtype=dtype, device=torch_device),
    )


def center_and_normalize(
    x: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    scale = centered.pow(2).mean().sqrt().clamp_min(eps)
    return centered / scale, mean, scale
