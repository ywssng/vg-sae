"""Synthetic sparse-coding data for VG-SAE experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .utils import as_tensor


@dataclass(init=False)
class SyntheticSparseCodingConfig:
    """Configuration for the generic sparse-coding data generator.

    ``n_features`` remains an accepted constructor alias so older callers can
    migrate. Stage 1 invariants are enforced by ``SweepConfig`` rather than by
    this reusable generator.
    """

    input_dim: int
    ground_truth_num_features: int
    n_samples: int
    support_density: float
    coherence: float
    noise_std: float
    frequency_skew: float
    amplitude_scale: float
    seed: int
    dtype: torch.dtype | str

    def __init__(
        self,
        input_dim: int = 16,
        ground_truth_num_features: int | None = None,
        n_samples: int = 1024,
        support_density: float = 0.05,
        coherence: float = 0.0,
        noise_std: float = 0.03,
        frequency_skew: float = 0.0,
        amplitude_scale: float = 1.0,
        seed: int = 0,
        dtype: torch.dtype | str = torch.float32,
        *,
        n_features: int | None = None,
    ) -> None:
        if ground_truth_num_features is None:
            ground_truth_num_features = 64 if n_features is None else n_features
        elif n_features is not None and ground_truth_num_features != n_features:
            raise ValueError(
                "ground_truth_num_features and legacy n_features disagree."
            )
        self.input_dim = input_dim
        self.ground_truth_num_features = ground_truth_num_features
        self.n_samples = n_samples
        self.support_density = support_density
        self.coherence = coherence
        self.noise_std = noise_std
        self.frequency_skew = frequency_skew
        self.amplitude_scale = amplitude_scale
        self.seed = seed
        self.dtype = dtype

    @property
    def n_features(self) -> int:
        """Deprecated ground-truth feature-count alias."""

        return self.ground_truth_num_features

    def validate(self) -> None:
        if self.input_dim <= 0 or self.ground_truth_num_features <= 0:
            raise ValueError("input_dim and ground_truth_num_features must be positive.")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if not 0.0 < self.support_density < 1.0:
            raise ValueError("support_density must be in (0, 1).")
        if self.frequency_skew < 0.0:
            raise ValueError("frequency_skew must be nonnegative.")
        if self.amplitude_scale <= 0.0:
            raise ValueError("amplitude_scale must be positive.")
        if not 0.0 <= self.coherence < 1.0:
            raise ValueError("coherence must satisfy 0 <= coherence < 1.")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be nonnegative.")


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
    config.validate()
    rng = np.random.default_rng(config.seed)
    dtype = _torch_dtype(config.dtype)
    torch_device = torch.device(device)

    dictionary = make_unit_dictionary(
        input_dim=config.input_dim,
        n_features=config.ground_truth_num_features,
        rng=rng,
        coherence=config.coherence,
    )
    probabilities = feature_probabilities(
        n_features=config.ground_truth_num_features,
        support_density=config.support_density,
        frequency_skew=config.frequency_skew,
    )
    shape = (config.n_samples, config.ground_truth_num_features)
    support = rng.binomial(1, probabilities[None, :], size=shape).astype(np.float64)
    amplitudes = rng.exponential(scale=config.amplitude_scale, size=shape)
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
