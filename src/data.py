"""Data utilities for Variational Garrote experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import as_tensor


@dataclass
class SyntheticConfig:
    """Synthetic spike-and-slab regression setup from the paper."""

    n_features: int = 256
    n_samples: int = 256
    rho_data: float = 5.0 / 256.0
    snr: float = 3.0
    seed: int = 0
    dtype: torch.dtype | str = torch.float32


@dataclass
class RegressionTensors:
    x: torch.Tensor
    y: torch.Tensor
    teacher_weights: torch.Tensor | None = None
    teacher_mask: torch.Tensor | None = None
    noise_std: torch.Tensor | None = None
    clean_y: torch.Tensor | None = None


class VGRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (M, N), got {tuple(x.shape)}.")
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError(f"Expected y with shape (M,), got {tuple(y.shape)}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def _torch_dtype(value: torch.dtype | str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value == "float32":
        return torch.float32
    if value == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype {value!r}; expected 'float32' or 'float64'.")


def spike_slab_upper_bound(rho_data: float) -> float:
    if not 0.0 < rho_data <= 1.0:
        raise ValueError(f"rho_data must be in (0, 1], got {rho_data}.")
    return float(np.sqrt(12.0 / rho_data - 0.75) - 0.5)


def sample_spike_and_slab(
    n_features: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample teacher weights with an exact finite-N active count."""
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}.")

    n_active = int(round(rho * n_features))
    weights = np.zeros(n_features, dtype=np.float64)
    if n_active == 0:
        return weights

    effective_rho = n_active / n_features
    w_bar = spike_slab_upper_bound(effective_rho)
    active_idx = rng.choice(n_features, size=n_active, replace=False)
    magnitudes = rng.uniform(np.nextafter(1.0, np.inf), w_bar, size=n_active)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_active)
    weights[active_idx] = signs * magnitudes
    return weights


def sample_spike_slab_weights(
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    weights = sample_spike_and_slab(config.n_features, config.rho_data, rng)
    return weights, (weights != 0.0).astype(np.float64)


def generate_synthetic_dataset(
    n_features: int,
    n_samples: int,
    rho_data: float,
    snr: float,
    rng: np.random.Generator,
    teacher_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Generate iid Gaussian features, noisy targets, and hidden teacher weights."""
    if snr <= 0.0:
        raise ValueError("snr must be positive.")
    if teacher_weights is None:
        teacher_weights = sample_spike_and_slab(n_features=n_features, rho=rho_data, rng=rng)
    else:
        teacher_weights = np.asarray(teacher_weights, dtype=np.float64)
        if teacher_weights.shape != (n_features,):
            raise ValueError(f"teacher_weights must have shape ({n_features},).")

    x = rng.standard_normal(size=(n_samples, n_features))
    clean_y = x @ teacher_weights
    signal_power = float(np.mean(clean_y ** 2))
    noise_std = float(np.sqrt(signal_power / snr)) if signal_power > 0.0 else 0.0
    noise = rng.normal(0.0, noise_std, size=n_samples)
    y = clean_y + noise
    return x, y, teacher_weights, noise_std, clean_y


def make_synthetic_regression(
    config: SyntheticConfig,
    device: torch.device | str = "cpu",
    teacher_weights: torch.Tensor | np.ndarray | None = None,
    teacher_mask: torch.Tensor | np.ndarray | None = None,
) -> RegressionTensors:
    rng = np.random.default_rng(config.seed)
    dtype = _torch_dtype(config.dtype)
    torch_device = torch.device(device)

    teacher_np = None if teacher_weights is None else np.asarray(
        teacher_weights.detach().cpu() if isinstance(teacher_weights, torch.Tensor) else teacher_weights,
        dtype=np.float64,
    )
    x_np, y_np, weights_np, noise_std, clean_y_np = generate_synthetic_dataset(
        n_features=config.n_features,
        n_samples=config.n_samples,
        rho_data=config.rho_data,
        snr=config.snr,
        rng=rng,
        teacher_weights=teacher_np,
    )
    mask_np = (weights_np != 0.0).astype(np.float64) if teacher_mask is None else np.asarray(
        teacher_mask.detach().cpu() if isinstance(teacher_mask, torch.Tensor) else teacher_mask,
        dtype=np.float64,
    )

    return RegressionTensors(
        x=as_tensor(x_np, dtype=dtype, device=torch_device),
        y=as_tensor(y_np, dtype=dtype, device=torch_device),
        teacher_weights=as_tensor(weights_np, dtype=dtype, device=torch_device),
        teacher_mask=as_tensor(mask_np, dtype=dtype, device=torch_device),
        noise_std=as_tensor(np.asarray(noise_std), dtype=dtype, device=torch_device),
        clean_y=as_tensor(clean_y_np, dtype=dtype, device=torch_device),
    )


def make_synthetic_train_test(
    train_config: SyntheticConfig,
    n_test: int,
    test_seed_offset: int = 10000,
    device: torch.device | str = "cpu",
) -> tuple[RegressionTensors, RegressionTensors]:
    train = make_synthetic_regression(train_config, device=device)
    test_config = SyntheticConfig(
        n_features=train_config.n_features,
        n_samples=n_test,
        rho_data=train_config.rho_data,
        snr=train_config.snr,
        seed=train_config.seed + test_seed_offset,
        dtype=train_config.dtype,
    )
    test = make_synthetic_regression(
        test_config,
        device=device,
        teacher_weights=train.teacher_weights,
        teacher_mask=train.teacher_mask,
    )
    return train, test


def standardize_columns(values: np.ndarray, eps: float = 1.0e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    standardized = (values - mean) / np.maximum(std, eps)
    return standardized, mean, std


def positive_log_with_cutoff(values: np.ndarray, cutoff: float = -3.0) -> np.ndarray:
    clipped = np.maximum(values, np.exp(cutoff))
    return np.log(clipped)


def signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def load_tabular_csv(
    path: str | Path,
    target_column: str,
    feature_columns: Sequence[str] | None = None,
    transform: str = "none",
    standardize: bool = True,
    positive_log_cutoff: float = -3.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> RegressionTensors:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for load_tabular_csv.") from exc

    frame = pd.read_csv(path)
    if feature_columns is None:
        feature_columns = [column for column in frame.columns if column != target_column]

    x_np = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)
    y_np = frame.loc[:, target_column].to_numpy(dtype=np.float64)

    if transform == "positive_log":
        x_np = positive_log_with_cutoff(x_np, cutoff=positive_log_cutoff)
    elif transform == "signed_log1p":
        x_np = signed_log1p(x_np)
    elif transform != "none":
        raise ValueError("transform must be one of: 'none', 'positive_log', 'signed_log1p'.")

    if standardize:
        x_np, _, _ = standardize_columns(x_np)
        y_np, _, _ = standardize_columns(y_np[:, None])
        y_np = y_np[:, 0]

    torch_device = torch.device(device)
    return RegressionTensors(
        x=as_tensor(x_np, dtype=dtype, device=torch_device),
        y=as_tensor(y_np, dtype=dtype, device=torch_device),
    )


def random_train_test_split(
    tensors: RegressionTensors,
    train_fraction: float,
    seed: int,
) -> tuple[RegressionTensors, RegressionTensors]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")
    generator = torch.Generator(device=tensors.x.device)
    generator.manual_seed(seed)
    n_samples = tensors.x.shape[0]
    permutation = torch.randperm(n_samples, generator=generator, device=tensors.x.device)
    n_train = int(round(train_fraction * n_samples))
    train_idx = permutation[:n_train]
    test_idx = permutation[n_train:]
    train = RegressionTensors(x=tensors.x[train_idx], y=tensors.y[train_idx])
    test = RegressionTensors(x=tensors.x[test_idx], y=tensors.y[test_idx])
    return train, test
