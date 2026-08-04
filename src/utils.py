"""Shared utilities for the VG implementation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_string(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    normalized = value.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype {value!r}; expected 'float32' or 'float64'.")


def as_tensor(data: np.ndarray | torch.Tensor, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    return torch.as_tensor(data, dtype=dtype, device=device)


def numpy_to_torch(
    x: np.ndarray,
    y: np.ndarray,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    return as_tensor(x, dtype=dtype, device=device), as_tensor(y, dtype=dtype, device=device)


def rho_model_to_gamma_grid(
    n_gamma: int = 30,
    gamma_min: float = 0.01,
    gamma_max: float = 20.0,
) -> np.ndarray:
    if n_gamma <= 0:
        raise ValueError("n_gamma must be positive.")
    if gamma_min <= 0.0 or gamma_max <= gamma_min:
        raise ValueError("Require 0 < gamma_min < gamma_max.")
    return np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_gamma)


def standardize(
    x_train: np.ndarray,
    x_test: np.ndarray | None = None,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray | None]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    x_train_std = (x_train - mean) / np.maximum(std, eps)
    x_test_std = None if x_test is None else (x_test - mean) / np.maximum(std, eps)
    return x_train_std, x_test_std
