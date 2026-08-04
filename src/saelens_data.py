"""Small, deterministic data boundary for SAELens-compatible SAE training.

SAELens trainers consume an ``Iterator[Tensor]`` whose batches have shape
``(batch, d_in)``.  This module keeps that contract independent of SAELens so
the same activations and batch order can be shared by official baselines and
the custom VG-SAE.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal

import torch

Normalization = Literal["none", "expected_average_only_in"]


def as_activation_tensor(source: Any) -> torch.Tensor:
    """Extract and flatten activations to ``(samples, d_in)``.

    Accepted inputs are a tensor, an object with an ``x`` tensor (including
    :class:`src.sae_data.SparseCodingTensors`), or a mapping containing
    ``"activations"``.
    """
    if isinstance(source, torch.Tensor):
        activations = source
    elif isinstance(source, Mapping):
        activations = source.get("activations")
    else:
        activations = getattr(source, "x", None)
    if not isinstance(activations, torch.Tensor):
        raise TypeError(
            "Expected a tensor, an object with tensor .x, or an 'activations' mapping."
        )
    if activations.ndim < 2 or activations.shape[-1] == 0:
        raise ValueError("Activations must have shape (..., d_in) with d_in > 0.")
    return activations.reshape(-1, activations.shape[-1])


@dataclass(frozen=True)
class ActivationScale:
    """Scalar normalization matching SAELens ``expected_average_only_in``."""

    factor: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.factor) or self.factor <= 0.0:
            raise ValueError("factor must be positive and finite.")

    @classmethod
    def fit(cls, activations: Any, eps: float = 1.0e-12) -> ActivationScale:
        x = as_activation_tensor(activations)
        if len(x) == 0:
            raise ValueError("Cannot fit activation scale on an empty tensor.")
        mean_norm = x.float().norm(dim=-1).mean().item()
        if mean_norm <= eps:
            raise ValueError("Cannot normalize activations with near-zero mean norm.")
        return cls(factor=x.shape[-1] ** 0.5 / mean_norm)

    def scale(self, activations: torch.Tensor) -> torch.Tensor:
        return activations * self.factor

    def unscale(self, activations: torch.Tensor) -> torch.Tensor:
        return activations / self.factor

    __call__ = scale


@dataclass(frozen=True)
class SplitIndices:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor


@dataclass(frozen=True)
class ActivationSplits:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    indices: SplitIndices
    scale: ActivationScale


def make_split_indices(
    n_samples: int,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
    group_ids: torch.Tensor | None = None,
) -> SplitIndices:
    """Create one seeded sample or group permutation for aligned splits."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if validation_fraction < 0.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "validation_fraction must be >= 0 and leave a non-empty test fraction."
        )
    if group_ids is not None:
        groups = torch.as_tensor(group_ids).reshape(-1).cpu()
        if len(groups) != n_samples:
            raise ValueError("group_ids must contain one value per activation sample.")
        _, sample_groups = torch.unique(groups, sorted=True, return_inverse=True)
        n_units = int(sample_groups.max()) + 1
    else:
        sample_groups = None
        n_units = n_samples

    n_train = int(n_units * train_fraction)
    n_validation = int(n_units * validation_fraction)
    if n_train == 0 or n_train + n_validation >= n_units:
        raise ValueError("Split fractions produce an empty train or test split.")
    order = torch.randperm(n_units, generator=torch.Generator().manual_seed(seed))
    if sample_groups is not None:
        partitions = (
            order[:n_train],
            order[n_train : n_train + n_validation],
            order[n_train + n_validation :],
        )
        train, validation, test = (
            torch.isin(sample_groups, partition).nonzero().flatten()
            for partition in partitions
        )
        return SplitIndices(train=train, validation=validation, test=test)
    return SplitIndices(
        train=order[:n_train],
        validation=order[n_train : n_train + n_validation],
        test=order[n_train + n_validation :],
    )


def split_activations(
    activations: Any,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
    normalization: Normalization = "expected_average_only_in",
    group_ids: torch.Tensor | None = None,
) -> ActivationSplits:
    """Split activations and fit normalization on the train split only.

    ``group_ids`` keeps all positions from a cached sequence in one split.  If
    omitted, an input object's or mapping's ``group_ids`` is used when present.
    This is a fallback for a single cache; separate source-dataset splits are
    stronger because packed rows need not preserve original document identity.
    """
    x = as_activation_tensor(activations)
    if group_ids is None:
        group_ids = (
            activations.get("group_ids")
            if isinstance(activations, Mapping)
            else getattr(activations, "group_ids", None)
        )
    indices = make_split_indices(
        len(x),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
        group_ids=group_ids,
    )
    if normalization not in ("none", "expected_average_only_in"):
        raise ValueError(f"Unsupported normalization: {normalization!r}.")
    scale = (
        ActivationScale.fit(x[indices.train])
        if normalization == "expected_average_only_in"
        else ActivationScale()
    )
    return ActivationSplits(
        train=scale(x[indices.train]),
        validation=scale(x[indices.validation]),
        test=scale(x[indices.test]),
        indices=indices,
        scale=scale,
    )


class TensorActivationProvider(Iterator[torch.Tensor]):
    """Seeded tensor batches implementing the SAELens ``DataProvider`` protocol."""

    def __init__(
        self,
        activations: Any,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        repeat: bool = True,
        drop_last: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        self.activations = as_activation_tensor(activations)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(self.activations) == 0:
            raise ValueError("activations must contain at least one sample.")
        if drop_last and len(self.activations) < batch_size:
            raise ValueError("drop_last=True requires at least one full batch.")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.drop_last = drop_last
        self.device = device
        self._generator = torch.Generator().manual_seed(seed)
        self._order = torch.empty(0, dtype=torch.long)
        self._offset = 0
        self._exhausted = False

    def __iter__(self) -> TensorActivationProvider:
        return self

    def _start_epoch(self) -> None:
        if self._exhausted:
            raise StopIteration
        n = len(self.activations)
        self._order = (
            torch.randperm(n, generator=self._generator)
            if self.shuffle
            else torch.arange(n)
        )
        self._offset = 0

    def __next__(self) -> torch.Tensor:
        if len(self._order) == 0:
            self._start_epoch()
        remaining = len(self._order) - self._offset
        if remaining == 0 or (self.drop_last and remaining < self.batch_size):
            if not self.repeat:
                self._exhausted = True
                raise StopIteration
            self._start_epoch()
            remaining = len(self._order)
        take = min(self.batch_size, remaining)
        index = self._order[self._offset : self._offset + take]
        self._offset += take
        batch = self.activations[index]
        return batch if self.device is None else batch.to(self.device)


@dataclass(frozen=True)
class ActivationCache:
    activations: torch.Tensor
    token_ids: torch.Tensor | None = None
    group_ids: torch.Tensor | None = None
    hook_name: str | None = None

    @property
    def x(self) -> torch.Tensor:
        return self.activations


def _resolve_hook_name(column_names: list[str], hook_name: str | None) -> str:
    candidates = [name for name in column_names if name != "token_ids"]
    if hook_name is None:
        if len(candidates) != 1:
            raise ValueError(f"Specify hook_name; activation columns are {candidates}.")
        return candidates[0]
    if hook_name not in candidates:
        raise KeyError(
            f"Hook {hook_name!r} is not present; activation columns are {candidates}."
        )
    return hook_name


def load_activation_cache(
    path: str | Path,
    *,
    hook_name: str | None = None,
    max_samples: int | None = None,
    dtype: torch.dtype | None = torch.float32,
    excluded_token_ids: Collection[int] | None = None,
) -> ActivationCache:
    """Load a bounded local ``.pt`` cache or SAELens Arrow cache directory.

    SAELens stores one sequence per Arrow row with the hook column shaped
    ``(context_size, d_in)``. Leading dimensions are flattened to samples and
    the Arrow row number is retained as ``group_ids``. ``max_samples`` should
    be set for caches too large to materialize in RAM. Token exclusions keep
    activations, token IDs, and group IDs aligned.

    Arrow row grouping prevents positions from one packed context crossing a
    local split. Prefer separate source-dataset train/evaluation caches when
    available, since different rows may still originate from one document.
    """
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")
    excluded_values = () if excluded_token_ids is None else excluded_token_ids
    excluded = tuple(int(token_id) for token_id in excluded_values)

    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if isinstance(payload, torch.Tensor):
            activations, token_ids, group_ids = payload, None, None
        elif isinstance(payload, Mapping):
            activations = payload.get("activations")
            token_ids = payload.get("token_ids")
            group_ids = payload.get("group_ids")
        else:
            raise TypeError("A .pt activation cache must contain a tensor or mapping.")
        activations = as_activation_tensor(activations)
        resolved_hook = hook_name
    else:
        try:
            from datasets import load_from_disk
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "Loading a SAELens Arrow cache requires the 'datasets' package."
            ) from exc
        dataset = load_from_disk(str(cache_path))
        resolved_hook = _resolve_hook_name(list(dataset.column_names), hook_name)
        rows = dataset
        if max_samples is not None and len(dataset):
            context_size = len(dataset[0][resolved_hook])
            n_rows = min(
                len(dataset), (max_samples + context_size - 1) // context_size
            )
            rows = dataset.select(range(n_rows))
        columns = [resolved_hook] + (
            ["token_ids"] if "token_ids" in dataset.column_names else []
        )
        values = rows.with_format("torch", columns=columns)[:]
        activation_rows = values[resolved_hook]
        activations = as_activation_tensor(activation_rows)
        token_ids = values.get("token_ids")
        samples_per_row = activations.shape[0] // len(rows) if len(rows) else 0
        group_ids = torch.arange(len(rows)).repeat_interleave(samples_per_row)

    token_ids = token_ids.reshape(-1) if isinstance(token_ids, torch.Tensor) else None
    group_ids = group_ids.reshape(-1) if isinstance(group_ids, torch.Tensor) else None
    for name, values in (("token_ids", token_ids), ("group_ids", group_ids)):
        if values is not None and len(values) != len(activations):
            raise ValueError(f"{name} must contain one value per activation sample.")
    if excluded:
        if token_ids is None:
            raise ValueError(
                "excluded_token_ids requires token_ids in the activation cache."
            )
        keep = ~torch.isin(token_ids, torch.tensor(excluded, dtype=token_ids.dtype))
        activations, token_ids = activations[keep], token_ids[keep]
        if group_ids is not None:
            group_ids = group_ids[keep]
    if max_samples is not None:
        activations = activations[:max_samples]
        token_ids = token_ids[:max_samples] if token_ids is not None else None
        group_ids = group_ids[:max_samples] if group_ids is not None else None
    if dtype is not None:
        activations = activations.to(dtype=dtype)
    return ActivationCache(
        activations=activations,
        token_ids=token_ids,
        group_ids=group_ids,
        hook_name=resolved_hook,
    )
