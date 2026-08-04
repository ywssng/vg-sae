from __future__ import annotations

import math

import pytest
import torch

from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
from src.saelens_data import (
    ActivationCache,
    ActivationScale,
    TensorActivationProvider,
    load_activation_cache,
    make_split_indices,
    split_activations,
)


def test_sparse_coding_tensor_uses_one_seeded_split_and_train_only_scale() -> None:
    data = make_synthetic_sparse_coding(
        SyntheticSparseCodingConfig(input_dim=4, n_features=8, n_samples=30, seed=7)
    )
    first = split_activations(data, train_fraction=0.6, validation_fraction=0.2, seed=11)
    second = split_activations(data.x, train_fraction=0.6, validation_fraction=0.2, seed=11)

    assert torch.equal(first.indices.train, second.indices.train)
    assert torch.equal(first.train, second.train)
    assert first.train.norm(dim=-1).mean().item() == pytest.approx(math.sqrt(4))
    assert torch.equal(first.validation, first.scale(data.x[first.indices.validation]))
    assert torch.equal(first.test, first.scale(data.x[first.indices.test]))
    assert len(torch.unique(torch.cat(tuple(vars(first.indices).values())))) == len(data.x)


def test_split_normalization_does_not_fit_validation_or_test() -> None:
    indices = make_split_indices(10, train_fraction=0.6, validation_fraction=0.2, seed=3)
    x = torch.ones(10, 2)
    x[indices.validation] *= 10.0
    x[indices.test] *= 20.0
    splits = split_activations(x, train_fraction=0.6, validation_fraction=0.2, seed=3)

    assert splits.scale.factor == pytest.approx(1.0)
    assert splits.validation.norm(dim=-1).mean().item() == pytest.approx(10 * math.sqrt(2))
    assert splits.test.norm(dim=-1).mean().item() == pytest.approx(20 * math.sqrt(2))


def test_grouped_split_is_inferred_deterministic_and_disjoint() -> None:
    x = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    groups = torch.arange(4).repeat_interleave(3)
    cache = ActivationCache(x, group_ids=groups)

    first = split_activations(
        cache,
        train_fraction=0.5,
        validation_fraction=0.25,
        seed=9,
    )
    second = split_activations(
        x,
        train_fraction=0.5,
        validation_fraction=0.25,
        seed=9,
        group_ids=groups,
    )

    assert all(
        torch.equal(getattr(first.indices, name), getattr(second.indices, name))
        for name in ("train", "validation", "test")
    )
    split_groups = [
        set(groups[getattr(first.indices, name)].tolist())
        for name in ("train", "validation", "test")
    ]
    assert [len(values) for values in split_groups] == [2, 1, 1]
    assert not (split_groups[0] & split_groups[1])
    assert not (split_groups[0] & split_groups[2])
    assert not (split_groups[1] & split_groups[2])
    expected_scale = ActivationScale.fit(x[first.indices.train])
    assert first.scale.factor == pytest.approx(expected_scale.factor)


def test_grouped_split_rejects_misaligned_ids() -> None:
    with pytest.raises(ValueError, match="one value per activation"):
        split_activations(torch.ones(4, 2), group_ids=torch.arange(3))


def test_activation_scale_matches_saelens_expected_average_only_in() -> None:
    x = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    ours = ActivationScale.fit(x)

    sae_lens = pytest.importorskip("sae_lens")
    official = sae_lens.training.activation_scaler.ActivationScaler()
    official.estimate_scaling_factor(2, iter([x]), n_batches_for_norm_estimate=1)
    assert ours.factor == pytest.approx(official.scaling_factor)


def test_tensor_provider_is_reproducible_and_repeats_full_batches() -> None:
    x = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    providers = [TensorActivationProvider(x, 3, seed=5) for _ in range(2)]
    batches = [[next(provider) for _ in range(5)] for provider in providers]

    assert all(torch.equal(left, right) for left, right in zip(*batches, strict=True))
    assert all(batch.shape == (3, 2) for batch in batches[0])

    finite = TensorActivationProvider(x, 3, shuffle=False, repeat=False, drop_last=False)
    assert [len(next(finite)) for _ in range(3)] == [3, 3, 1]
    with pytest.raises(StopIteration):
        next(finite)


def test_load_pt_cache_and_limit_samples(tmp_path) -> None:
    path = tmp_path / "activations.pt"
    activations = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)
    token_ids = torch.arange(8).reshape(2, 4)
    group_ids = torch.arange(2).repeat_interleave(4)
    torch.save(
        {"activations": activations, "token_ids": token_ids, "group_ids": group_ids},
        path,
    )

    cache = load_activation_cache(path, max_samples=5)
    assert cache.activations.shape == (5, 3)
    assert cache.activations.dtype == torch.float32
    assert torch.equal(cache.token_ids, torch.arange(5))
    assert torch.equal(cache.group_ids, group_ids[:5])


def test_token_exclusion_preserves_cache_alignment(tmp_path) -> None:
    path = tmp_path / "activations.pt"
    activations = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    token_ids = torch.tensor([10, 11, 12, 10, 13, 14])
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    torch.save(
        {"activations": activations, "token_ids": token_ids, "group_ids": group_ids},
        path,
    )

    cache = load_activation_cache(path, excluded_token_ids={10, 13})
    keep = torch.tensor([1, 2, 5])
    assert torch.equal(cache.activations, activations[keep])
    assert torch.equal(cache.token_ids, token_ids[keep])
    assert torch.equal(cache.group_ids, group_ids[keep])


def test_token_exclusion_requires_token_ids(tmp_path) -> None:
    path = tmp_path / "activations.pt"
    torch.save(torch.ones(3, 2), path)

    with pytest.raises(ValueError, match="requires token_ids"):
        load_activation_cache(path, excluded_token_ids={0})


def test_load_official_saelens_arrow_cache_shape(tmp_path) -> None:
    datasets = pytest.importorskip("datasets")
    hook_name = "blocks.1.hook_resid_post"
    activations = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    token_ids = torch.arange(6, dtype=torch.int32).reshape(3, 2)
    features = datasets.Features(
        {
            hook_name: datasets.Array2D(shape=(2, 3), dtype="float32"),
            "token_ids": datasets.Sequence(datasets.Value("int32"), length=2),
        }
    )
    dataset = datasets.Dataset.from_dict(
        {hook_name: activations, "token_ids": token_ids}, features=features
    )
    path = tmp_path / "saelens_cache"
    dataset.save_to_disk(path)

    cache = load_activation_cache(path, hook_name=hook_name, max_samples=5)
    assert cache.hook_name == hook_name
    assert torch.equal(cache.activations, activations.reshape(-1, 3)[:5])
    assert torch.equal(cache.token_ids, token_ids.reshape(-1)[:5])
    assert torch.equal(cache.group_ids, torch.tensor([0, 0, 1, 1, 2]))

    filtered = load_activation_cache(
        path, hook_name=hook_name, excluded_token_ids={1, 4}
    )
    keep = torch.tensor([0, 2, 3, 5])
    assert torch.equal(filtered.activations, activations.reshape(-1, 3)[keep])
    assert torch.equal(filtered.token_ids, token_ids.reshape(-1)[keep])
    assert torch.equal(filtered.group_ids, torch.tensor([0, 1, 1, 2]))
