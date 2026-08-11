from __future__ import annotations

import numpy as np
import pytest
import torch

from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
from src.sae_sweep import (
    SweepConfig,
    SyntheticDataConfig,
    build_model,
    build_specs,
    make_train_test,
)


def test_stage1_data_is_clean_overcomplete_skewed_and_nonnegative() -> None:
    data = make_synthetic_sparse_coding(
        SyntheticSparseCodingConfig(
            input_dim=4,
            ground_truth_num_features=9,
            n_samples=20_000,
            support_density=0.15,
            noise_std=0.0,
            frequency_skew=0.7,
            amplitude_scale=2.0,
            seed=3,
        )
    )

    assert data.dictionary.shape == (4, 9)
    assert torch.allclose(data.dictionary.norm(dim=0), torch.ones(9))
    assert torch.equal(data.x, data.clean_x)
    assert torch.all(data.z >= 0.0)
    assert torch.equal(data.z > 0.0, data.support.bool())
    assert torch.all(data.feature_probabilities[:-1] > data.feature_probabilities[1:])
    assert data.feature_probabilities.mean().item() == pytest.approx(0.15)

    support_correlation = np.corrcoef(data.support.numpy(), rowvar=False)
    off_diagonal = support_correlation[~np.eye(9, dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.03


def test_ground_truth_width_and_sae_width_are_independent() -> None:
    config = SweepConfig(
        data=SyntheticDataConfig(
            input_dim=3,
            ground_truth_num_features=7,
            sae_width=5,
            n_train=8,
            n_test=4,
        ),
        methods=["vgsae", "l1", "topk", "batchtopk", "jumprelu", "gated"],
        controls={
            "vgsae": [0.0],
            "l1": [0.0],
            "topk": [2],
            "batchtopk": [2.0],
            "jumprelu": [0.0],
            "gated": [0.0],
        },
    )
    specs = build_specs(config)
    train, test = make_train_test(config, specs[0].seed)

    assert train.x.shape == (8, 3)
    assert test.x.shape == (4, 3)
    assert train.z.shape == (8, 7)
    assert train.dictionary.shape == (3, 7)
    for spec in specs:
        model = build_model(config, spec)
        model_width = model.cfg.d_sae if hasattr(model, "cfg") else model.config.n_latents
        assert model_width == 5


def test_legacy_n_features_sets_both_feature_counts() -> None:
    data = SyntheticDataConfig(input_dim=3, n_features=7)
    assert data.ground_truth_num_features == 7
    assert data.sae_width == 7

    payload = SweepConfig(
        data=data,
        methods=["vgsae"],
        controls={"vgsae": [0.0]},
    ).to_dict()
    assert "n_features" not in payload["data"]
    assert payload["data"]["ground_truth_num_features"] == 7
    assert payload["data"]["sae_width"] == 7

    legacy_data = dict(payload["data"])
    legacy_data["n_features"] = legacy_data.pop("ground_truth_num_features")
    legacy_data.pop("sae_width")
    restored = SyntheticDataConfig.from_dict(legacy_data)
    assert restored.ground_truth_num_features == restored.sae_width == 7


def test_legacy_sweep_data_keeps_coherence_noise_and_uniform_frequency() -> None:
    data = SyntheticDataConfig.from_dict(
        {
            "kind": "synthetic_sparse_coding",
            "input_dim": 3,
            "n_features": 7,
            "coherence": 0.2,
            "noise_std": 0.1,
            "frequency_skew": 0.0,
        }
    )
    config = SweepConfig(
        data=data,
        methods=["vgsae"],
        controls={"vgsae": [0.0]},
    )

    config.validate()
    assert data.ground_truth_num_features == data.sae_width == 7
    assert data.coherence == 0.2
    assert data.noise_std == 0.1


def test_generic_generator_preserves_legacy_coherence_and_noise_controls() -> None:
    data = make_synthetic_sparse_coding(
        SyntheticSparseCodingConfig(
            input_dim=4,
            n_features=8,
            n_samples=32,
            coherence=0.2,
            noise_std=0.1,
            frequency_skew=0.0,
            seed=5,
        )
    )

    assert not torch.equal(data.x, data.clean_x)
    assert torch.allclose(data.feature_probabilities, torch.full((8,), 0.05))


@pytest.mark.parametrize(
    "override, message",
    [
        ({"ground_truth_num_features": 4}, "ground_truth_num_features > input_dim"),
        ({"frequency_skew": 0.0}, "frequency_skew must be positive"),
        ({"support_density": 0.9}, "too high to preserve its requested mean"),
        ({"coherence": 0.1}, "does not add dictionary coherence"),
        ({"noise_std": 0.1}, "does not add observation noise"),
    ],
)
def test_stage1_config_rejects_nonbaseline_data(
    override: dict[str, float | int], message: str
) -> None:
    values = {"input_dim": 4, "ground_truth_num_features": 8, **override}
    config = SweepConfig(
        data=SyntheticDataConfig(**values),
        methods=["vgsae"],
        controls={"vgsae": [0.0]},
    )
    with pytest.raises(ValueError, match=message):
        config.validate()
