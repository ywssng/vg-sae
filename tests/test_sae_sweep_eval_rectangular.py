from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.sae_sweep_eval as sweep_eval
from src.sae_model import VGSAEConfig, VariationalGarroteSAE
from src.sae_sweep import (
    SweepConfig,
    SyntheticDataConfig,
    TrainingConfig,
    build_model,
    build_specs,
    make_train_test,
)


class _DummyModel(torch.nn.Module):
    pass


def _evaluate_with_fixed_latents(
    monkeypatch,
    *,
    test_h: torch.Tensor,
    test_mask: torch.Tensor,
    support: torch.Tensor,
    z: torch.Tensor,
    learned_idx: list[int],
    true_idx: list[int],
    method: str,
):
    train_x = torch.eye(2)
    test_x = torch.eye(2)
    train_h = torch.ones_like(test_h)
    train_data = SimpleNamespace(x=train_x)
    test_data = SimpleNamespace(
        x=test_x,
        clean_x=test_x,
        dictionary=torch.zeros(2, support.shape[1]),
        support=support,
        z=z,
    )

    def fake_latents(_model, x, l1_threshold=None):
        del l1_threshold
        if x is train_x:
            return train_h, torch.ones_like(train_h), {"l1_gmm_threshold": 0.5}
        return test_h, test_mask, {"l1_raw_relu_density": -1.0}

    monkeypatch.setattr(sweep_eval, "_latents_and_masks", fake_latents)
    monkeypatch.setattr(sweep_eval, "_reconstruct", lambda _model, x: x)
    monkeypatch.setattr(sweep_eval, "_decode_latents", lambda _model, _h: test_x)
    monkeypatch.setattr(
        sweep_eval,
        "_decoder_matching",
        lambda _model, _dictionary: (
            np.asarray(learned_idx),
            np.asarray(true_idx),
            np.ones(len(learned_idx)),
            0.9,
        ),
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            mask_threshold=0.5,
            train_steps=10,
            dead_feature_window=2,
            dead_threshold=1e-6,
        )
    )
    spec = SimpleNamespace(
        seed=0,
        init_seed=1,
        method=method,
        control_name="control",
        control_value=1.0,
    )
    model = _DummyModel()
    model.W_dec = torch.eye(test_h.shape[1])
    return sweep_eval.evaluate_model(
        model, train_data, test_data, config, spec, run_id="fixed"
    )


def test_wider_sae_counts_unmatched_latents_as_false_positives(monkeypatch):
    test_h = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]])
    test_mask = (test_h > 0).float()
    support = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z = torch.tensor([[1.0, 0.0], [0.0, 3.0]])

    row, arrays = _evaluate_with_fixed_latents(
        monkeypatch,
        test_h=test_h,
        test_mask=test_mask,
        support=support,
        z=z,
        learned_idx=[0, 1],
        true_idx=[0, 1],
        method="l1",
    )

    assert row["rho_model"] == pytest.approx(4 / 6)
    assert row["average_l0"] == pytest.approx(2.0)
    assert row["expected_l0"] == pytest.approx(2.0)
    assert row["l1_raw_relu_rho_model"] == pytest.approx(4 / 6)
    assert row["selection_error"] == pytest.approx(1 / 3)
    assert row["support_precision"] == pytest.approx(0.5)
    assert row["support_recall"] == pytest.approx(1.0)
    assert row["support_f1"] == pytest.approx(2 / 3)
    assert row["generalization_error"] == pytest.approx(np.sqrt(2.0))
    assert row["hard_generalization_error"] == pytest.approx(np.sqrt(2.0))
    assert row["hard_selection_error"] == pytest.approx(1 / 3)
    assert row["matched_latent_count"] == 2
    assert row["union_width"] == 3
    assert row["unmatched_ground_truth_features"] == 0
    assert row["unmatched_sae_latents"] == 1
    assert row["ground_truth_match_coverage"] == pytest.approx(1.0)
    assert row["sae_latent_match_coverage"] == pytest.approx(2 / 3)
    assert row["decoder_recovery_cosine"] == pytest.approx(0.6)
    assert row["matched_decoder_recovery_cosine"] == pytest.approx(0.9)
    assert row["decoder_pairwise_cosine_similarity"] == pytest.approx(0.0)
    assert arrays["mask"].shape == (2, 3)
    np.testing.assert_array_equal(arrays["true_support"][:, 2], 0.0)
    np.testing.assert_array_equal(arrays["raw_mask"], test_mask.numpy())
    np.testing.assert_array_equal(arrays["hard_mask"], test_mask.numpy())
    np.testing.assert_array_equal(arrays["union_learned_idx"], [0, 1, 2])
    np.testing.assert_array_equal(arrays["union_true_idx"], [0, 1, -1])


def test_narrower_sae_counts_unmatched_truth_as_false_negatives(monkeypatch):
    test_h = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    test_mask = (test_h > 0).float()
    support = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    z = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    row, arrays = _evaluate_with_fixed_latents(
        monkeypatch,
        test_h=test_h,
        test_mask=test_mask,
        support=support,
        z=z,
        learned_idx=[0, 1],
        true_idx=[0, 1],
        method="vgsae",
    )

    assert row["rho_model"] == pytest.approx(0.5)
    assert row["average_l0"] == pytest.approx(1.0)
    assert row["expected_l0"] == pytest.approx(1.0)
    assert row["selection_error"] == pytest.approx(1 / 6)
    assert row["support_precision"] == pytest.approx(1.0)
    assert row["support_recall"] == pytest.approx(2 / 3)
    assert row["support_f1"] == pytest.approx(0.8)
    assert row["generalization_error"] == pytest.approx(np.sqrt(4 / 14))
    assert row["hard_generalization_error"] == pytest.approx(np.sqrt(4 / 14))
    assert row["hard_selection_error"] == pytest.approx(1 / 6)
    assert row["matched_latent_count"] == 2
    assert row["union_width"] == 3
    assert row["unmatched_ground_truth_features"] == 1
    assert row["unmatched_sae_latents"] == 0
    assert row["ground_truth_match_coverage"] == pytest.approx(2 / 3)
    assert row["sae_latent_match_coverage"] == pytest.approx(1.0)
    assert row["decoder_recovery_cosine"] == pytest.approx(0.6)
    assert row["matched_decoder_recovery_cosine"] == pytest.approx(0.9)
    np.testing.assert_array_equal(arrays["mask"][:, 2], 0.0)
    np.testing.assert_array_equal(arrays["true_support"], support.numpy())
    np.testing.assert_array_equal(arrays["raw_mask"], test_mask.numpy())
    np.testing.assert_array_equal(
        arrays["hard_mask"], [[1, 0, 0], [0, 1, 0]]
    )
    np.testing.assert_array_equal(arrays["union_learned_idx"], [0, 1, -1])
    np.testing.assert_array_equal(arrays["union_true_idx"], [0, 1, 2])


def test_equal_width_union_preserves_original_alignment():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    learned_idx = np.array([1, 0])
    true_idx = np.array([0, 1])
    signs = np.array([-1.0, 1.0])

    aligned, union_target, union_learned, union_true = sweep_eval._rectangular_union(
        values, target, learned_idx, true_idx, signs
    )

    np.testing.assert_array_equal(
        aligned,
        sweep_eval._align(values, learned_idx, true_idx, target.shape[1], signs),
    )
    np.testing.assert_array_equal(union_target, target.numpy())
    np.testing.assert_array_equal(union_learned, [1, 0])
    np.testing.assert_array_equal(union_true, [0, 1])


def test_vg_hard_latents_threshold_posterior_before_applying_amplitude():
    model = VariationalGarroteSAE(
        VGSAEConfig(input_dim=2, n_latents=2, decoder_bias=False)
    )
    x = torch.zeros(1, 2)
    with torch.no_grad():
        model.gate_encoder.weight.zero_()
        model.gate_encoder.bias.copy_(torch.logit(torch.tensor([0.4, 0.75])))
        model.amplitude_encoder.weight.zero_()
        model.amplitude_encoder.bias.copy_(
            torch.log(torch.expm1(torch.tensor([2.0, 3.0])))
        )

    native_h, mask, _ = sweep_eval._latents_and_masks(model, x)
    hard_h = sweep_eval._hard_latents(model, x, native_h, mask, threshold=0.5)

    assert torch.allclose(native_h, torch.tensor([[0.8, 2.25]]), atol=1e-6)
    assert torch.allclose(hard_h, torch.tensor([[0.0, 3.0]]), atol=1e-6)


@pytest.mark.parametrize(
    ("method", "control"),
    [
        ("vgsae", 0.5),
        ("l1", 0.01),
        ("topk", 2),
        ("batchtopk", 2),
        ("jumprelu", 0.01),
        ("gated", 0.01),
    ],
)
def test_every_stage1_architecture_decodes_its_hard_code(
    method: str, control: float
) -> None:
    config = SweepConfig(
        data=SyntheticDataConfig(
            input_dim=3,
            ground_truth_num_features=5,
            sae_width=5,
            n_train=16,
            n_test=8,
            support_density=0.2,
        ),
        training=TrainingConfig(train_steps=1, history_every=1),
        methods=[method],
        controls={method: [control]},
    )
    config.validate()
    spec = build_specs(config)[0]
    train_data, test_data = make_train_test(config, spec.seed, "cpu")

    row, cache = sweep_eval.evaluate_model(
        build_model(config, spec), train_data, test_data, config, spec
    )

    for metric in (
        "hard_reconstruction_error",
        "hard_generalization_error",
        "hard_selection_error",
        "rho_model_hard",
    ):
        assert np.isfinite(row[metric])
    assert row["rho_model_hard"] == pytest.approx(
        row["average_l0"] / config.data.sae_width
    )
    assert set(np.unique(cache["hard_mask"])) <= {0, 1}
