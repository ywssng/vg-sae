"""Analytic checks of the specific published SAE variants used by this project.

Standard and Gated use RI-L1 (Anthropic April 2024; JumpReLU paper Appendix D),
not the original unit-decoder, frozen-auxiliary-decoder Gated training recipe.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from sae_lens.training.activation_scaler import ActivationScaler

from src.sae_baselines import (
    BatchTopKSAE,
    BatchTopKSAEConfig,
    GatedSAE,
    GatedSAEConfig,
    JumpReLU,
    StandardSAE,
    StandardSAEConfig,
    Step,
    TopKSAE,
    TopKSAEConfig,
    to_inference_sae,
)
from src.sae_loss import saelens_sae_loss_terms
from src.sae_train import _fold_activation_norm_scaling_factor, fit_sae


def test_standard_ri_l1_has_decoder_weighted_penalty_and_feature_sum_mse() -> None:
    model = StandardSAE(
        StandardSAEConfig(d_in=2, d_sae=2, l1_coefficient=0.5, apply_b_dec_to_input=False)
    )
    with torch.no_grad():
        model.W_enc.copy_(torch.eye(2))
        model.W_dec.copy_(torch.diag(torch.tensor([2.0, 3.0])))
    terms = saelens_sae_loss_terms(model, torch.tensor([[1.0, 2.0]]))

    # h = (1, 2), reconstruction = (2, 6), weighted L1 = 2 + 6.
    assert terms.reconstruction_loss.item() == pytest.approx(17.0)
    assert terms.reconstruction_mse.item() == pytest.approx(8.5)
    assert terms.sparsity_loss.item() == pytest.approx(4.0)
    assert terms.loss.item() == pytest.approx(21.0)
    decoder_grad = torch.autograd.grad(terms.sparsity_loss, model.W_dec)[0]
    torch.testing.assert_close(decoder_grad, torch.diag(torch.tensor([0.5, 1.0])))


def test_topk_aux_uses_dead_features_and_detaches_main_residual() -> None:
    model = TopKSAE(
        TopKSAEConfig(
            d_in=2,
            d_sae=2,
            k=1,
            aux_loss_coefficient=0.25,
            apply_b_dec_to_input=False,
        )
    )
    with torch.no_grad():
        model.W_enc.copy_(torch.eye(2))
        model.W_dec.copy_(torch.diag(torch.tensor([2.0, 3.0])))
    terms = saelens_sae_loss_terms(
        model, torch.tensor([[2.0, 1.0]]), torch.tensor([False, True])
    )

    # Norm-weighted top-k is (4, 0); main residual is (-2, 1).
    # The dead feature reconstructs (0, 3), so auxiliary loss is .25 * 8.
    torch.testing.assert_close(terms.feature_acts, torch.tensor([[4.0, 0.0]]))
    assert terms.reconstruction_loss.item() == pytest.approx(5.0)
    assert terms.auxiliary_loss.item() == pytest.approx(2.0)
    encoder_grad, decoder_grad = torch.autograd.grad(
        terms.auxiliary_loss, (model.W_enc, model.W_dec)
    )
    torch.testing.assert_close(encoder_grad, torch.tensor([[0.0, 6.0], [0.0, 3.0]]))
    torch.testing.assert_close(decoder_grad, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))


def test_gated_ri_l1_keeps_auxiliary_decoder_trainable() -> None:
    model = GatedSAE(
        GatedSAEConfig(d_in=2, d_sae=2, l1_coefficient=0.2, apply_b_dec_to_input=False)
    )
    with torch.no_grad():
        model.W_enc.copy_(torch.eye(2))
        model.W_dec.copy_(torch.diag(torch.tensor([2.0, 3.0])))
        model.b_gate.copy_(torch.tensor([-0.5, 0.5]))
        model.b_mag.copy_(torch.tensor([0.25, -0.5]))
        model.r_mag.copy_(torch.tensor([math.log(2.0), 0.0]))
    terms = saelens_sae_loss_terms(model, torch.tensor([[1.0, 2.0]]))

    # Main code (2.25, 1.5), gate auxiliary code (.5, 2.5).
    assert terms.reconstruction_loss.item() == pytest.approx(18.5)
    assert terms.sparsity_loss.item() == pytest.approx(1.7)
    assert terms.auxiliary_loss.item() == pytest.approx(30.25)
    decoder_grad, bias_grad, magnitude_grad = torch.autograd.grad(
        terms.auxiliary_loss, (model.W_dec, model.b_dec, model.b_mag), allow_unused=True
    )
    torch.testing.assert_close(decoder_grad, torch.tensor([[0.0, 5.5], [0.0, 27.5]]))
    torch.testing.assert_close(bias_grad, torch.tensor([0.0, 11.0]))
    assert magnitude_grad is None


def test_jumprelu_rectangle_surrogate_matches_paper_in_positive_window() -> None:
    pre = torch.tensor([[-0.2, 0.49, 0.5, 0.51, 0.8]], requires_grad=True)
    threshold = torch.full((5,), 0.5, requires_grad=True)
    output = JumpReLU.apply(pre, threshold, 0.1)
    support = Step.apply(pre, threshold, 0.1)
    torch.testing.assert_close(output, torch.tensor([[0.0, 0.0, 0.0, 0.51, 0.8]]))
    torch.testing.assert_close(support, torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]]))
    pre_grad, threshold_grad = torch.autograd.grad(
        output.sum() + 0.2 * support.sum(), (pre, threshold)
    )
    torch.testing.assert_close(pre_grad, torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]]))
    torch.testing.assert_close(threshold_grad, torch.tensor([0.0, -7.0, -7.0, -7.0, 0.0]))


def test_batchtopk_global_budget_and_exported_threshold_are_distinct() -> None:
    model = BatchTopKSAE(
        BatchTopKSAEConfig(d_in=2, d_sae=2, k=1.0, topk_threshold_lr=0.5)
    )
    with torch.no_grad():
        model.W_enc.copy_(torch.eye(2))
        model.W_dec.copy_(torch.eye(2))
    x = torch.tensor([[3.0, 2.0], [1.0, 0.5]])
    terms = saelens_sae_loss_terms(model, x, update_state=True)
    torch.testing.assert_close(terms.feature_acts, torch.tensor([[3.0, 2.0], [0.0, 0.0]]))
    assert model.topk_threshold.item() == pytest.approx(1.0)
    inference = to_inference_sae(model, fold_decoder_norm=True)
    torch.testing.assert_close(inference.encode(x), terms.feature_acts)
    torch.testing.assert_close(inference.encode(x[:1]), inference.encode(x)[:1])


@pytest.mark.parametrize("norm_weighted", [False, True])
def test_batchtopk_activation_scale_fold_preserves_inference(norm_weighted: bool) -> None:
    model = BatchTopKSAE(
        BatchTopKSAEConfig(
            d_in=2,
            d_sae=2,
            k=1.0,
            normalize_activations="expected_average_only_in",
            rescale_acts_by_decoder_norm=norm_weighted,
        )
    )
    with torch.no_grad():
        model.W_enc.copy_(torch.eye(2))
        model.W_dec.copy_(torch.eye(2))
        model.topk_threshold.fill_(1.0)
    x = torch.tensor([[0.75, 0.1], [0.2, 0.75]])
    expected = torch.tensor([[0.75, 0.0], [0.0, 0.75]])
    original = to_inference_sae(model, fold_decoder_norm=True)
    torch.testing.assert_close(original(2.0 * x) / 2.0, expected)

    _fold_activation_norm_scaling_factor(model, 2.0)
    folded = to_inference_sae(model, fold_decoder_norm=True)
    torch.testing.assert_close(folded(x), expected)
    torch.testing.assert_close(folded.encode(x).gt(0), original.encode(2.0 * x).gt(0))
    assert model.topk_threshold.item() == pytest.approx(0.5 if norm_weighted else 1.0)


def test_batchtopk_fit_folds_best_and_last_thresholds(monkeypatch) -> None:
    def fixed_scaling(self, **kwargs):
        self.scaling_factor = 2.0

    monkeypatch.setattr(ActivationScaler, "estimate_scaling_factor", fixed_scaling)
    torch.manual_seed(0)
    candidate = BatchTopKSAE(
        BatchTopKSAEConfig(
            d_in=2,
            d_sae=2,
            k=1.0,
            normalize_activations="expected_average_only_in",
            topk_threshold_lr=0.5,
        )
    )
    with torch.no_grad():
        candidate.W_enc.copy_(torch.eye(2))
        candidate.W_dec.copy_(torch.eye(2))
    reference = copy.deepcopy(candidate)
    reference.cfg.normalize_activations = "none"
    x = torch.tensor([[0.75, 0.1], [0.2, 0.75]])
    settings = dict(max_steps=3, batch_size=2, history_every=1, seed=7)
    actual = fit_sae(candidate, x, **settings)
    expected = fit_sae(reference, 2.0 * x, **settings)

    for actual_state, expected_state in (
        (actual.model.state_dict(), expected.model.state_dict()),
        (actual.best_state_dict, expected.best_state_dict),
    ):
        assert actual_state is not None and expected_state is not None
        folded = copy.deepcopy(actual.model)
        scaled = copy.deepcopy(expected.model)
        folded.load_state_dict(actual_state)
        scaled.load_state_dict(expected_state)
        folded = to_inference_sae(folded, fold_decoder_norm=True)
        scaled = to_inference_sae(scaled, fold_decoder_norm=True)
        torch.testing.assert_close(folded(x), scaled(2.0 * x) / 2.0)
        torch.testing.assert_close(folded.encode(x).gt(0), scaled.encode(2.0 * x).gt(0))
