import math

import pytest
import torch

from src.sae_loss import sae_loss_terms
from src.sae_model import (
    BatchTopKSAE,
    BatchTopKSAEConfig,
    GatedSAE,
    GatedSAEConfig,
    JumpReLU,
    JumpReLUSAE,
    JumpReLUSAEConfig,
    L1ReLUSAE,
    L1SAEConfig,
    Step,
    TopKSAE,
    TopKSAEConfig,
)
from src.sae_train import build_sae, fit_sae


def test_baselines_use_centered_inputs_and_tied_unit_norm_initialization() -> None:
    models = [
        L1ReLUSAE(L1SAEConfig(3, 5)),
        TopKSAE(TopKSAEConfig(3, 5, k=2)),
        BatchTopKSAE(BatchTopKSAEConfig(3, 5, k=2)),
        JumpReLUSAE(JumpReLUSAEConfig(3, 5)),
        GatedSAE(GatedSAEConfig(3, 5)),
    ]
    for model in models:
        assert torch.allclose(model.encoder.weight, model.decoder.weight.t())
        assert torch.allclose(model.decoder_column_norms(), torch.ones(5))
        with torch.no_grad():
            model.decoder.bias.fill_(2.0)
            model.encoder.weight.zero_()
            model.encoder.weight[:, 0] = 1.0
            model.encoder.bias.zero_()
        assert torch.allclose(model.pre_activations(torch.full((1, 3), 2.0)), torch.zeros(1, 5))


def test_topk_selects_raw_order_then_relu_and_reports_actual_l0() -> None:
    model = TopKSAE(TopKSAEConfig(2, 3, k=2, decoder_bias=False))
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(torch.tensor([-3.0, -1.0, -2.0]))
    output = model(torch.zeros(1, 2))
    assert torch.equal(output["selection_mask"], torch.tensor([[0.0, 1.0, 1.0]]))
    assert output["mask"].sum() == 0
    assert sae_loss_terms(model, torch.zeros(1, 2)).rho == 0


def test_topk_auxk_uses_scaled_preactivations_detached_residual_and_no_bias() -> None:
    model = TopKSAE(TopKSAEConfig(4, 4, k=1, decoder_bias=True))
    with torch.no_grad():
        model.decoder.weight.copy_(torch.eye(4))
        model.decoder.bias.fill_(7.0)
    x_hat = torch.zeros(1, 4, requires_grad=True)
    output = {
        "x_hat": x_hat,
        "hidden_pre": torch.tensor([[0.0, 3.0, 2.0, 0.0]], requires_grad=True),
    }
    x = torch.tensor([[0.0, 3.0, 2.0, 0.0]])
    loss = model.auxiliary_loss(x, output, torch.tensor([False, True, True, False]))
    assert loss == 0
    loss.backward()
    assert x_hat.grad is None


def test_topk_auxk_rescales_non_unit_decoder_consistently() -> None:
    model = TopKSAE(TopKSAEConfig(2, 2, k=1, decoder_bias=False))
    with torch.no_grad():
        model.decoder.weight.copy_(torch.diag(torch.tensor([2.0, 3.0])))
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(torch.tensor([1.0, -1.0]))
    output = model(torch.zeros(1, 2))
    assert torch.equal(output["hidden_pre"], torch.tensor([[2.0, -3.0]]))

    aux_output = {"x_hat": torch.zeros(1, 2), "hidden_pre": output["hidden_pre"]}
    loss = model.auxiliary_loss(
        torch.tensor([[2.0, 0.0]]), aux_output, torch.tensor([True, False])
    )
    assert loss == 0


def test_gated_sae_has_shared_hard_gate_and_full_saelens_loss() -> None:
    model = GatedSAE(GatedSAEConfig(2, 2, l1_coefficient=0.25, gate_bias_init=0.0))
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(2))
        model.encoder.bias.zero_()
        model.r_mag.copy_(torch.tensor([math.log(2.0), 0.0]))
        model.b_mag.copy_(torch.tensor([0.0, 1.0]))
        model.decoder.weight.copy_(torch.eye(2))
        model.decoder.bias.copy_(torch.tensor([0.5, -0.5]))
    x = torch.tensor([[1.5, 0.5]])
    output, terms = model(x), sae_loss_terms(model, x)
    assert torch.equal(output["gate"], torch.ones(1, 2))
    assert torch.allclose(output["h"], torch.tensor([[2.0, 2.0]]))
    assert torch.allclose(terms.reconstruction_loss, torch.tensor(2.0))
    assert torch.allclose(terms.sparsity_loss, torch.tensor(0.5))
    assert torch.allclose(terms.auxiliary_loss, torch.tensor(0.0))
    assert torch.allclose(terms.loss, torch.tensor(2.5))


def test_gated_sae_reports_actual_positive_code_density() -> None:
    model = GatedSAE(GatedSAEConfig(1, 1, decoder_bias=False))
    with torch.no_grad():
        model.encoder.weight.fill_(1.0)
        model.encoder.bias.zero_()
        model.b_mag.fill_(-2.0)
    output = model(torch.ones(1, 1))
    assert output["gate"].item() == 1.0
    assert output["mask"].item() == 0.0
    assert sae_loss_terms(model, torch.ones(1, 1)).rho.item() == 0.0


def test_batchtopk_global_count_ema_and_eval_are_batch_independent() -> None:
    model = BatchTopKSAE(BatchTopKSAEConfig(2, 3, k=1.5, topk_threshold_lr=0.5))
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(torch.tensor([0.2, 0.8, 1.2]))
    model.train()
    output = model(torch.zeros(2, 2))
    assert (output["h"] > 0).sum() == 3
    model.update_topk_threshold(output["h"])
    assert model.topk_threshold.dtype == torch.double
    assert model.topk_threshold == pytest.approx(0.4)

    model.topk_threshold.fill_(0.5)
    model.eval()
    one = model(torch.zeros(1, 2))["h"]
    many = model(torch.zeros(7, 2))["h"]
    assert torch.equal(one.expand_as(many), many)
    before = model.topk_threshold.clone()
    sae_loss_terms(model, torch.zeros(7, 2))
    assert torch.equal(model.topk_threshold, before)


def test_batchtopk_allows_zero_global_selections() -> None:
    model = BatchTopKSAE(BatchTopKSAEConfig(2, 3, k=0.1))
    with torch.no_grad():
        model.encoder.bias.fill_(1.0)
    assert model(torch.zeros(2, 2))["h"].count_nonzero() == 0


def test_batchtopk_counts_all_non_feature_dimensions() -> None:
    model = BatchTopKSAE(BatchTopKSAEConfig(4, 5, k=1.0, decoder_bias=False))
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(torch.arange(1.0, 6.0))
    assert model(torch.zeros(2, 3, 4))["h"].count_nonzero() == 6


def test_fit_updates_batchtopk_ema_once_not_during_history_eval() -> None:
    model = BatchTopKSAE(BatchTopKSAEConfig(2, 3, k=1.0, topk_threshold_lr=1.0))
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(torch.tensor([0.2, 0.8, 1.2]))
    fit_sae(model, torch.zeros(2, 2), lr=0.0, max_steps=1, history_every=1)
    assert model.topk_threshold.dtype == torch.double
    assert model.topk_threshold == pytest.approx(1.2)


def test_dead_features_start_strictly_after_window(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.sae_train as train_module

    model = TopKSAE(TopKSAEConfig(2, 3, k=1))
    seen: list[torch.Tensor] = []
    original = train_module.sae_loss_terms

    def record_mask(model_arg, batch, dead_feature_mask=None):  # type: ignore[no-untyped-def]
        if model_arg.training:
            seen.append(dead_feature_mask.clone())
        return original(model_arg, batch, dead_feature_mask)

    monkeypatch.setattr(train_module, "sae_loss_terms", record_mask)
    fit_sae(model, torch.zeros(2, 2), lr=0.0, max_steps=3, dead_feature_window=1)
    assert [bool(mask.any()) for mask in seen] == [False, False, True]


def test_jumprelu_and_step_match_strict_forward_and_analytical_ste() -> None:
    x = torch.tensor([[0.98, 1.0, 1.02]], requires_grad=True)
    threshold = torch.ones(3, requires_grad=True)
    output = JumpReLU.apply(x, threshold, 0.1)
    assert torch.equal(output, torch.tensor([[0.0, 0.0, 1.02]]))
    output.sum().backward()
    assert torch.equal(x.grad, torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(threshold.grad, torch.full((3,), -10.0))

    step_threshold = torch.ones(3, requires_grad=True)
    Step.apply(x.detach(), step_threshold, 0.1).sum().backward()
    assert torch.allclose(step_threshold.grad, torch.full((3,), -10.0))


def test_jumprelu_step_l0_and_optional_dead_pre_act_loss() -> None:
    model = JumpReLUSAE(
        JumpReLUSAEConfig(
            2,
            2,
            jumprelu_init_threshold=1.0,
            l0_coefficient=0.3,
            pre_act_loss_coefficient=0.2,
        )
    )
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(torch.tensor([1.0, 2.0]))
    terms = sae_loss_terms(model, torch.zeros(1, 2), torch.tensor([True, False]))
    assert float(terms.rho) == pytest.approx(0.5)
    assert float(terms.sparsity_loss.detach()) == pytest.approx(0.3)
    assert float(terms.auxiliary_loss.detach()) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("batch-topk", BatchTopKSAE),
        ("batch_topk", BatchTopKSAE),
        ("jump-relu", JumpReLUSAE),
        ("jump_relu", JumpReLUSAE),
    ],
)


def test_builder_aliases(alias: str, expected: type[torch.nn.Module]) -> None:
    model = build_sae(alias, 3, 6, k=2) if "batch" in alias else build_sae(alias, 3, 6)
    assert isinstance(model, expected)
