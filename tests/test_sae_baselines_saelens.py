from __future__ import annotations

import copy
import importlib.metadata
import json

import pytest
import sae_lens
import torch

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE as OfficialBatchTopKSAE,
    BatchTopKTrainingSAEConfig as OfficialBatchTopKSAEConfig,
)
from sae_lens.saes.gated_sae import (
    GatedTrainingSAE as OfficialGatedSAE,
    GatedTrainingSAEConfig as OfficialGatedSAEConfig,
)
from sae_lens.saes.jumprelu_sae import (
    JumpReLU as OfficialJumpReLU,
    JumpReLUSAE as OfficialJumpReLUInferenceSAE,
    JumpReLUSAEConfig as OfficialJumpReLUInferenceSAEConfig,
    JumpReLUTrainingSAE as OfficialJumpReLUSAE,
    JumpReLUTrainingSAEConfig as OfficialJumpReLUSAEConfig,
    Step as OfficialStep,
)
from sae_lens.saes.sae import TrainCoefficientConfig, TrainingSAE, TrainStepInput
from sae_lens.saes.standard_sae import (
    StandardSAE as OfficialL1InferenceSAE,
    StandardSAEConfig as OfficialL1InferenceSAEConfig,
    StandardTrainingSAE as OfficialL1SAE,
    StandardTrainingSAEConfig as OfficialL1SAEConfig,
)
from sae_lens.saes.topk_sae import (
    TopKTrainingSAE as OfficialTopKSAE,
    TopKTrainingSAEConfig as OfficialTopKSAEConfig,
)
from sae_lens.training.sae_trainer import SAETrainer as OfficialSAETrainer

import src.sae_baselines as baseline_module
import src.sae_model as model_module
from src.sae_loss import sae_loss_terms, saelens_sae_loss_terms
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
from src.sae_train import _CyclingTensorBatches, build_sae, fit_sae, to_inference_sae


SAELENS_COMMIT = "8be14080485952f729ed58d674bcddf9778e0aa4"


def test_saelens_dependency_is_the_reviewed_revision() -> None:
    assert sae_lens.__version__ == "6.47.0"
    direct_url = importlib.metadata.distribution("sae-lens").read_text("direct_url.json")
    assert direct_url is not None
    vcs_info = json.loads(direct_url)["vcs_info"]
    assert vcs_info["commit_id"] == SAELENS_COMMIT
    assert vcs_info["requested_revision"] == SAELENS_COMMIT


def test_public_baselines_and_configs_are_exact_official_objects() -> None:
    assert L1ReLUSAE is OfficialL1SAE
    assert L1SAEConfig is OfficialL1SAEConfig
    assert TopKSAE is OfficialTopKSAE
    assert TopKSAEConfig is OfficialTopKSAEConfig
    assert BatchTopKSAE is OfficialBatchTopKSAE
    assert BatchTopKSAEConfig is OfficialBatchTopKSAEConfig
    assert JumpReLUSAE is OfficialJumpReLUSAE
    assert JumpReLUSAEConfig is OfficialJumpReLUSAEConfig
    assert GatedSAE is OfficialGatedSAE
    assert GatedSAEConfig is OfficialGatedSAEConfig
    assert Step is OfficialStep
    assert JumpReLU is OfficialJumpReLU
    assert not hasattr(baseline_module, "SAEConfig")
    assert not hasattr(baseline_module, "CenteredLinearSAE")
    assert not hasattr(baseline_module, "UnitNormDecoderMixin")
    assert not hasattr(model_module, "SAEConfig")
    assert not hasattr(model_module, "CenteredLinearSAE")
    assert not hasattr(model_module, "UnitNormDecoderMixin")
    assert not hasattr(baseline_module, "SAELensTrainingSAEAdapter")
    assert not hasattr(baseline_module, "unwrap_saelens_training_sae")


def _coefficients(model: TrainingSAE) -> dict[str, float]:
    return {
        name: float(value.value if isinstance(value, TrainCoefficientConfig) else value)
        for name, value in model.get_coefficients().items()
    }


@pytest.mark.parametrize(
    ("model_type", "config_type", "kwargs"),
    [
        (L1ReLUSAE, L1SAEConfig, {"l1_coefficient": 0.2}),
        (TopKSAE, TopKSAEConfig, {"k": 2}),
        (BatchTopKSAE, BatchTopKSAEConfig, {"k": 2.0}),
        (JumpReLUSAE, JumpReLUSAEConfig, {"l0_coefficient": 0.2}),
        (GatedSAE, GatedSAEConfig, {"l1_coefficient": 0.2}),
    ],
)
def test_official_loss_mapping_and_one_step_smoke(
    model_type: type[TrainingSAE],
    config_type: type,
    kwargs: dict[str, float | int],
) -> None:
    torch.manual_seed(0)
    model = model_type(config_type(d_in=4, d_sae=6, **kwargs))
    x = torch.randn(8, 4)
    dead = torch.ones(6, dtype=torch.bool)
    step_input = TrainStepInput(x, _coefficients(model), dead, 0, False)
    official = (
        TrainingSAE.training_forward_pass(model, step_input)
        if isinstance(model, OfficialBatchTopKSAE)
        else model.training_forward_pass(step_input)
    )
    terms = saelens_sae_loss_terms(model, x, dead)

    assert torch.equal(terms.loss, official.loss)
    assert torch.equal(terms.reconstruction_loss, official.losses["mse_loss"])
    assert torch.equal(terms.feature_acts, official.feature_acts)
    assert torch.equal(terms.reconstruction_mse, official.losses["mse_loss"] / 4)

    result = fit_sae(model, x, max_steps=1, batch_size=8, history_every=1)
    assert result.model is model
    assert torch.isfinite(torch.tensor(result.history[-1]["loss"]))


def test_read_only_batchtopk_loss_does_not_update_ema() -> None:
    model = BatchTopKSAE(
        BatchTopKSAEConfig(d_in=2, d_sae=3, k=1.5, topk_threshold_lr=0.5)
    )
    before = model.topk_threshold.clone()
    sae_loss_terms(model, torch.randn(7, 2))
    assert torch.equal(model.topk_threshold, before)


@pytest.mark.parametrize(
    ("model_type", "config_type", "kwargs"),
    [
        pytest.param(
            L1ReLUSAE,
            L1SAEConfig,
            {"l1_coefficient": 0.2},
            id="standard-l1",
        ),
        pytest.param(
            JumpReLUSAE,
            JumpReLUSAEConfig,
            {"l0_coefficient": 0.2},
            id="jumprelu",
        ),
    ],
)
def test_fit_matches_official_trainer_on_a_deterministic_schedule(
    model_type: type[TrainingSAE],
    config_type: type,
    kwargs: dict[str, float],
) -> None:
    torch.manual_seed(0)
    candidate = model_type(config_type(d_in=2, d_sae=3, **kwargs))
    reference = copy.deepcopy(candidate)
    x = torch.tensor(
        [[-1.0, 0.5], [0.2, 1.0], [0.7, -0.4], [1.2, 0.3], [0.0, -0.8],
         [0.9, 0.6], [-0.5, -0.2], [0.4, 0.1]]
    )
    settings = dict(lr=3.0e-4, batch_size=4, max_steps=3, dead_feature_window=1, seed=9)
    fit_sae(candidate, x, history_every=1, **settings)

    provider = _CyclingTensorBatches(x, settings["batch_size"], settings["seed"])
    trainer = OfficialSAETrainer(
        cfg=SAETrainerConfig(
            total_training_samples=settings["max_steps"] * settings["batch_size"],
            train_batch_size_samples=settings["batch_size"],
            lr=settings["lr"],
            lr_end=settings["lr"],
            lr_scheduler_name="constant",
            device="cpu",
            dead_feature_window=settings["dead_feature_window"],
            logger=LoggingConfig(log_to_wandb=False),
        ),
        sae=reference,
        data_provider=provider,
    )
    trainer.fit()

    for name, expected in reference.state_dict().items():
        assert torch.equal(candidate.state_dict()[name], expected), name


def test_batchtopk_exports_to_official_jumprelu_before_folding() -> None:
    model = BatchTopKSAE(BatchTopKSAEConfig(d_in=2, d_sae=3, k=1.5))
    with torch.no_grad():
        model.W_enc.zero_()
        model.b_enc.copy_(torch.tensor([2.0, 8.0, 12.0]))
        model.topk_threshold.fill_(0.5)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    inference = to_inference_sae(model, fold_decoder_norm=True)

    assert type(inference) is OfficialJumpReLUInferenceSAE
    assert type(inference.cfg) is OfficialJumpReLUInferenceSAEConfig
    assert torch.allclose(inference.W_dec.norm(dim=1), torch.ones(3))
    assert torch.allclose(inference.threshold, torch.full((3,), 0.5))
    one = inference.encode(torch.zeros(1, 2))
    many = inference.encode(torch.zeros(7, 2))
    assert torch.equal(one.expand_as(many), many)
    for name, expected in before.items():
        assert torch.equal(model.state_dict()[name], expected), name


def test_l1_exports_to_official_standard_and_preserves_function() -> None:
    model = L1ReLUSAE(L1SAEConfig(d_in=2, d_sae=3, l1_coefficient=0.2))
    x = torch.randn(5, 2)
    expected = model(x).detach()
    before = {name: value.clone() for name, value in model.state_dict().items()}

    inference = to_inference_sae(model, fold_decoder_norm=True)

    assert type(inference) is OfficialL1InferenceSAE
    assert type(inference.cfg) is OfficialL1InferenceSAEConfig
    assert torch.allclose(inference.W_dec.norm(dim=1), torch.ones(3))
    assert torch.allclose(inference(x), expected)
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value), name


@pytest.mark.parametrize(
    ("alias", "expected", "kwargs"),
    [
        ("l1-relu", OfficialL1SAE, {}),
        ("top-k", OfficialTopKSAE, {"k": 2}),
        ("batch_topk", OfficialBatchTopKSAE, {"k": 2.0}),
        ("jump-relu", OfficialJumpReLUSAE, {}),
        ("gated-sae", OfficialGatedSAE, {}),
    ],
)
def test_builder_is_the_only_legacy_dimension_seam(
    alias: str,
    expected: type[TrainingSAE],
    kwargs: dict[str, float | int],
) -> None:
    model = build_sae(alias, input_dim=3, n_latents=6, **kwargs)
    assert type(model) is expected
    assert model.cfg.d_in == 3
    assert model.cfg.d_sae == 6


def test_l1_factory_uses_official_standard_defaults() -> None:
    model = build_sae("l1", input_dim=3, n_latents=6)

    assert type(model) is OfficialL1SAE
    assert model.cfg.l1_coefficient == 1.0
    assert model.cfg.decoder_init_norm == 0.1
