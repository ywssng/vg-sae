import json
import math

import pytest
import torch

sae_lens = pytest.importorskip("sae_lens")
if sae_lens.__version__ != "6.47.0":
    pytest.skip("VG SAELens adapter is pinned to sae-lens 6.47.0", allow_module_level=True)

from sae_lens import SAE, TrainingSAE
from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.sae import TrainStepInput
from sae_lens.synthetic import (
    ConstantFiringProbabilityConfig,
    SyntheticActivationIterator,
    SyntheticModel,
    SyntheticModelConfig,
    eval_sae_on_synthetic_data,
)
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.sae_trainer import SAETrainer

from src.sae_model import VGSAEConfig as CoreVGSAEConfig
from src.sae_model import VariationalGarroteSAE
from src.sae_train import fit_sae
from src.saelens_vg import (
    VGSAE,
    VGSAEConfig,
    VGSAETrainer,
    VGTrainingSAE,
    VGTrainingSAEConfig,
    register_vg_saes,
)


def _step_input(x: torch.Tensor, gamma: float = 0.3) -> TrainStepInput:
    return TrainStepInput(x, {"lambda_sparsity": gamma}, None, 0, False)


def test_registration_is_idempotent_and_constructs_both_classes() -> None:
    register_vg_saes()
    inference = SAE.from_dict(VGSAEConfig(d_in=3, d_sae=5).to_dict())
    training = TrainingSAE.from_dict(VGTrainingSAEConfig(d_in=3, d_sae=5).to_dict())
    assert isinstance(inference, VGSAE)
    assert isinstance(training, VGTrainingSAE)


@pytest.mark.parametrize(
    "model_class,config",
    [
        (VGSAE, VGSAEConfig(d_in=3, d_sae=5)),
        (VGTrainingSAE, VGTrainingSAEConfig(d_in=3, d_sae=5)),
    ],
)
def test_error_term_is_rejected(model_class, config) -> None:
    with pytest.raises(ValueError, match="use_error_term"):
        model_class(config, use_error_term=True)


@pytest.mark.parametrize("warmup", [-1, 1.5, True])
def test_config_rejects_unsupported_normalization_and_invalid_warmup(warmup) -> None:
    with pytest.raises(ValueError, match="supports"):
        VGSAEConfig(
            d_in=3,
            d_sae=5,
            normalize_activations="constant_norm_rescale",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        VGTrainingSAEConfig(d_in=3, d_sae=5, lambda_warm_up_steps=warmup)


@pytest.mark.parametrize("beta_mode", ["profiled", "fixed", "learned"])
def test_training_forward_delegates_free_energy_and_beta_modes(beta_mode: str) -> None:
    cfg = VGTrainingSAEConfig(
        d_in=3,
        d_sae=4,
        beta=1.7,
        beta_mode=beta_mode,  # type: ignore[arg-type]
        lambda_sparsity=0.3,
    )
    model = VGTrainingSAE(cfg)
    reference = VariationalGarroteSAE(
        CoreVGSAEConfig(  # type: ignore[arg-type]
            3, 4, beta=1.7, beta_mode=beta_mode, lambda_sparsity=0.3
        )
    )
    reference.load_state_dict(model.core.state_dict())
    x = torch.randn(6, 3)
    actual = model.training_forward_pass(_step_input(x))
    expected = reference.free_energy(x, lambda_sparsity=0.3)
    assert torch.allclose(actual.loss, expected["loss"])
    assert torch.allclose(actual.sae_out, expected["x_hat"])
    assert torch.allclose(actual.feature_acts, model.encode(x))
    assert set(actual.losses) == {
        "free_energy",
        "expected_reconstruction_energy",
        "posterior_variance_energy",
        "posterior_prior_energy",
        "posterior_negative_entropy_energy",
    }
    assert set(actual.metrics) == {
        "posterior_expected_l0",
        "hard_support_l0",
        "posterior_mean_probability",
        "posterior_bernoulli_variance",
        "beta_precision",
        "decoder_norm_error",
    }
    actual.loss.backward()
    decoder_grad = model.core.decoder.weight.grad
    assert decoder_grad is not None
    assert torch.allclose(
        (decoder_grad * model.core.decoder.weight).sum(dim=0),
        torch.zeros(model.cfg.d_sae),
        atol=1.0e-6,
    )
    if beta_mode == "learned":
        assert model.core.log_beta is not None
        assert model.core.log_beta.grad is not None


def test_public_encode_is_hard_but_training_reconstruction_is_expected() -> None:
    model = VGTrainingSAE(VGTrainingSAEConfig(d_in=2, d_sae=2))
    with torch.no_grad():
        model.core.gate_encoder.weight.zero_()
        model.core.gate_encoder.bias.copy_(torch.tensor([-1.0, 1.0]))
        model.core.amplitude_encoder.weight.zero_()
        model.core.amplitude_encoder.bias.zero_()
    x = torch.zeros(3, 2)
    posterior = model.posterior(x)
    public_code = model.encode(x)
    step = model.training_forward_pass(_step_input(x, model.cfg.lambda_sparsity))
    assert torch.equal(model.support_mask(x), torch.tensor([[False, True]]).expand(3, 2))
    assert torch.equal(public_code[:, 0], torch.zeros(3))
    assert torch.all(public_code[:, 1] > 0)
    assert torch.all(posterior["expected_code"] > 0)
    assert torch.equal(step.feature_acts, public_code)
    assert not torch.allclose(model.decode(step.feature_acts), step.sae_out)
    assert torch.equal(model.firing_mask(step), model.support_mask(x))


def test_vg_trainer_uses_hard_firing_and_enforces_unit_decoder() -> None:
    model = VGTrainingSAE(VGTrainingSAEConfig(d_in=2, d_sae=2))
    with torch.no_grad():
        model.core.gate_encoder.weight.zero_()
        model.core.gate_encoder.bias.copy_(torch.tensor([-10.0, 10.0]))
    cfg = SAETrainerConfig(
        total_training_samples=4,
        train_batch_size_samples=4,
        lr=1.0e-3,
        lr_end=1.0e-3,
        logger=LoggingConfig(log_to_wandb=False),
    )
    trainer = VGSAETrainer(cfg, model, iter([torch.randn(4, 2)]))
    output = trainer.step(torch.randn(4, 2))
    assert torch.equal(trainer.act_freq_scores, torch.tensor([0.0, 4.0]))
    assert torch.allclose(model.W_dec.norm(dim=-1), torch.ones(2), atol=1.0e-6)
    log = trainer.build_train_step_log_dict(output, 4)
    assert log["metrics/hard_support_l0"] == pytest.approx(1.0)
    assert "metrics/l0" not in log
    assert "metrics/explained_variance" not in log
    assert "metrics/expected_reconstruction_explained_variance" in log
    assert "metrics/expected_reconstruction_explained_variance_legacy" in log
    assert "metrics/expected_reconstruction_explained_variance_legacy_std" in log


def test_project_fit_dispatches_to_vg_trainer_and_records_free_energy() -> None:
    model = VGTrainingSAE(
        VGTrainingSAEConfig(d_in=2, d_sae=3, beta_mode="fixed")
    )
    result = fit_sae(
        model,
        torch.randn(8, 2),
        max_steps=2,
        batch_size=4,
        history_every=1,
    )

    assert result.model is model
    assert torch.allclose(model.W_dec.norm(dim=-1), torch.ones(3), atol=1.0e-6)
    assert {
        "free_energy",
        "expected_reconstruction_energy",
        "posterior_expected_l0",
        "hard_support_l0",
    } <= result.history[-1].keys()


def test_base_trainer_uses_hard_firing_and_renormalizes_on_next_forward() -> None:
    model = VGTrainingSAE(VGTrainingSAEConfig(d_in=2, d_sae=2))
    with torch.no_grad():
        model.core.gate_encoder.weight.zero_()
        model.core.gate_encoder.bias.copy_(torch.tensor([-10.0, 10.0]))
    cfg = SAETrainerConfig(
        total_training_samples=4,
        train_batch_size_samples=4,
        lr=1.0e-3,
        lr_end=1.0e-3,
        logger=LoggingConfig(log_to_wandb=False),
    )
    trainer = SAETrainer(cfg, model, iter([torch.randn(4, 2)]))
    trainer.step(torch.randn(4, 2))
    assert torch.equal(trainer.act_freq_scores, torch.tensor([0.0, 4.0]))

    model.encode(torch.randn(2, 2))
    assert torch.allclose(model.W_dec.norm(dim=-1), torch.ones(2), atol=1.0e-6)


@pytest.mark.parametrize("beta_mode", ["profiled", "fixed", "learned"])
def test_activation_norm_folding_preserves_raw_function(beta_mode: str) -> None:
    model = VGSAE(
        VGSAEConfig(
            d_in=3,
            d_sae=4,
            beta=1.5,
            beta_mode=beta_mode,  # type: ignore[arg-type]
            normalize_activations="expected_average_only_in",
        )
    )
    raw, scaling = torch.randn(5, 3), 2.5
    scaled_posterior = model.posterior(raw * scaling)
    scaled_reconstruction = model.decode(model.encode(raw * scaling))
    beta_before = (
        model.core.log_beta.exp().clone()
        if model.core.log_beta is not None
        else torch.tensor(model.cfg.beta)
    )

    model.fold_activation_norm_scaling_factor(scaling)

    assert torch.allclose(model.posterior(raw)["m"], scaled_posterior["m"])
    assert torch.allclose(model.posterior(raw)["a"], scaled_posterior["a"])
    assert torch.allclose(
        model.decode(model.encode(raw)), scaled_reconstruction / scaling
    )
    assert model.cfg.normalize_decoder is False
    assert model.core.config.normalize_decoder is False
    with pytest.raises(NotImplementedError, match="cannot be folded"):
        model.fold_W_dec_norm()
    if beta_mode != "profiled":
        beta_after = (
            model.core.log_beta.exp()
            if model.core.log_beta is not None
            else torch.tensor(model.cfg.beta)
        )
        assert torch.allclose(beta_after, beta_before * scaling**2)


def test_training_and_inference_save_load_boundary(tmp_path) -> None:
    model = VGTrainingSAE(
        VGTrainingSAEConfig(d_in=3, d_sae=4, beta_mode="learned")
    )
    x = torch.randn(5, 3)
    training_path = tmp_path / "training"
    inference_path = tmp_path / "inference"
    model.save_model(training_path)
    model.save_inference_model(inference_path)

    loaded_training = TrainingSAE.load_from_disk(training_path)
    loaded_inference = SAE.load_from_disk(inference_path)
    assert isinstance(loaded_training, VGTrainingSAE)
    assert isinstance(loaded_inference, VGSAE)
    assert loaded_training._decoder_grad_hook is not None
    assert torch.allclose(loaded_training.encode(x), model.encode(x))
    assert torch.allclose(loaded_inference.encode(x), model.encode(x))
    assert json.loads((inference_path / "cfg.json").read_text())["architecture"] == "vg"
    assert "lambda_warm_up_steps" not in json.loads(
        (inference_path / "cfg.json").read_text()
    )


def test_official_synthetic_eval_external_scaler_and_generic_export() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        synthetic = SyntheticModel(
            SyntheticModelConfig(
                num_features=3,
                hidden_dim=2,
                firing_probability=ConstantFiringProbabilityConfig(0.5),
                mean_firing_magnitudes=1.0,
                std_firing_magnitudes=0.0,
                bias=False,
                seed=None,
            )
        )
    model = VGTrainingSAE(
        VGTrainingSAEConfig(d_in=2, d_sae=3, beta_mode="fixed")
    )
    scaler = ActivationScaler(scaling_factor=1.7)

    torch.manual_seed(11)
    pre_fit = eval_sae_on_synthetic_data(
        model,
        synthetic.feature_dict,
        synthetic.activation_generator,
        num_samples=8,
        batch_size=4,
        activation_scaler=scaler,
    )
    assert math.isfinite(pre_fit.explained_variance)

    provider = SyntheticActivationIterator(
        synthetic.feature_dict,
        synthetic.activation_generator,
        batch_size=4,
    )
    trainer = VGSAETrainer(
        SAETrainerConfig(
            total_training_samples=8,
            train_batch_size_samples=4,
            lr=1.0e-3,
            lr_end=1.0e-3,
            logger=LoggingConfig(log_to_wandb=False),
        ),
        model,
        provider,
    )
    trainer.activation_scaler.scaling_factor = scaler.scaling_factor
    torch.manual_seed(13)
    trained = trainer.fit()
    assert trainer.activation_scaler.scaling_factor is None

    state = {name: value.detach().clone() for name, value in trained.state_dict().items()}
    trained.process_state_dict_for_saving_inference(state)
    inference = SAE.from_dict(trained.cfg.get_inference_sae_cfg_dict())
    inference.load_state_dict(state)
    assert isinstance(inference, VGSAE)

    torch.manual_seed(17)
    result = eval_sae_on_synthetic_data(
        inference,
        synthetic.feature_dict,
        synthetic.activation_generator,
        num_samples=8,
        batch_size=4,
    )
    assert 0.0 <= result.sae_l0 <= inference.cfg.d_sae
    assert math.isfinite(result.mcc)
