import numpy as np
import torch

from src.data import (
    SyntheticConfig,
    make_synthetic_regression,
    make_synthetic_train_test,
    sample_spike_and_slab,
)
from src.evaluate import (
    generalization_error_numpy,
    infer_data_sparsity,
    selection_error,
    selection_uncertainty,
    theoretical_sigma_sel,
)
from src.loss import energy_term, vg_loss_terms
from src.model import VGConfig, VariationalGarrote
from src.train import train_vg


def _logit(values: torch.Tensor) -> torch.Tensor:
    return torch.log(values / (1.0 - values))


def test_energy_term_matches_paper_equation() -> None:
    m = torch.tensor([0.25, 0.75])
    w = torch.tensor([2.0, -1.0])
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([1.0, -2.0])

    assert torch.allclose(energy_term(m, w, x, y), torch.tensor(7.75))


def test_free_energy_uses_prior_corrected_gamma_sign() -> None:
    config = VGConfig(n_features=2, gamma=0.7, mask_init=0.5, loss_eps=1e-12)
    model = VariationalGarrote(config)
    with torch.no_grad():
        model.mask_logits.copy_(_logit(torch.tensor([0.25, 0.75])))
        model.weight.copy_(torch.tensor([2.0, -1.0]))

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([1.0, -2.0])
    terms = vg_loss_terms(model, x, y)
    m = torch.tensor([0.25, 0.75])
    entropy = (-m * torch.log(m) - (1.0 - m) * torch.log(1.0 - m)).sum()
    expected = torch.log(torch.tensor(7.75)) - entropy + 0.7 * m.sum()

    assert torch.allclose(terms.free_energy, expected)
    assert terms.sparsity_penalty.item() > 0.0


def test_positive_gamma_penalizes_larger_masks() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([1.0, -2.0])
    small = VariationalGarrote(VGConfig(n_features=2, gamma=0.7, mask_init=0.5))
    large = VariationalGarrote(VGConfig(n_features=2, gamma=0.7, mask_init=0.5))
    with torch.no_grad():
        small.mask_logits.copy_(_logit(torch.tensor([0.25, 0.25])))
        large.mask_logits.copy_(_logit(torch.tensor([0.75, 0.75])))
        small.weight.zero_()
        large.weight.zero_()

    assert vg_loss_terms(large, x, y).free_energy > vg_loss_terms(small, x, y).free_energy


def test_sigmoid_mask_init_has_nonzero_gradient() -> None:
    model = VariationalGarrote(VGConfig(n_features=4, mask_init=0.999))
    loss = model.mask().sum()
    loss.backward()

    assert torch.all(model.mask_logits.grad > 0.0)


def test_spike_and_slab_active_weights_have_exact_count_and_paper_support() -> None:
    rng = np.random.default_rng(0)
    n_features = 256
    rho = 8 / 256
    weights = sample_spike_and_slab(n_features=n_features, rho=rho, rng=rng)
    active = weights[weights != 0.0]
    w_bar = np.sqrt(12.0 / rho - 0.75) - 0.5

    assert len(active) == 8
    assert np.count_nonzero(weights == 0.0) == n_features - 8
    assert np.all(np.abs(active) > 1.0)
    assert np.all(np.abs(active) < w_bar)


def test_synthetic_dataset_adds_snr_calibrated_noise() -> None:
    data = make_synthetic_regression(
        SyntheticConfig(n_features=32, n_samples=64, rho_data=4 / 32, snr=3.0, seed=2)
    )
    noise = data.y - data.clean_y

    assert data.noise_std.item() > 0.0
    assert noise.abs().mean().item() > 0.0


def test_train_test_share_teacher_but_use_different_samples() -> None:
    config = SyntheticConfig(n_features=32, n_samples=64, rho_data=4 / 32, seed=3)
    train, test = make_synthetic_train_test(config, n_test=64, test_seed_offset=10)

    assert torch.allclose(train.teacher_weights, test.teacher_weights)
    assert not torch.allclose(train.x, test.x)


def test_generalization_error_is_root_relative_error() -> None:
    w = np.array([1.0, 0.0])
    m = np.array([1.0, 1.0])
    x = np.array([[1.0, 0.0], [2.0, 0.0]])
    y = np.array([2.0, 4.0])

    assert np.isclose(generalization_error_numpy(w, m, x, y), np.sqrt(5.0 / 20.0))


def test_selection_metrics_match_paper_equations() -> None:
    masks = np.array([
        [1.0, 0.5, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    s_true = np.array([1.0, 1.0, 0.0, 0.0])
    mean_mask = masks.mean(axis=0)

    expected_error = np.mean(s_true * (1.0 - masks) + (1.0 - s_true) * masks)
    expected_uncertainty = np.mean(mean_mask * (1.0 - mean_mask))

    assert np.isclose(selection_error(masks, s_true), expected_error)
    assert np.isclose(selection_uncertainty(masks), expected_uncertainty)


def test_theoretical_sigma_sel_matches_paper_kernel() -> None:
    assert np.isclose(theoretical_sigma_sel(0.1, 0.25), 0.1 / 0.25 * (0.25 - 0.1))
    assert np.isclose(theoretical_sigma_sel(0.4, 0.25), (0.4 - 0.25) * (1.0 - 0.4) / (1.0 - 0.25))


def test_infer_data_sparsity_returns_normalized_posterior() -> None:
    rho_model_values = np.array([0.05, 0.15, 0.25, 0.35])
    candidate_rhos = np.array([0.1, 0.25, 0.4])
    sigma = np.array([theoretical_sigma_sel(rho, 0.25) for rho in rho_model_values])

    candidates, posterior = infer_data_sparsity(rho_model_values, sigma, candidate_rhos)

    assert np.allclose(candidates, candidate_rhos)
    assert np.isclose(posterior.sum(), 1.0)
    assert candidates[np.argmax(posterior)] == 0.25


def test_tiny_training_smoke_run_is_finite() -> None:
    data = make_synthetic_regression(
        SyntheticConfig(n_features=8, n_samples=16, rho_data=2 / 8, seed=4)
    )
    result = train_vg(
        data.x,
        data.y,
        VGConfig(n_features=8, gamma=0.5, mask_init=0.999),
        lr=0.01,
        max_steps=3,
        history_every=1,
    )

    assert len(result.history.steps) == 3
    assert np.isfinite(result.history.free_energy).all()
