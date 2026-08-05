import csv
import itertools
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
from src.sae_loss import bernoulli_kl_from_lambda, sae_loss_terms, vg_sae_loss_terms
from src.sae_model import (
    BatchTopKSAE,
    BatchTopKSAEConfig,
    GatedSAE,
    GatedSAEConfig,
    JumpReLUSAE,
    JumpReLUSAEConfig,
    StandardSAE,
    StandardSAEConfig,
    TopKSAE,
    TopKSAEConfig,
    VGSAEConfig,
    VariationalGarroteSAE,
)
from src.sae_train import fit_sae


def test_vg_sae_expected_reconstruction_variance_matches_enumeration() -> None:
    model = VariationalGarroteSAE(
        VGSAEConfig(input_dim=2, n_latents=2, beta=1.0, lambda_sparsity=0.0, use_entropy_term=False)
    )
    x = torch.tensor([[0.3, -0.7], [1.0, 0.2]])
    with torch.no_grad():
        model.decoder.weight.copy_(torch.tensor([[1.0, 0.2], [0.0, 0.8]]))
        if model.pre_bias is not None:
            model.pre_bias.zero_()
        model.gate_encoder.weight.zero_()
        model.gate_encoder.bias.copy_(torch.tensor([-0.4, 0.7]))
        model.amplitude_encoder.weight.zero_()
        model.amplitude_encoder.bias.copy_(torch.tensor([0.5, -0.2]))

    output = model(x)
    m = output["m"]
    a = output["a"]
    formula = (x - output["x_hat"]).pow(2).sum(dim=1)
    formula = formula + (m * (1.0 - m) * a.pow(2) * model.decoder_column_sqnorms()).sum(dim=1)

    brute_force = []
    for b in range(x.shape[0]):
        expected = 0.0
        for bits in itertools.product([0.0, 1.0], repeat=2):
            s = torch.tensor(bits)
            prob = torch.prod(torch.where(s > 0.0, m[b], 1.0 - m[b]))
            recon = model.decode(s * a[b])
            expected = expected + prob * (x[b] - recon).pow(2).sum()
        brute_force.append(expected)
    brute_force_tensor = torch.stack(brute_force)

    assert torch.allclose(formula, brute_force_tensor, atol=1.0e-6)


def test_bernoulli_kl_lambda_sign_penalizes_dense_support() -> None:
    sparse_m = torch.tensor([0.1])
    dense_m = torch.tensor([0.9])

    assert bernoulli_kl_from_lambda(dense_m, lambda_sparsity=2.0) > bernoulli_kl_from_lambda(
        sparse_m, lambda_sparsity=2.0
    )


def test_vg_sae_decoder_columns_are_unit_norm() -> None:
    model = VariationalGarroteSAE(VGSAEConfig(input_dim=4, n_latents=6))

    assert torch.allclose(model.decoder_column_sqnorms(), torch.ones(6), atol=1.0e-6)


def test_vg_sae_loss_terms_wrap_free_energy_loss() -> None:
    model = VariationalGarroteSAE(
        VGSAEConfig(input_dim=4, n_latents=6, beta=2.0, lambda_sparsity=0.3)
    )
    x = torch.randn(5, 4)

    terms = vg_sae_loss_terms(model, x)
    output = model.free_energy(x)

    assert torch.allclose(terms.loss, output["loss"])
    assert torch.allclose(terms.reconstruction_mse, 2.0 * output["recon"] / model.config.input_dim)


def test_vg_sae_accepts_signed_finite_sparsity_field() -> None:
    VariationalGarroteSAE(VGSAEConfig(input_dim=4, n_latents=6, lambda_sparsity=-0.1))
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            VariationalGarroteSAE(VGSAEConfig(input_dim=4, n_latents=6, lambda_sparsity=value))


def test_vg_sae_accepts_dtype_string() -> None:
    model = VariationalGarroteSAE(VGSAEConfig(input_dim=4, n_latents=6, dtype="float64"))

    assert model.decoder.weight.dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_bernoulli_kl_is_finite_and_stable_at_endpoints(dtype: torch.dtype) -> None:
    gamma = -0.7
    m = torch.tensor([0.0, 1.0], dtype=dtype, requires_grad=True)
    actual = bernoulli_kl_from_lambda(m, gamma)
    expected = torch.stack(
        [
            F.softplus(torch.tensor(-gamma, dtype=dtype)),
            F.softplus(torch.tensor(gamma, dtype=dtype)),
        ]
    )
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=1.0e-2, rtol=1.0e-2)
    actual.sum().backward()
    assert torch.isfinite(m.grad).all()


def test_vg_sae_trace_beta_counts_vector_observations() -> None:
    model = VariationalGarroteSAE(
        VGSAEConfig(
            input_dim=3,
            n_latents=2,
            lambda_sparsity=0.0,
            use_entropy_term=False,
            use_variance_term=False,
        )
    )
    x = torch.ones(4, 3)
    with torch.no_grad():
        model.decoder.weight.zero_()
        if model.pre_bias is not None:
            model.pre_bias.zero_()

    output = model.free_energy(x)
    energy_sum = 0.5 * x.pow(2).sum()
    n_scalar_observations = x.numel()
    gaussian = 0.5 * n_scalar_observations * torch.log(2.0 * energy_sum / n_scalar_observations)
    expected = gaussian / x.shape[0] + model.config.n_latents * torch.log(torch.tensor(2.0))

    assert torch.allclose(output["loss"], expected)


def test_topk_sae_has_exact_active_count_when_pre_topk_activations_are_positive() -> None:
    model = TopKSAE(TopKSAEConfig(d_in=3, d_sae=5, k=2))
    with torch.no_grad():
        model.W_enc.zero_()
        model.b_enc.fill_(1.0)
    h = model.encode(torch.randn(7, 3))

    assert torch.all((h > 0.0).sum(dim=1) == 2)


def test_topk_sae_reports_actual_activation_density() -> None:
    model = TopKSAE(TopKSAEConfig(d_in=3, d_sae=5, k=2))
    with torch.no_grad():
        model.W_enc.zero_()
        model.b_enc.zero_()
    terms = sae_loss_terms(model, torch.randn(7, 3))

    assert torch.allclose(terms.rho, torch.tensor(0.0))


def test_synthetic_sparse_coding_shapes_and_sparsity() -> None:
    data = make_synthetic_sparse_coding(
        SyntheticSparseCodingConfig(
            input_dim=5, n_features=11, n_samples=2000, support_density=0.1, seed=2
        )
    )

    assert data.x.shape == (2000, 5)
    assert data.z.shape == (2000, 11)
    assert data.support.shape == (2000, 11)
    assert data.dictionary.shape == (5, 11)
    assert np.isclose(float(data.support.mean()), 0.1, atol=0.02)


def test_one_step_training_is_finite_for_vg_sae_and_baselines() -> None:
    data = make_synthetic_sparse_coding(
        SyntheticSparseCodingConfig(
            input_dim=4, n_features=8, n_samples=16, support_density=0.2, seed=3
        )
    )
    models = [
        VariationalGarroteSAE(VGSAEConfig(input_dim=4, n_latents=8, lambda_sparsity=0.5)),
        StandardSAE(StandardSAEConfig(d_in=4, d_sae=8)),
        TopKSAE(TopKSAEConfig(d_in=4, d_sae=8, k=2)),
        BatchTopKSAE(BatchTopKSAEConfig(d_in=4, d_sae=8, k=2)),
        JumpReLUSAE(JumpReLUSAEConfig(d_in=4, d_sae=8)),
        GatedSAE(GatedSAEConfig(d_in=4, d_sae=8)),
    ]

    for model in models:
        result = fit_sae(model, data.x, max_steps=1, batch_size=8, history_every=1)
        assert np.isfinite(result.history[-1]["loss"])
        assert torch.isfinite(sae_loss_terms(model, data.x).loss)


def test_synthetic_sweep_writes_csv_and_figure(tmp_path) -> None:
    output_dir = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_synthetic_sweep.py",
            "--output-dir",
            str(output_dir),
            "--input-dim",
            "4",
            "--widths",
            "8",
            "--n-samples",
            "32",
            "--lambdas",
            "0.0,1.0",
            "--steps",
            "2",
            "--batch-size",
            "16",
            "--beta",
            "1.7",
            "--beta-mode",
            "learned",
            "--include-no-variance",
            "--include-baselines",
        ],
        check=True,
    )

    csv_path = output_dir / "synthetic_sweep.csv"
    fig_path = output_dir / "v_eff_vs_lambda.png"
    assert csv_path.exists()
    assert fig_path.exists()
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(np.isfinite(float(row["mse"])) for row in rows)
    assert {row["beta_mode"] for row in rows} == {"learned"}
    assert {float(row["beta"]) for row in rows} == {1.7}
    assert {row["model"] for row in rows} == {
        "vg",
        "vg_no_variance",
        "l1",
        "topk",
        "batchtopk",
        "jumprelu",
        "gated",
    }
