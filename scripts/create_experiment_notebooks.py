"""Create experiment notebooks for the VG-SAE research proposal."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def _lines(source: str) -> list[str]:
    text = dedent(source).strip() + "\n"
    return text.splitlines(keepends=True)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (vg-sae)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = r"""
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
DEVICE = torch.device("cpu")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "notebooks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
"""


def non_amortized_notebook() -> dict:
    return notebook(
        [
            md(
                """
                # Experiment 1: Non-Amortized VG Support-Inference Sanity Check

                Proposal target: optimize the variational support probabilities `m_{bi}` directly per sample before trusting an amortized encoder. This isolates the VG free energy from encoder amortization noise.

                This notebook fixes the decoder to the ground-truth synthetic dictionary and optimizes per-sample gate logits and amplitudes:

                `x_b = D* (m_b * a_b) + noise`

                The diagnostic is whether `V_eff(lambda)=mean[m(1-m)]`, support precision/recall, and MSE behave sensibly under a lambda sweep.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
                from src.sae_loss import bernoulli_kl_from_lambda

                cfg = SyntheticSparseCodingConfig(
                    input_dim=8,
                    n_features=16,
                    n_samples=128,
                    support_density=0.125,
                    coherence=0.1,
                    noise_std=0.02,
                    seed=0,
                )
                data = make_synthetic_sparse_coding(cfg, device=DEVICE)
                D = data.dictionary
                print(data.x.shape, D.shape, float(data.support.mean()))
                """
            ),
            code(
                """
                def optimize_non_amortized(x, dictionary, lambda_sparsity, beta=1.0, steps=600, lr=5e-2, use_variance_term=True):
                    B = x.shape[0]
                    N = dictionary.shape[1]
                    logits = torch.zeros(B, N, requires_grad=True)
                    raw_a = torch.zeros(B, N, requires_grad=True)
                    optimizer = torch.optim.AdamW([logits, raw_a], lr=lr)
                    history = []
                    decoder_norm_square = dictionary.norm(dim=0).pow(2)
                    for step in range(steps):
                        optimizer.zero_grad(set_to_none=True)
                        m = torch.sigmoid(logits)
                        a = torch.nn.functional.softplus(raw_a)
                        h = m * a
                        x_hat = h @ dictionary.T
                        residual = (x - x_hat).pow(2).sum(dim=1)
                        variance = (m * (1.0 - m) * a.pow(2) * decoder_norm_square).sum(dim=1)
                        if not use_variance_term:
                            variance = torch.zeros_like(variance)
                        kl = bernoulli_kl_from_lambda(m, lambda_sparsity).sum(dim=1)
                        loss = (0.5 * beta * (residual + variance) + kl).mean()
                        loss.backward()
                        optimizer.step()
                        if step % 100 == 0 or step == steps - 1:
                            history.append({
                                "step": step,
                                "loss": float(loss.detach()),
                                "mse": float((x - x_hat).pow(2).mean().detach()),
                                "rho": float(m.mean().detach()),
                                "v_eff": float((m * (1.0 - m)).mean().detach()),
                            })
                    with torch.no_grad():
                        m = torch.sigmoid(logits)
                        a = torch.nn.functional.softplus(raw_a)
                        x_hat = (m * a) @ dictionary.T
                    return {"m": m.detach(), "a": a.detach(), "x_hat": x_hat.detach(), "history": history}
                """
            ),
            code(
                """
                lambdas = [0.0, 0.5, 1.0, 2.0, 3.0]
                rows = []
                for lam in lambdas:
                    result = optimize_non_amortized(data.x, D, lam, steps=400)
                    m = result["m"]
                    pred = m > 0.5
                    target = data.support > 0.5
                    tp = torch.logical_and(pred, target).sum().item()
                    fp = torch.logical_and(pred, ~target.bool()).sum().item()
                    fn = torch.logical_and(~pred, target.bool()).sum().item()
                    rows.append({
                        "lambda": lam,
                        "mse": float((data.x - result["x_hat"]).pow(2).mean()),
                        "rho": float(m.mean()),
                        "v_eff": float((m * (1.0 - m)).mean()),
                        "support_precision": tp / max(tp + fp, 1),
                        "support_recall": tp / max(tp + fn, 1),
                    })
                df = pd.DataFrame(rows)
                df
                """
            ),
            code(
                """
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
                axes[0].plot(df["lambda"], df["v_eff"], marker="o")
                axes[0].set_title("Gate variance")
                axes[0].set_xlabel("lambda")
                axes[0].set_ylabel("V_eff")
                axes[1].plot(df["lambda"], df["rho"], marker="o")
                axes[1].axhline(cfg.support_density, color="black", linestyle="--", linewidth=1)
                axes[1].set_title("Active density")
                axes[1].set_xlabel("lambda")
                axes[2].plot(df["lambda"], df["support_precision"], marker="o", label="precision")
                axes[2].plot(df["lambda"], df["support_recall"], marker="o", label="recall")
                axes[2].set_title("Support recovery")
                axes[2].set_xlabel("lambda")
                axes[2].legend()
                fig.tight_layout()
                out = OUTPUT_DIR / "exp01_non_amortized"
                out.mkdir(parents=True, exist_ok=True)
                df.to_csv(out / "non_amortized_lambda_sweep.csv", index=False)
                fig.savefig(out / "non_amortized_lambda_sweep.png", dpi=160)
                """
            ),
            md(
                """
                **Read this ruthlessly:** this experiment is not an SAE result yet. It only verifies that the VG support-inference objective has a measurable support-uncertainty response when amortization is removed. If this plot is flat or support recall collapses for all lambda, the amortized VG-SAE experiments are premature.
                """
            ),
        ]
    )


def synthetic_sparse_coding_notebook() -> dict:
    return notebook(
        [
            md(
                """
                # Experiment 2: Synthetic Sparse-Coding Transition and Baselines

                Proposal target: train VG-SAE on `x = D* z* + noise`, sweep `lambda`, compare against L1-ReLU, TopK, a deterministic sigmoid-gated SAE, and ablate the VG variance term.

                Logged quantities: MSE, active density, gate entropy, `V_eff`, susceptibility, interference, dead latents, decoder recovery cosine, support precision/recall, and amplitude shrinkage.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
                from src.sae_evaluate import (
                    amplitude_shrinkage,
                    decoder_recovery_cosine,
                    support_precision_recall,
                    susceptibility,
                    vg_sae_observables,
                )
                from src.sae_model import (
                    GatedSAE, GatedSAEConfig,
                    L1ReLUSAE, L1SAEConfig,
                    TopKSAE, TopKSAEConfig,
                    VGSAEConfig, VariationalGarroteSAE,
                )
                from src.sae_train import fit_sae

                input_dim = 16
                widths = [32, 64]
                lambdas = [0.0, 0.5, 1.0, 2.0, 3.0]
                train_steps = 250
                batch_size = 128
                data_cfg = dict(
                    input_dim=input_dim,
                    n_samples=512,
                    support_density=0.06,
                    coherence=0.1,
                    noise_std=0.03,
                    frequency_skew=0.0,
                    seed=1,
                )
                """
            ),
            code(
                """
                def run_vg(data, width, lambda_sparsity, use_variance_term=True, label="vg"):
                    model = VariationalGarroteSAE(
                        VGSAEConfig(
                            input_dim=data.x.shape[1],
                            n_latents=width,
                            lambda_sparsity=lambda_sparsity,
                            beta=1.0,
                            use_variance_term=use_variance_term,
                        )
                    )
                    fit_sae(model, data.x, max_steps=train_steps, batch_size=batch_size, lr=3e-3, history_every=100, seed=0)
                    obs = vg_sae_observables(model, data.x)
                    precision, recall = support_precision_recall(model, data.x, data.support, data.dictionary)
                    return {
                        "model": label,
                        "width": width,
                        "lambda": lambda_sparsity,
                        "mse": obs.mse,
                        "rho": obs.rho,
                        "entropy": obs.entropy,
                        "v_eff": obs.v_eff,
                        "dead_fraction": obs.dead_fraction,
                        "interference_energy": obs.interference_energy,
                        "variance_energy": obs.variance_energy,
                        "decoder_recovery_cosine": decoder_recovery_cosine(model.decoder.weight, data.dictionary),
                        "support_precision": precision,
                        "support_recall": recall,
                        "amplitude_shrinkage": amplitude_shrinkage(model, data.x, data.z, data.dictionary),
                    }

                def run_baselines(data, width):
                    k = max(1, round(data.support.mean().item() * width))
                    baselines = [
                        ("l1", L1ReLUSAE(L1SAEConfig(input_dim=data.x.shape[1], n_latents=width, l1_coefficient=1e-3))),
                        ("topk", TopKSAE(TopKSAEConfig(input_dim=data.x.shape[1], n_latents=width, k=k))),
                        ("gated", GatedSAE(GatedSAEConfig(input_dim=data.x.shape[1], n_latents=width, l1_coefficient=1e-3))),
                    ]
                    rows = []
                    for name, model in baselines:
                        result = fit_sae(model, data.x, max_steps=train_steps, batch_size=batch_size, lr=3e-3, history_every=100, seed=0)
                        final = result.history[-1]
                        rows.append({
                            "model": name,
                            "width": width,
                            "lambda": np.nan,
                            "mse": final["reconstruction_mse"],
                            "rho": final["rho"],
                            "entropy": np.nan,
                            "v_eff": np.nan,
                            "dead_fraction": np.nan,
                            "interference_energy": np.nan,
                            "variance_energy": np.nan,
                            "decoder_recovery_cosine": decoder_recovery_cosine(model.decoder.weight, data.dictionary),
                            "support_precision": np.nan,
                            "support_recall": np.nan,
                            "amplitude_shrinkage": np.nan,
                        })
                    return rows
                """
            ),
            code(
                """
                rows = []
                for width in widths:
                    data = make_synthetic_sparse_coding(SyntheticSparseCodingConfig(n_features=width, **data_cfg), device=DEVICE)
                    for lam in lambdas:
                        rows.append(run_vg(data, width, lam, use_variance_term=True, label="vg"))
                        rows.append(run_vg(data, width, lam, use_variance_term=False, label="vg_no_variance"))
                    rows.extend(run_baselines(data, width))

                df = pd.DataFrame(rows)
                for model_name in ["vg", "vg_no_variance"]:
                    for width in widths:
                        mask = (df["model"] == model_name) & (df["width"] == width)
                        df.loc[mask, "susceptibility"] = susceptibility(
                            df.loc[mask, "lambda"].to_numpy(dtype=float),
                            df.loc[mask, "rho"].to_numpy(dtype=float),
                        )
                df.head()
                """
            ),
            code(
                """
                out = OUTPUT_DIR / "exp02_synthetic_sparse_coding"
                out.mkdir(parents=True, exist_ok=True)
                df.to_csv(out / "synthetic_sparse_coding_sweep.csv", index=False)

                fig, axes = plt.subplots(2, 2, figsize=(11, 8))
                for model_name, style in [("vg", "-o"), ("vg_no_variance", "--o")]:
                    for width in widths:
                        sub = df[(df["model"] == model_name) & (df["width"] == width)].sort_values("lambda")
                        axes[0, 0].plot(sub["lambda"], sub["v_eff"], style, label=f"{model_name}, N={width}")
                        axes[0, 1].plot(sub["lambda"], sub["rho"], style, label=f"{model_name}, N={width}")
                        axes[1, 0].plot(sub["lambda"], sub["support_recall"], style, label=f"{model_name}, N={width}")
                        axes[1, 1].plot(sub["lambda"], sub["decoder_recovery_cosine"], style, label=f"{model_name}, N={width}")
                axes[0, 0].set_title("V_eff transition")
                axes[0, 1].set_title("Active density")
                axes[1, 0].set_title("Support recall")
                axes[1, 1].set_title("Decoder recovery")
                for ax in axes.ravel():
                    ax.set_xlabel("lambda")
                    ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(out / "synthetic_sparse_coding_summary.png", dpi=160)
                df
                """
            ),
            md(
                """
                **Interpretation rule:** the variance-term ablation is the novelty defense. If `vg_no_variance` matches VG on uncertainty calibration and support recovery, the proposal’s central claim is weak and needs revision.
                """
            ),
        ]
    )


def synthetic_rho_model_comparison_notebook() -> dict:
    return notebook(
        [
            md(
                """
                # Experiment 7: Rho-Model Comparison on Synthetic Sparse Coding

                Goal: compare VG-SAE, L1-ReLU SAE, TopK SAE, and a deterministic sigmoid-gated SAE on the same synthetic sparse-coding task, using measured `rho_model` as the x-axis. This follows the paper's comparison logic: sweep the sparsity field/control, compute mask-density values, then plot latent-vector generalization error, reconstruction error, selection error, and mask uncertainty against `rho_model`.

                Definitions used here:

                - VG-SAE mask values are `m = sigmoid(gate_encoder(x))`.
                - Gated SAE mask values are its deterministic sigmoid gates.
                - TopK mask values are exactly the selected TopK indices from the model definition.
                - L1-ReLU primary mask values use a two-component GMM elbow on ReLU activation magnitudes, analogous to the paper's LASSO near-zero/broad-component thresholding. Raw ReLU support is logged separately.
                - `generalization_error` is the root-relative error between decoder-matched learned latents and the true sparse code `z`.
                - `reconstruction_error` is the held-out root-relative reconstruction error in input space.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                from types import SimpleNamespace

                import torch.nn.functional as F
                from scipy.optimize import linear_sum_assignment
                from sklearn.mixture import GaussianMixture
                from sklearn.metrics import average_precision_score, roc_auc_score

                from src.evaluate import selection_error, selection_uncertainty
                from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
                from src.sae_model import (
                    GatedSAE, GatedSAEConfig,
                    L1ReLUSAE, L1SAEConfig,
                    TopKSAE, TopKSAEConfig,
                    VGSAEConfig, VariationalGarroteSAE,
                )
                from src.sae_train import fit_sae
                from src.utils import set_seed
                """
            ),
            md(
                """
                ## Configuration

                `VGSAE_NOTEBOOK_FAST_DEV_RUN=1` shrinks the sweep for smoke execution. For publishable plots, increase `seeds`, `train_steps`, and the control grids.
                """
            ),
            code(
                """
                FAST_DEV_RUN = bool(int(os.environ.get("VGSAE_NOTEBOOK_FAST_DEV_RUN", "0")))

                input_dim = 16
                n_features = 32
                support_density = 0.06
                coherence = 0.1
                noise_std = 0.03
                frequency_skew = 0.0
                amplitude_scale = 1.0

                if FAST_DEV_RUN:
                    seeds = [0]
                    n_train = 96
                    n_test = 96
                    train_steps = 4
                    history_every = 1
                    gamma_values = [0.0, 1.0]
                    l1_coefficients = [1.0e-4, 1.0e-3]
                    gated_l1_coefficients = [1.0e-4, 1.0e-3]
                    topk_values = [1, 2]
                else:
                    seeds = [0]
                    n_train = 512
                    n_test = 1024
                    train_steps = 350
                    history_every = 25
                    gamma_values = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
                    l1_coefficients = [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2]
                    gated_l1_coefficients = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2]
                    topk_values = [1, 2, 3, 4, 6, 8, 12, 16]

                batch_size = 128
                lr = 3.0e-3
                beta = 1.0
                gradient_clip_norm = 1.0
                dead_threshold = 1.0e-6
                mask_threshold = 0.5

                topk_values = sorted({int(k) for k in topk_values if 0 < int(k) <= n_features})
                EXP_DIR = OUTPUT_DIR / "exp07_synthetic_sparse_coding_rho_model_comparison"
                EXP_DIR.mkdir(parents=True, exist_ok=True)
                """
            ),
            code(
                """
                def split_sparse_coding_data(data, n_train):
                    train = SimpleNamespace(
                        x=data.x[:n_train],
                        z=data.z[:n_train],
                        support=data.support[:n_train],
                        dictionary=data.dictionary,
                        clean_x=data.clean_x[:n_train],
                        feature_probabilities=data.feature_probabilities,
                    )
                    test = SimpleNamespace(
                        x=data.x[n_train:],
                        z=data.z[n_train:],
                        support=data.support[n_train:],
                        dictionary=data.dictionary,
                        clean_x=data.clean_x[n_train:],
                        feature_probabilities=data.feature_probabilities,
                    )
                    return train, test


                def make_train_test(seed):
                    data = make_synthetic_sparse_coding(
                        SyntheticSparseCodingConfig(
                            input_dim=input_dim,
                            n_features=n_features,
                            n_samples=n_train + n_test,
                            support_density=support_density,
                            coherence=coherence,
                            noise_std=noise_std,
                            frequency_skew=frequency_skew,
                            amplitude_scale=amplitude_scale,
                            seed=seed,
                        ),
                        device=DEVICE,
                    )
                    return split_sparse_coding_data(data, n_train)


                train0, test0 = make_train_test(seeds[0])
                print(
                    "train", tuple(train0.x.shape),
                    "test", tuple(test0.x.shape),
                    "empirical support density", float(train0.support.mean()),
                )
                """
            ),
            md(
                """
                ## Mask and Metric Helpers

                Learned latents are matched to true sparse-coding features by maximum absolute decoder cosine. Selection metrics are computed after this matching.
                """
            ),
            code(
                """
                METHOD_LABELS = {
                    "vgsae": "VG-SAE",
                    "l1": "L1-ReLU",
                    "topk": "TopK",
                    "gated": "Gated",
                }
                METHOD_COLORS = {
                    "vgsae": "tab:blue",
                    "l1": "tab:orange",
                    "topk": "tab:green",
                    "gated": "tab:red",
                }


                def build_specs():
                    specs = []
                    specs.extend(
                        {
                            "method": "vgsae",
                            "control_name": "gamma",
                            "control_value": float(gamma),
                        }
                        for gamma in gamma_values
                    )
                    specs.extend(
                        {
                            "method": "l1",
                            "control_name": "l1_coefficient",
                            "control_value": float(coef),
                        }
                        for coef in l1_coefficients
                    )
                    specs.extend(
                        {
                            "method": "topk",
                            "control_name": "k",
                            "control_value": int(k),
                        }
                        for k in topk_values
                    )
                    specs.extend(
                        {
                            "method": "gated",
                            "control_name": "l1_coefficient",
                            "control_value": float(coef),
                        }
                        for coef in gated_l1_coefficients
                    )
                    return specs


                def build_model(spec):
                    method = spec["method"]
                    value = spec["control_value"]
                    if method == "vgsae":
                        return VariationalGarroteSAE(
                            VGSAEConfig(
                                input_dim=input_dim,
                                n_latents=n_features,
                                lambda_sparsity=float(value),
                                beta=beta,
                            )
                        )
                    if method == "l1":
                        return L1ReLUSAE(
                            L1SAEConfig(
                                input_dim=input_dim,
                                n_latents=n_features,
                                l1_coefficient=float(value),
                            )
                        )
                    if method == "topk":
                        return TopKSAE(
                            TopKSAEConfig(
                                input_dim=input_dim,
                                n_latents=n_features,
                                k=int(value),
                            )
                        )
                    if method == "gated":
                        return GatedSAE(
                            GatedSAEConfig(
                                input_dim=input_dim,
                                n_latents=n_features,
                                l1_coefficient=float(value),
                            )
                        )
                    raise ValueError(f"Unknown method: {method}")


                def relative_reconstruction_error(x_hat, target, eps=1.0e-12):
                    numerator = (x_hat - target).pow(2).sum()
                    denominator = target.pow(2).sum().clamp_min(eps)
                    return float(torch.sqrt(numerator / denominator).detach().cpu())


                def relative_latent_error(h_hat, z_true, eps=1.0e-12):
                    h_hat = np.asarray(h_hat, dtype=np.float64)
                    z_true = np.asarray(z_true, dtype=np.float64)
                    numerator = np.sum((h_hat - z_true) ** 2)
                    denominator = max(float(np.sum(z_true ** 2)), eps)
                    return float(np.sqrt(numerator / denominator))


                def decoder_matching(model, true_dictionary):
                    learned = model.decoder.weight.detach().cpu()
                    true = true_dictionary.detach().cpu()
                    learned = learned / learned.norm(dim=0, keepdim=True).clamp_min(1.0e-12)
                    true = true / true.norm(dim=0, keepdim=True).clamp_min(1.0e-12)
                    signed_cosines = (learned.T @ true).numpy()
                    cosines = np.abs(signed_cosines)
                    learned_idx, true_idx = linear_sum_assignment(-cosines)
                    signs = np.sign(signed_cosines[learned_idx, true_idx]) if learned_idx.size else np.array([])
                    signs = np.where(signs == 0.0, 1.0, signs)
                    recovery = float(cosines[learned_idx, true_idx].mean()) if learned_idx.size else 0.0
                    return learned_idx, true_idx, signs, recovery


                def align_to_true_features(values, learned_idx, true_idx, n_true, signs=None):
                    if isinstance(values, torch.Tensor):
                        array = values.detach().cpu().numpy()
                    else:
                        array = np.asarray(values)
                    aligned = np.zeros((array.shape[0], n_true), dtype=np.float64)
                    selected = array[:, learned_idx]
                    if signs is not None:
                        selected = selected * np.asarray(signs, dtype=np.float64)[None, :]
                    aligned[:, true_idx] = selected
                    return aligned


                def l1_gmm_activation_threshold(h):
                    values = h.detach().cpu().numpy().reshape(-1, 1)
                    if values.size == 0 or np.nanmax(values) <= 0.0:
                        return np.inf
                    logged = np.log1p(values)
                    if float(np.nanmax(logged) - np.nanmin(logged)) < 1.0e-8:
                        return 0.0
                    try:
                        gmm = GaussianMixture(
                            n_components=2,
                            covariance_type="full",
                            means_init=np.array([[0.0], [float(np.percentile(logged, 95.0))]]),
                            random_state=0,
                        )
                        gmm.fit(logged)
                        means = np.sort(gmm.means_.reshape(-1))
                        return float(max(np.expm1(0.5 * (means[0] + means[-1])), 0.0))
                    except Exception:
                        positive = values[values[:, 0] > 0.0, 0]
                        return float(np.percentile(positive, 50.0)) if positive.size else np.inf


                @torch.no_grad()
                def latent_values_and_masks(model, x):
                    model.eval()
                    if isinstance(model, VariationalGarroteSAE):
                        output = model(x)
                        return output["h"], output["m"], {"mask_family": "bernoulli_probability"}
                    if isinstance(model, GatedSAE):
                        output = model(x)
                        return output["h"], output["gate"], {"mask_family": "sigmoid_gate"}
                    if isinstance(model, TopKSAE):
                        acts = F.relu(model.encoder(x))
                        _, indices = torch.topk(acts, k=model.config.k, dim=1)
                        mask = torch.zeros_like(acts)
                        mask.scatter_(1, indices, 1.0)
                        h = mask * acts
                        return h, mask, {"mask_family": "exact_topk"}
                    if isinstance(model, L1ReLUSAE):
                        h = model.encode(x)
                        threshold = l1_gmm_activation_threshold(h)
                        mask = (h > threshold).to(x.dtype)
                        return h, mask, {
                            "mask_family": "gmm_relu_activation",
                            "l1_gmm_threshold": threshold,
                            "l1_raw_relu_density": float((h > 0.0).to(x.dtype).mean().detach().cpu()),
                        }
                    raise TypeError(f"Unsupported model type: {type(model).__name__}")


                def binary_support_metrics(mask_values, true_support, threshold=0.5):
                    pred = mask_values >= threshold
                    target = true_support >= 0.5
                    tp = np.logical_and(pred, target).sum()
                    fp = np.logical_and(pred, np.logical_not(target)).sum()
                    fn = np.logical_and(np.logical_not(pred), target).sum()
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
                    return float(precision), float(recall), float(f1)


                def ranking_support_metrics(mask_values, true_support):
                    target = (true_support.reshape(-1) >= 0.5).astype(int)
                    scores = mask_values.reshape(-1)
                    if target.min() == target.max():
                        return np.nan, np.nan
                    average_precision = average_precision_score(target, scores)
                    roc_auc = roc_auc_score(target, scores)
                    return float(average_precision), float(roc_auc)


                @torch.no_grad()
                def evaluate_trained_model(model, train_data, test_data, spec, seed, run_id):
                    train_h, train_mask, train_mask_info = latent_values_and_masks(model, train_data.x)
                    test_h, test_mask, test_mask_info = latent_values_and_masks(model, test_data.x)
                    output = model(test_data.x)
                    learned_idx, true_idx, signs, decoder_recovery = decoder_matching(model, test_data.dictionary)

                    mask_aligned = align_to_true_features(test_mask, learned_idx, true_idx, test_data.support.shape[1])
                    h_aligned = align_to_true_features(test_h, learned_idx, true_idx, test_data.support.shape[1], signs=signs)
                    true_support = test_data.support.detach().cpu().numpy()
                    true_z = test_data.z.detach().cpu().numpy()

                    precision, recall, f1 = binary_support_metrics(mask_aligned, true_support, threshold=mask_threshold)
                    average_precision, roc_auc = ranking_support_metrics(mask_aligned, true_support)
                    active = true_z > 1.0e-8
                    amplitude_ratio = h_aligned[active] / np.maximum(true_z[active], 1.0e-8) if active.any() else np.array([])
                    reconstruction_error = relative_reconstruction_error(output["x_hat"], test_data.x)
                    clean_reconstruction_error = relative_reconstruction_error(output["x_hat"], test_data.clean_x)

                    row = {
                        "run_id": run_id,
                        "seed": seed,
                        "method": spec["method"],
                        "method_label": METHOD_LABELS[spec["method"]],
                        "control_name": spec["control_name"],
                        "control_value": spec["control_value"],
                        "rho_model": float(mask_aligned.mean()),
                        "generalization_error": relative_latent_error(h_aligned, true_z),
                        "reconstruction_error": reconstruction_error,
                        "clean_reconstruction_error": clean_reconstruction_error,
                        "reconstruction_mse": float((output["x_hat"] - test_data.x).pow(2).mean().detach().cpu()),
                        "selection_error": selection_error(mask_aligned, true_support),
                        "mask_uncertainty": float(np.mean(mask_aligned * (1.0 - mask_aligned))),
                        "paper_style_sigma_sel": selection_uncertainty(mask_aligned),
                        "support_precision": precision,
                        "support_recall": recall,
                        "support_f1": f1,
                        "support_average_precision": average_precision,
                        "support_roc_auc": roc_auc,
                        "decoder_recovery_cosine": decoder_recovery,
                        "dead_fraction": float((train_h.mean(dim=0) <= dead_threshold).to(torch.float32).mean().detach().cpu()),
                        "amplitude_shrinkage": float(amplitude_ratio.mean()) if amplitude_ratio.size else np.nan,
                        "mean_activation": float(test_h.mean().detach().cpu()),
                    }
                    row.update({k: v for k, v in train_mask_info.items() if not isinstance(v, str)})
                    row.update({k: v for k, v in test_mask_info.items() if not isinstance(v, str)})

                    if spec["method"] == "l1":
                        raw_mask = (test_h > 0.0).to(test_h.dtype)
                        raw_aligned = align_to_true_features(raw_mask, learned_idx, true_idx, test_data.support.shape[1])
                        row["l1_raw_relu_rho_model"] = float(raw_aligned.mean())
                        row["l1_raw_relu_selection_error"] = selection_error(raw_aligned, true_support)
                        row["l1_raw_relu_sigma_sel"] = selection_uncertainty(raw_aligned)
                    else:
                        row["l1_raw_relu_rho_model"] = np.nan
                        row["l1_raw_relu_selection_error"] = np.nan
                        row["l1_raw_relu_sigma_sel"] = np.nan

                    cache = {
                        "mask": mask_aligned,
                        "true_support": true_support,
                        "h": h_aligned,
                    }
                    return row, cache
                """
            ),
            md(
                """
                ## Run the Sweep

                Each method is trained over its own sparsity-control grid. The comparison uses the measured mask density `rho_model`, not the raw control value.
                """
            ),
            code(
                """
                final_rows = []
                history_rows = []
                mask_cache = {}

                for seed in seeds:
                    train_data, test_data = make_train_test(seed)
                    for spec_index, spec in enumerate(build_specs()):
                        run_id = f"{spec['method']}_{spec['control_name']}={spec['control_value']}_seed={seed}"
                        print("training", run_id)
                        init_seed = 100_000 + 1_000 * seed + spec_index
                        set_seed(init_seed)
                        model = build_model(spec)
                        result = fit_sae(
                            model,
                            train_data.x,
                            lr=lr,
                            batch_size=batch_size,
                            max_steps=train_steps,
                            gradient_clip_norm=gradient_clip_norm,
                            history_every=history_every,
                            seed=init_seed,
                        )
                        row, cache = evaluate_trained_model(model, train_data, test_data, spec, seed, run_id)
                        final_rows.append(row)
                        mask_cache[run_id] = cache
                        for hrow in result.history:
                            history_rows.append({
                                "run_id": run_id,
                                "seed": seed,
                                "method": spec["method"],
                                "method_label": METHOD_LABELS[spec["method"]],
                                "control_name": spec["control_name"],
                                "control_value": spec["control_value"],
                                **hrow,
                            })

                final_df = pd.DataFrame(final_rows).sort_values(["method", "rho_model"]).reset_index(drop=True)
                history_df = pd.DataFrame(history_rows)
                final_df
                """
            ),
            code(
                """
                final_df.to_csv(EXP_DIR / "final_metrics.csv", index=False)
                history_df.to_csv(EXP_DIR / "training_curves.csv", index=False)

                summary_cols = [
                    "method_label", "control_name", "control_value", "rho_model",
                    "generalization_error", "reconstruction_error", "selection_error",
                    "mask_uncertainty", "paper_style_sigma_sel", "support_f1",
                    "decoder_recovery_cosine", "support_average_precision", "support_roc_auc",
                ]
                final_df[summary_cols]
                """
            ),
            md(
                """
                ## Paper-Style Rho-Model Plots

                Read the hard-mask uncertainty correctly: TopK and thresholded L1 have zero intrinsic `m(1-m)` by definition. Their `paper_style_sigma_sel` can still be nonzero because it measures variation in selected features across samples.
                """
            ),
            code(
                """
                metrics = [
                    ("generalization_error", "Gen. error"),
                    ("reconstruction_error", "Recon. error"),
                    ("selection_error", "selection error"),
                    ("mask_uncertainty", "mean m(1-m)"),
                    ("paper_style_sigma_sel", "sample-mean mask variance"),
                    ("support_average_precision", "support average precision"),
                    ("decoder_recovery_cosine", "decoder recovery cosine"),
                    ("dead_fraction", "dead latent fraction"),
                ]

                fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=False)
                for ax, (metric, ylabel) in zip(axes.ravel(), metrics, strict=True):
                    for method in ["vgsae", "l1", "topk", "gated"]:
                        sub = final_df[final_df["method"] == method].sort_values("rho_model")
                        if sub.empty:
                            continue
                        ax.plot(
                            sub["rho_model"],
                            sub[metric],
                            marker="o",
                            linewidth=1.5,
                            color=METHOD_COLORS[method],
                            label=METHOD_LABELS[method],
                        )
                    ax.axvline(support_density, color="black", linestyle="--", linewidth=1, alpha=0.6)
                    ax.set_xlabel("rho_model")
                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=0.25)
                axes[0, 0].legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(EXP_DIR / "rho_model_metric_comparison.png", dpi=160)
                """
            ),
            md(
                """
                ## Training Curves

                These curves are intentionally raw. If a method wins a final metric only because training has not converged, this plot should expose it.
                """
            ),
            code(
                """
                curve_metrics = [
                    ("loss", "training loss"),
                    ("reconstruction_mse", "train reconstruction MSE"),
                    ("rho", "train rho"),
                ]

                fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
                for ax, (metric, ylabel) in zip(axes, curve_metrics, strict=True):
                    for method in ["vgsae", "l1", "topk", "gated"]:
                        sub = history_df[history_df["method"] == method]
                        if sub.empty or metric not in sub.columns:
                            continue
                        for _, run in sub.groupby("run_id"):
                            run = run.sort_values("step")
                            ax.plot(
                                run["step"],
                                run[metric],
                                color=METHOD_COLORS[method],
                                alpha=0.35,
                                linewidth=1,
                            )
                    ax.set_xlabel("step")
                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=0.25)

                handles = [
                    plt.Line2D([0], [0], color=METHOD_COLORS[m], label=METHOD_LABELS[m])
                    for m in ["vgsae", "l1", "topk", "gated"]
                ]
                axes[0].legend(handles=handles, fontsize=8)
                fig.tight_layout()
                fig.savefig(EXP_DIR / "training_curves.png", dpi=160)
                """
            ),
            md(
                """
                ## Representative Mask Heatmaps

                For each method, pick the run whose measured `rho_model` is closest to the true support density. This is a sanity check for whether the same density corresponds to the same sample-feature structure.
                """
            ),
            code(
                """
                representatives = []
                for method, sub in final_df.groupby("method"):
                    idx = (sub["rho_model"] - support_density).abs().idxmin()
                    representatives.append(final_df.loc[idx])
                representatives = pd.DataFrame(representatives).sort_values("method")

                n_show = min(80, n_test)
                fig, axes = plt.subplots(len(representatives), 2, figsize=(10, 2.2 * len(representatives)), sharex=True)
                if len(representatives) == 1:
                    axes = np.asarray([axes])

                for row_i, (_, row) in enumerate(representatives.iterrows()):
                    cache = mask_cache[row["run_id"]]
                    axes[row_i, 0].imshow(cache["true_support"][:n_show], aspect="auto", interpolation="nearest", vmin=0, vmax=1)
                    axes[row_i, 0].set_ylabel(row["method_label"])
                    axes[row_i, 0].set_title("true support")
                    axes[row_i, 1].imshow(cache["mask"][:n_show], aspect="auto", interpolation="nearest", vmin=0, vmax=1)
                    axes[row_i, 1].set_title(
                        f"mask, rho={row['rho_model']:.3f}, sel err={row['selection_error']:.3f}"
                    )

                for ax in axes.ravel():
                    ax.set_xlabel("matched feature")
                fig.tight_layout()
                fig.savefig(EXP_DIR / "mask_heatmaps.png", dpi=160)
                representatives[["method_label", "control_name", "control_value", "rho_model", "selection_error"]]
                """
            ),
            md(
                """
                ## Minimal Interpretation Checklist

                - The fair comparison is by measured `rho_model`, not by `gamma`, `k`, or L1 coefficient.
                - If TopK dominates at the correct density, VG-SAE's probabilistic mask is not buying selection quality on this synthetic setting.
                - If VG-SAE has higher `mask_uncertainty` near the true support density while also improving support F1 or clean error, the uncertainty variable is doing useful work rather than merely softening a bad mask.
                - L1's raw ReLU support is usually too dense; the GMM elbow columns show whether the paper-style threshold fixes that or just hides the density mismatch.
                """
            ),
        ]
    )


def vg_sparsity_sweep_notebook() -> dict:
    return notebook(
        [
            md(
                """
                # Experiment 8: VG-SAE Sparsity Sweep on Synthetic Sparse Coding

                Goal: isolate VG-SAE and sweep the paper-aligned nonnegative sparsity field `gamma`. This is the notebook to use when the question is not "which baseline wins?" but "how do VG-SAE reconstruction, support density, Bernoulli uncertainty, and dictionary recovery change as sparsity pressure increases?"

                The sweep uses `gamma >= 0` because the VG prior is `P(s_j) proportional to exp(-gamma s_j)`. Negative values favor dense supports and do not have the sparsity-field meaning used in the paper.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                from types import SimpleNamespace

                from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
                from src.sae_evaluate import (
                    amplitude_shrinkage,
                    decoder_recovery_cosine,
                    support_precision_recall,
                    susceptibility,
                    vg_sae_observables,
                )
                from src.sae_model import VGSAEConfig, VariationalGarroteSAE
                from src.sae_train import fit_sae
                from src.utils import set_seed
                """
            ),
            md(
                """
                ## Configuration

                Set `VGSAE_NOTEBOOK_FAST_DEV_RUN=1` for a quick smoke run. For paper figures, increase `seeds`, `train_steps`, and the gamma grid.
                """
            ),
            code(
                """
                FAST_DEV_RUN = bool(int(os.environ.get("VGSAE_NOTEBOOK_FAST_DEV_RUN", "0")))

                input_dim = 16
                n_features = 32
                support_density = 0.06
                coherence = 0.1
                noise_std = 0.03
                frequency_skew = 0.0
                amplitude_scale = 1.0

                if FAST_DEV_RUN:
                    seeds = [0]
                    n_train = 96
                    n_test = 96
                    train_steps = 4
                    history_every = 1
                    gamma_values = [0.0, 1.0]
                else:
                    seeds = [0, 1, 2]
                    n_train = 1024
                    n_test = 1024
                    train_steps = 800
                    history_every = 50
                    gamma_values = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

                batch_size = 128
                lr = 3.0e-3
                beta = 1.0
                gradient_clip_norm = 1.0
                EXP_DIR = OUTPUT_DIR / "exp08_synthetic_sparse_coding_vg_sparsity_sweep"
                EXP_DIR.mkdir(parents=True, exist_ok=True)
                """
            ),
            code(
                """
                def split_sparse_coding_data(data, n_train):
                    train = SimpleNamespace(
                        x=data.x[:n_train],
                        z=data.z[:n_train],
                        support=data.support[:n_train],
                        dictionary=data.dictionary,
                        clean_x=data.clean_x[:n_train],
                        feature_probabilities=data.feature_probabilities,
                    )
                    test = SimpleNamespace(
                        x=data.x[n_train:],
                        z=data.z[n_train:],
                        support=data.support[n_train:],
                        dictionary=data.dictionary,
                        clean_x=data.clean_x[n_train:],
                        feature_probabilities=data.feature_probabilities,
                    )
                    return train, test


                def make_train_test(seed):
                    data = make_synthetic_sparse_coding(
                        SyntheticSparseCodingConfig(
                            input_dim=input_dim,
                            n_features=n_features,
                            n_samples=n_train + n_test,
                            support_density=support_density,
                            coherence=coherence,
                            noise_std=noise_std,
                            frequency_skew=frequency_skew,
                            amplitude_scale=amplitude_scale,
                            seed=seed,
                        ),
                        device=DEVICE,
                    )
                    return split_sparse_coding_data(data, n_train)


                train0, test0 = make_train_test(seeds[0])
                print(
                    "train", tuple(train0.x.shape),
                    "test", tuple(test0.x.shape),
                    "empirical train density", float(train0.support.mean()),
                    "empirical test density", float(test0.support.mean()),
                )
                """
            ),
            md(
                """
                ## Run Sweep

                Decoder atoms are evaluated against the ground-truth dictionary, so support precision and recall are meaningful only after decoder matching.
                """
            ),
            code(
                """
                final_rows = []
                history_rows = []

                for seed in seeds:
                    train_data, test_data = make_train_test(seed)
                    for gamma_index, gamma in enumerate(gamma_values):
                        run_id = f"vgsae_gamma={gamma}_seed={seed}"
                        print("training", run_id)
                        init_seed = 100_000 + 1_000 * seed + gamma_index
                        set_seed(init_seed)
                        model = VariationalGarroteSAE(
                            VGSAEConfig(
                                input_dim=input_dim,
                                n_latents=n_features,
                                lambda_sparsity=float(gamma),
                                beta=beta,
                            )
                        )
                        result = fit_sae(
                            model,
                            train_data.x,
                            lr=lr,
                            batch_size=batch_size,
                            max_steps=train_steps,
                            gradient_clip_norm=gradient_clip_norm,
                            history_every=history_every,
                            seed=init_seed,
                        )
                        obs = vg_sae_observables(model, test_data.x)
                        precision, recall = support_precision_recall(
                            model,
                            test_data.x,
                            test_data.support,
                            test_data.dictionary,
                        )
                        final_rows.append({
                            "run_id": run_id,
                            "seed": seed,
                            "gamma": float(gamma),
                            "empirical_support_density": float(test_data.support.mean()),
                            "reconstruction_mse": obs.mse,
                            "rho_model": obs.rho,
                            "entropy": obs.entropy,
                            "v_eff": obs.v_eff,
                            "dead_fraction": obs.dead_fraction,
                            "interference_energy": obs.interference_energy,
                            "variance_energy": obs.variance_energy,
                            "decoder_recovery_cosine": decoder_recovery_cosine(model.decoder.weight, test_data.dictionary),
                            "support_precision": precision,
                            "support_recall": recall,
                            "amplitude_shrinkage": amplitude_shrinkage(
                                model,
                                test_data.x,
                                test_data.z,
                                test_data.dictionary,
                            ),
                        })
                        for hrow in result.history:
                            history_rows.append({"run_id": run_id, "seed": seed, "gamma": float(gamma), **hrow})

                final_df = pd.DataFrame(final_rows).sort_values(["seed", "gamma"]).reset_index(drop=True)
                history_df = pd.DataFrame(history_rows)
                for seed, sub in final_df.groupby("seed"):
                    idx = sub.index
                    final_df.loc[idx, "susceptibility"] = susceptibility(
                        sub["gamma"].to_numpy(dtype=float),
                        sub["rho_model"].to_numpy(dtype=float),
                    )
                final_df
                """
            ),
            code(
                """
                final_df.to_csv(EXP_DIR / "vg_sparsity_sweep.csv", index=False)
                history_df.to_csv(EXP_DIR / "vg_sparsity_training_curves.csv", index=False)

                metrics = [
                    ("rho_model", "rho_model"),
                    ("v_eff", "mean m(1-m)"),
                    ("reconstruction_mse", "test MSE"),
                    ("support_recall", "support recall"),
                    ("decoder_recovery_cosine", "decoder recovery"),
                    ("susceptibility", "-d rho / d gamma"),
                ]

                fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
                for ax, (metric, ylabel) in zip(axes.ravel(), metrics, strict=True):
                    for seed, sub in final_df.groupby("seed"):
                        sub = sub.sort_values("gamma")
                        ax.plot(sub["gamma"], sub[metric], marker="o", linewidth=1.5, label=f"seed={seed}")
                    ax.axhline(support_density, color="black", linestyle="--", linewidth=1, alpha=0.5) if metric == "rho_model" else None
                    ax.set_xlabel("gamma")
                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=0.25)
                axes[0, 0].legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(EXP_DIR / "vg_sparsity_sweep.png", dpi=160)
                """
            ),
            md(
                """
                ## Interpretation Checks

                - `rho_model` should generally decrease as `gamma` increases. If not, optimization is dominating the prior.
                - A useful VG-SAE regime should keep reconstruction error low while moving `rho_model` toward the empirical support density.
                - A peak in susceptibility or `v_eff` is a transition diagnostic, not proof that the true number of features has been recovered.
                """
            ),
        ]
    )


def toy_superposition_notebook() -> dict:
    return notebook(
        [
            md(
                """
                # Experiment 3: Toy Superposition Phase Diagram

                Proposal target: map phases over feature-frequency skew, lambda, and overcomplete dictionary size. This notebook uses the synthetic sparse-coding generator as a controlled toy superposition source with `N >> d`, feature probabilities `p_i ∝ i^{-alpha}`, and a lambda sweep.

                The phase labels are operational proxies:
                under-selected, monosemantic recovery, uncertain/interference, and overcomplete unstable. They are not metaphysical truth; they are diagnostics.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
                from src.sae_evaluate import decoder_recovery_cosine, support_precision_recall, vg_sae_observables
                from src.sae_model import VGSAEConfig, VariationalGarroteSAE
                from src.sae_train import fit_sae

                input_dim = 8
                width = 32
                alphas = [0.0, 0.5, 1.0, 1.5]
                lambdas = [0.0, 0.5, 1.0, 2.0, 3.0]
                train_steps = 200
                """
            ),
            code(
                """
                rows = []
                for alpha in alphas:
                    data = make_synthetic_sparse_coding(
                        SyntheticSparseCodingConfig(
                            input_dim=input_dim,
                            n_features=width,
                            n_samples=512,
                            support_density=0.08,
                            coherence=0.15,
                            noise_std=0.03,
                            frequency_skew=alpha,
                            seed=10 + int(alpha * 10),
                        ),
                        device=DEVICE,
                    )
                    for lam in lambdas:
                        model = VariationalGarroteSAE(
                            VGSAEConfig(input_dim=input_dim, n_latents=width, lambda_sparsity=lam, beta=1.0)
                        )
                        fit_sae(model, data.x, max_steps=train_steps, batch_size=128, lr=3e-3, history_every=100, seed=0)
                        obs = vg_sae_observables(model, data.x)
                        precision, recall = support_precision_recall(model, data.x, data.support, data.dictionary)
                        rows.append({
                            "alpha": alpha,
                            "lambda": lam,
                            "mse": obs.mse,
                            "rho": obs.rho,
                            "v_eff": obs.v_eff,
                            "entropy": obs.entropy,
                            "interference_energy": obs.interference_energy,
                            "dead_fraction": obs.dead_fraction,
                            "decoder_recovery_cosine": decoder_recovery_cosine(model.decoder.weight, data.dictionary),
                            "support_precision": precision,
                            "support_recall": recall,
                        })
                df = pd.DataFrame(rows)
                df.head()
                """
            ),
            code(
                """
                mse_cut = df["mse"].median()
                veff_cut = df["v_eff"].quantile(0.70)
                recovery_cut = df["decoder_recovery_cosine"].median()

                def assign_phase(row):
                    if row["rho"] < 0.04 or row["mse"] > mse_cut * 1.25:
                        return 0  # under-selected
                    if row["mse"] <= mse_cut and row["v_eff"] <= veff_cut and row["decoder_recovery_cosine"] >= recovery_cut:
                        return 1  # monosemantic recovery proxy
                    if row["v_eff"] > veff_cut or row["interference_energy"] > df["interference_energy"].quantile(0.70):
                        return 2  # uncertain/interference
                    return 3  # unstable/ambiguous

                df["phase"] = df.apply(assign_phase, axis=1)
                phase_names = {
                    0: "under-selected",
                    1: "recovery",
                    2: "uncertain/interference",
                    3: "unstable/ambiguous",
                }
                df["phase_name"] = df["phase"].map(phase_names)
                df
                """
            ),
            code(
                """
                def heatmap(metric, ax, title, cmap="viridis"):
                    pivot = df.pivot(index="alpha", columns="lambda", values=metric).sort_index()
                    image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap=cmap)
                    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
                    ax.set_yticks(range(len(pivot.index)), pivot.index)
                    ax.set_xlabel("lambda")
                    ax.set_ylabel("alpha")
                    ax.set_title(title)
                    return image

                out = OUTPUT_DIR / "exp03_toy_superposition"
                out.mkdir(parents=True, exist_ok=True)
                df.to_csv(out / "toy_superposition_phase_diagram.csv", index=False)

                fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
                for metric, ax, title in [
                    ("v_eff", axes[0], "V_eff"),
                    ("decoder_recovery_cosine", axes[1], "Decoder recovery"),
                    ("phase", axes[2], "Operational phase"),
                ]:
                    image = heatmap(metric, ax, title, cmap="tab10" if metric == "phase" else "viridis")
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(out / "toy_superposition_phase_diagram.png", dpi=160)
                df[["alpha", "lambda", "phase_name", "v_eff", "decoder_recovery_cosine", "mse"]]
                """
            ),
            md(
                """
                **Physics sanity check:** a phase diagram earns the name only if the transition region is reproducible under seed and width changes. Treat this notebook as the first map, not the final territory.
                """
            ),
        ]
    )


def gpt2_notebook() -> dict:
    return notebook(
        [
            md(
                """
                # Experiment 4: GPT-2 Small Layer-8 Residual-Stream Sweep

                Proposal target: train VG-SAE on GPT-2 small residual-stream activations, sweep dictionary width and lambda, and plot `lambda*(N)=argmax_lambda V_eff(lambda,N)`.

                Interpretation is deliberately narrow: `lambda*` is a recoverability-transition diagnostic, not the true number of language-model features.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                from src.gpt2_activations import ActivationCacheConfig, cache_gpt2_residual_activations, load_activation_cache
                from src.sae_evaluate import susceptibility, vg_sae_observables
                from src.sae_model import VGSAEConfig, VariationalGarroteSAE
                from src.sae_train import fit_sae

                cache_path = PROJECT_ROOT / "outputs" / "gpt2" / "gpt2_layer8_resid.pt"
                max_tokens = 2048
                """
            ),
            code(
                """
                if not cache_path.exists():
                    cache_gpt2_residual_activations(
                        ActivationCacheConfig(
                            model_name="gpt2",
                            layer=8,
                            max_tokens=max_tokens,
                            batch_size=4,
                            sequence_length=64,
                            device="cpu",
                        ),
                        cache_path,
                    )
                x = load_activation_cache(cache_path)[:max_tokens].to(DEVICE)
                print(x.shape, float(x.mean()), float(x.std()))
                """
            ),
            code(
                """
                expansion_factors = [2, 4]
                lambdas = [0.0, 0.5, 1.0, 2.0, 3.0]
                train_steps = 300
                rows = []
                for expansion in expansion_factors:
                    width = int(expansion * x.shape[1])
                    for lam in lambdas:
                        model = VariationalGarroteSAE(
                            VGSAEConfig(input_dim=x.shape[1], n_latents=width, lambda_sparsity=lam, beta=1.0)
                        )
                        fit_sae(model, x, max_steps=train_steps, batch_size=128, lr=1e-3, history_every=100, seed=0)
                        obs = vg_sae_observables(model, x)
                        rows.append({
                            "expansion_factor": expansion,
                            "width": width,
                            "lambda": lam,
                            "mse": obs.mse,
                            "rho": obs.rho,
                            "entropy": obs.entropy,
                            "v_eff": obs.v_eff,
                            "dead_fraction": obs.dead_fraction,
                            "interference_energy": obs.interference_energy,
                            "variance_energy": obs.variance_energy,
                        })
                df = pd.DataFrame(rows)
                for expansion in expansion_factors:
                    mask = df["expansion_factor"] == expansion
                    df.loc[mask, "susceptibility"] = susceptibility(
                        df.loc[mask, "lambda"].to_numpy(dtype=float),
                        df.loc[mask, "rho"].to_numpy(dtype=float),
                    )
                df
                """
            ),
            code(
                """
                out = OUTPUT_DIR / "exp04_gpt2_layer8"
                out.mkdir(parents=True, exist_ok=True)
                df.to_csv(out / "gpt2_layer8_lambda_width_sweep.csv", index=False)

                lambda_star = df.loc[df.groupby("width")["v_eff"].idxmax(), ["width", "lambda", "v_eff"]].rename(
                    columns={"lambda": "lambda_star"}
                )
                fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
                for width, sub in df.groupby("width"):
                    sub = sub.sort_values("lambda")
                    axes[0].plot(sub["lambda"], sub["v_eff"], marker="o", label=f"N={width}")
                    axes[1].plot(sub["lambda"], sub["rho"], marker="o", label=f"N={width}")
                axes[0].set_title("Recoverability transition diagnostic")
                axes[0].set_xlabel("lambda")
                axes[0].set_ylabel("V_eff")
                axes[1].set_title("Active density")
                axes[1].set_xlabel("lambda")
                axes[1].set_ylabel("rho")
                for ax in axes:
                    ax.legend()
                fig.tight_layout()
                fig.savefig(out / "gpt2_layer8_v_eff_sweep.png", dpi=160)
                lambda_star
                """
            ),
            md(
                """
                **Do not oversell this:** if `lambda_star` shifts with width, the result is still informative. It says recoverable feature scale is width-dependent in this setup.
                """
            ),
        ]
    )


def feature_quality_notebook() -> dict:
    return notebook(
        [
            md(
                """
                # Experiment 5: Feature Uncertainty vs Feature Quality

                Proposal target: test whether feature-level support uncertainty `U_i = E_b[m_{bi}(1-m_{bi})]` predicts polysemanticity, instability, and weaker feature quality.

                This notebook gives a runnable synthetic version with ground-truth decoder recovery and seed stability. It also has an optional hook for external feature-quality scores such as FMS, JS separability, or autointerpretability.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                from scipy.stats import spearmanr

                from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
                from src.sae_evaluate import decoder_cosine_matrix, feature_uncertainty
                from src.sae_model import VGSAEConfig, VariationalGarroteSAE
                from src.sae_train import fit_sae

                cfg = SyntheticSparseCodingConfig(
                    input_dim=16,
                    n_features=32,
                    n_samples=768,
                    support_density=0.06,
                    coherence=0.15,
                    noise_std=0.03,
                    frequency_skew=1.0,
                    seed=20,
                )
                data = make_synthetic_sparse_coding(cfg, device=DEVICE)
                lambda_sparsity = 1.0
                seeds = [0, 1, 2]
                """
            ),
            code(
                """
                models = []
                for seed in seeds:
                    model = VariationalGarroteSAE(
                        VGSAEConfig(
                            input_dim=cfg.input_dim,
                            n_latents=cfg.n_features,
                            lambda_sparsity=lambda_sparsity,
                            beta=1.0,
                        )
                    )
                    fit_sae(model, data.x, max_steps=300, batch_size=128, lr=3e-3, history_every=100, seed=seed)
                    models.append(model)
                """
            ),
            code(
                """
                reference = models[0]
                uncertainty = feature_uncertainty(reference, data.x).detach().cpu().numpy()
                true_cosines = decoder_cosine_matrix(reference.decoder.weight, data.dictionary)
                recovery = true_cosines.max(axis=1)

                stability_scores = []
                for i in range(reference.config.n_latents):
                    scores = []
                    ref_vec = reference.decoder.weight[:, i].detach().cpu()
                    ref_vec = ref_vec / ref_vec.norm().clamp_min(1e-12)
                    for other in models[1:]:
                        other_decoder = other.decoder.weight.detach().cpu()
                        other_decoder = other_decoder / other_decoder.norm(dim=0, keepdim=True).clamp_min(1e-12)
                        scores.append(float(torch.abs(ref_vec @ other_decoder).max()))
                    stability_scores.append(np.mean(scores))

                quality_df = pd.DataFrame({
                    "feature": np.arange(reference.config.n_latents),
                    "U_i": uncertainty,
                    "decoder_recovery_cosine": recovery,
                    "seed_stability_cosine": stability_scores,
                })
                quality_df["recovery_rank"] = quality_df["decoder_recovery_cosine"].rank(ascending=False)
                quality_df.head()
                """
            ),
            code(
                """
                for metric in ["decoder_recovery_cosine", "seed_stability_cosine"]:
                    corr = spearmanr(quality_df["U_i"], quality_df[metric], nan_policy="omit")
                    print(f"Spearman U_i vs {metric}: rho={corr.statistic:.3f}, p={corr.pvalue:.3g}")

                external_scores = PROJECT_ROOT / "outputs" / "feature_scores.csv"
                if external_scores.exists():
                    score_df = pd.read_csv(external_scores)
                    merged = quality_df.merge(score_df, on="feature", how="inner")
                    for column in [c for c in merged.columns if c not in quality_df.columns]:
                        corr = spearmanr(merged["U_i"], merged[column], nan_policy="omit")
                        print(f"Spearman U_i vs external {column}: rho={corr.statistic:.3f}, p={corr.pvalue:.3g}")
                """
            ),
            code(
                """
                out = OUTPUT_DIR / "exp05_uncertainty_quality"
                out.mkdir(parents=True, exist_ok=True)
                quality_df.to_csv(out / "feature_uncertainty_quality.csv", index=False)

                fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
                axes[0].scatter(quality_df["U_i"], quality_df["decoder_recovery_cosine"], alpha=0.8)
                axes[0].set_xlabel("U_i")
                axes[0].set_ylabel("decoder recovery cosine")
                axes[0].set_title("Uncertainty vs recovery")
                axes[1].scatter(quality_df["U_i"], quality_df["seed_stability_cosine"], alpha=0.8)
                axes[1].set_xlabel("U_i")
                axes[1].set_ylabel("seed stability cosine")
                axes[1].set_title("Uncertainty vs stability")
                fig.tight_layout()
                fig.savefig(out / "feature_uncertainty_quality.png", dpi=160)
                """
            ),
            md(
                """
                **Hard criterion:** if high `U_i` does not anticorrelate with recovery or stability in synthetic ground-truth settings, do not move this claim to real LMs yet. Fix calibration first.
                """
            ),
        ]
    )


def ioi_notebook() -> dict:
    return notebook(
        [
            md(
                """
                # Experiment 6: IOI Causal-Control Case Study

                Proposal target: test whether low-uncertainty feature sets produce cleaner causal interventions. This notebook implements the first operational GPT-2 IOI patching scaffold:

                1. build clean/corrupted IOI prompts;
                2. collect layer-8 residual vectors at the answer position;
                3. train or load a small VG-SAE on GPT-2 residual activations;
                4. select feature patches by clean-corrupt latent differences;
                5. compare logit-difference recovery and `U_patch = sum_i U_i`.

                This is intentionally a scaffold for the causal benchmark. The real paper version should broaden prompts, seeds, and controls.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                from src.gpt2_activations import ActivationCacheConfig, cache_gpt2_residual_activations, load_activation_cache
                from src.sae_evaluate import feature_uncertainty
                from src.sae_model import VGSAEConfig, VariationalGarroteSAE
                from src.sae_train import fit_sae

                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_name = "gpt2"
                layer = 8
                cache_path = PROJECT_ROOT / "outputs" / "gpt2" / "gpt2_layer8_resid.pt"
                """
            ),
            code(
                """
                if not cache_path.exists():
                    cache_gpt2_residual_activations(
                        ActivationCacheConfig(model_name=model_name, layer=layer, max_tokens=2048, sequence_length=64),
                        cache_path,
                    )

                train_x = load_activation_cache(cache_path)[:2048].to(DEVICE)
                sae = VariationalGarroteSAE(
                    VGSAEConfig(input_dim=train_x.shape[1], n_latents=2 * train_x.shape[1], lambda_sparsity=1.0, beta=1.0)
                )
                fit_sae(sae, train_x, max_steps=300, batch_size=128, lr=1e-3, history_every=100, seed=0)
                U = feature_uncertainty(sae, train_x)
                """
            ),
            code(
                """
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                lm = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
                lm.eval()

                clean_prompts = [
                    "When John and Mary went to the store, John gave a bottle to",
                    "When Alice and Bob visited the park, Alice handed a book to",
                    "When Sarah and Tom left the office, Sarah passed the keys to",
                ]
                corrupted_prompts = [
                    "When Mary and John went to the store, John gave a bottle to",
                    "When Bob and Alice visited the park, Alice handed a book to",
                    "When Tom and Sarah left the office, Sarah passed the keys to",
                ]
                correct_answers = [" Mary", " Bob", " Tom"]
                incorrect_answers = [" John", " Alice", " Sarah"]
                correct_ids = torch.tensor([tokenizer.encode(a, add_special_tokens=False)[0] for a in correct_answers], device=DEVICE)
                incorrect_ids = torch.tensor([tokenizer.encode(a, add_special_tokens=False)[0] for a in incorrect_answers], device=DEVICE)
                """
            ),
            code(
                """
                def encode_prompts(prompts):
                    return tokenizer(prompts, padding=True, return_tensors="pt").to(DEVICE)

                def final_positions(encoded):
                    return encoded["attention_mask"].sum(dim=1) - 1

                def hidden_at_answer(prompts):
                    encoded = encode_prompts(prompts)
                    with torch.no_grad():
                        outputs = lm(**encoded, output_hidden_states=True, use_cache=False)
                    positions = final_positions(encoded)
                    hidden = outputs.hidden_states[layer + 1][torch.arange(len(prompts)), positions]
                    logits = outputs.logits[torch.arange(len(prompts)), positions]
                    return hidden.detach(), logits.detach(), positions, encoded

                def logit_diff(logits):
                    return (logits[torch.arange(logits.shape[0]), correct_ids] - logits[torch.arange(logits.shape[0]), incorrect_ids]).detach()

                clean_h, clean_logits, clean_pos, clean_encoded = hidden_at_answer(clean_prompts)
                corrupt_h, corrupt_logits, corrupt_pos, corrupt_encoded = hidden_at_answer(corrupted_prompts)
                print("clean logit diff", logit_diff(clean_logits))
                print("corrupt logit diff", logit_diff(corrupt_logits))
                """
            ),
            code(
                """
                with torch.no_grad():
                    clean_latents = sae(clean_h)["h"]
                    corrupt_latents = sae(corrupt_h)["h"]
                    delta = (clean_latents - corrupt_latents).abs().mean(dim=0)
                    candidate_features = torch.topk(delta, k=32).indices

                rows = []
                for k in [2, 4, 8, 16, 32]:
                    patch_features = candidate_features[:k]
                    with torch.no_grad():
                        corrupt_out = sae(corrupt_h)
                        clean_out = sae(clean_h)
                        patched_latents = corrupt_out["h"].clone()
                        patched_latents[:, patch_features] = clean_out["h"][:, patch_features]
                        patched_residual = sae.decode(patched_latents)
                    rows.append({
                        "k": k,
                        "U_patch": float(U[patch_features].sum()),
                        "patch_residual_norm": float((patched_residual - corrupt_h).norm(dim=1).mean()),
                    })
                patch_df = pd.DataFrame(rows)
                patch_df
                """
            ),
            code(
                """
                def run_with_residual_patch(encoded, positions, patched_residual):
                    batch_indices = torch.arange(patched_residual.shape[0], device=patched_residual.device)

                    def hook(_module, _inputs, outputs):
                        hidden = outputs[0].clone()
                        hidden[batch_indices, positions] = patched_residual
                        return (hidden,) + outputs[1:]

                    handle = lm.transformer.h[layer].register_forward_hook(hook)
                    try:
                        with torch.no_grad():
                            outputs = lm(**encoded, use_cache=False)
                    finally:
                        handle.remove()
                    logits = outputs.logits[batch_indices, positions]
                    return logits.detach()

                rows = []
                for k in [2, 4, 8, 16, 32]:
                    patch_features = candidate_features[:k]
                    with torch.no_grad():
                        corrupt_out = sae(corrupt_h)
                        clean_out = sae(clean_h)
                        patched_latents = corrupt_out["h"].clone()
                        patched_latents[:, patch_features] = clean_out["h"][:, patch_features]
                        patched_residual = sae.decode(patched_latents)
                    logits = run_with_residual_patch(corrupt_encoded, corrupt_pos, patched_residual)
                    rows.append({
                        "k": k,
                        "U_patch": float(U[patch_features].sum()),
                        "patched_logit_diff": float(logit_diff(logits).mean()),
                        "clean_logit_diff": float(logit_diff(clean_logits).mean()),
                        "corrupt_logit_diff": float(logit_diff(corrupt_logits).mean()),
                    })
                causal_df = pd.DataFrame(rows)
                causal_df["recovery_fraction"] = (
                    (causal_df["patched_logit_diff"] - causal_df["corrupt_logit_diff"])
                    / (causal_df["clean_logit_diff"] - causal_df["corrupt_logit_diff"]).replace(0.0, np.nan)
                )
                causal_df
                """
            ),
            code(
                """
                out = OUTPUT_DIR / "exp06_ioi_causal_control"
                out.mkdir(parents=True, exist_ok=True)
                causal_df.to_csv(out / "ioi_causal_control.csv", index=False)

                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.scatter(causal_df["U_patch"], causal_df["recovery_fraction"])
                for _, row in causal_df.iterrows():
                    ax.annotate(f"k={int(row['k'])}", (row["U_patch"], row["recovery_fraction"]))
                ax.set_xlabel("U_patch")
                ax.set_ylabel("logit-diff recovery fraction")
                ax.set_title("IOI patch reliability vs uncertainty")
                fig.tight_layout()
                fig.savefig(out / "ioi_causal_control.png", dpi=160)
                """
            ),
            md(
                """
                **Critical caveat:** this is a first causal-control scaffold, not a publishable IOI benchmark. The publishable version needs more prompt templates, answer-token validation, negative controls, seed sweeps, activation-patching baselines, and comparison against TopK/Gated feature sets.
                """
            ),
        ]
    )


def write_notebooks() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_non_amortized_vg_sanity_check.ipynb": non_amortized_notebook(),
        "02_synthetic_sparse_coding_transition.ipynb": synthetic_sparse_coding_notebook(),
        "03_toy_superposition_phase_diagram.ipynb": toy_superposition_notebook(),
        "04_gpt2_small_layer8_residual_stream.ipynb": gpt2_notebook(),
        "05_feature_uncertainty_quality.ipynb": feature_quality_notebook(),
        "06_ioi_causal_control_case_study.ipynb": ioi_notebook(),
        "07_synthetic_sparse_coding_rho_model_comparison.ipynb": synthetic_rho_model_comparison_notebook(),
        "08_synthetic_sparse_coding_vg_sparsity_sweep.ipynb": vg_sparsity_sweep_notebook(),
    }
    for name, payload in notebooks.items():
        (NOTEBOOK_DIR / name).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    readme = dedent(
        """
        # VG-SAE Experiment Notebooks

        These notebooks are split by proposal experiment rather than merged into one file.

        1. `01_non_amortized_vg_sanity_check.ipynb`: direct per-sample VG support inference.
        2. `02_synthetic_sparse_coding_transition.ipynb`: synthetic sparse coding, baselines, and variance-term ablation.
        3. `03_toy_superposition_phase_diagram.ipynb`: phase diagram over feature-frequency skew and lambda.
        4. `04_gpt2_small_layer8_residual_stream.ipynb`: GPT-2 small layer-8 residual-stream lambda/width sweep.
        5. `05_feature_uncertainty_quality.ipynb`: feature uncertainty versus recovery and seed stability.
        6. `06_ioi_causal_control_case_study.ipynb`: first IOI causal-control patching scaffold.
        7. `07_synthetic_sparse_coding_rho_model_comparison.ipynb`: VG-SAE and baseline synthetic sparse-coding comparison by measured rho_model.
        8. `08_synthetic_sparse_coding_vg_sparsity_sweep.ipynb`: VG-SAE-only nonnegative gamma sparsity sweep.

        Run notebooks from the project root or from this directory. Outputs are written under `outputs/notebooks/`.
        """
    ).strip() + "\n"
    (NOTEBOOK_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    write_notebooks()
