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
                def decoder_weight(model):
                    return model.W_dec.T if hasattr(model, "W_dec") else model.decoder.weight


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
                        "decoder_recovery_cosine": decoder_recovery_cosine(decoder_weight(model), data.dictionary),
                        "support_precision": precision,
                        "support_recall": recall,
                        "amplitude_shrinkage": amplitude_shrinkage(model, data.x, data.z, data.dictionary),
                    }

                def run_baselines(data, width):
                    k = max(1, round(data.support.mean().item() * width))
                    baselines = [
                        ("l1", L1ReLUSAE(L1SAEConfig(input_dim=data.x.shape[1], n_latents=width, l1_coefficient=1e-3))),
                        ("topk", TopKSAE(TopKSAEConfig(d_in=data.x.shape[1], d_sae=width, k=k))),
                        ("gated", GatedSAE(GatedSAEConfig(d_in=data.x.shape[1], d_sae=width, l1_coefficient=1e-3))),
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
                            "decoder_recovery_cosine": decoder_recovery_cosine(decoder_weight(model), data.dictionary),
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
        7. `07_synthetic_sparse_coding_rho_model_comparison.ipynb`: manually
           curated six-way comparison by measured rho_model. The bulk generator
           preserves it; current outputs use `exp07_saelens_v647_six_method/`,
           separate from the preserved four-method legacy artifacts.
        8. `08_synthetic_sparse_coding_vg_sparsity_sweep.ipynb`: VG-SAE-only nonnegative gamma sparsity sweep.

        Run notebooks from the project root or from this directory. Outputs are written under `outputs/notebooks/`.
        """
    ).strip() + "\n"
    (NOTEBOOK_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    write_notebooks()
