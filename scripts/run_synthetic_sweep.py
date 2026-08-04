"""Run a CPU-sized synthetic VG-SAE lambda sweep and write CSV/PNG outputs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
from src.sae_evaluate import (
    amplitude_shrinkage,
    decoder_recovery_cosine,
    support_precision_recall,
    susceptibility,
    vg_sae_observables,
)
from src.sae_model import (
    BatchTopKSAE,
    BatchTopKSAEConfig,
    GatedSAE,
    GatedSAEConfig,
    JumpReLUSAE,
    JumpReLUSAEConfig,
    L1ReLUSAE,
    L1SAEConfig,
    TopKSAE,
    TopKSAEConfig,
    VGSAEConfig,
    VariationalGarroteSAE,
)
from src.sae_train import fit_sae


def _float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def _int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic VG-SAE transition sweep.")
    parser.add_argument("--output-dir", default="outputs/synthetic_sweep")
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--widths", default="32")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--support-density", type=float, default=0.06)
    parser.add_argument("--coherence", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--frequency-skew", type=float, default=0.0)
    parser.add_argument("--lambdas", default="0.0,0.5,1.0,2.0,3.0,5.0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta-mode", choices=("profiled", "fixed", "learned"), default="profiled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--include-no-variance", action="store_true")
    return parser.parse_args()


def train_vg_variant(
    x,
    n_latents: int,
    lambda_value: float,
    args: argparse.Namespace,
    use_variance_term: bool,
) -> VariationalGarroteSAE:
    model = VariationalGarroteSAE(
        VGSAEConfig(
            input_dim=x.shape[1],
            n_latents=n_latents,
            lambda_sparsity=lambda_value,
            beta=args.beta,
            beta_mode=args.beta_mode,
            use_variance_term=use_variance_term,
        )
    )
    fit_sae(
        model,
        x,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.steps,
        history_every=max(args.steps // 3, 1),
        seed=args.seed,
    )
    return model


def append_vg_row(
    rows: list[dict[str, float | str]],
    model: VariationalGarroteSAE,
    data,
    width: int,
    lambda_value: float,
    label: str,
) -> None:
    obs = vg_sae_observables(model, data.x)
    precision, recall = support_precision_recall(model, data.x, data.support, data.dictionary)
    rows.append(
        {
            "model": label,
            "beta_mode": str(model.config.beta_mode),
            "beta": model.config.beta,
            "width": width,
            "lambda": lambda_value,
            "mse": obs.mse,
            "rho": obs.rho,
            "entropy": obs.entropy,
            "v_eff": obs.v_eff,
            "susceptibility": np.nan,
            "dead_fraction": obs.dead_fraction,
            "interference_energy": obs.interference_energy,
            "variance_energy": obs.variance_energy,
            "decoder_recovery_cosine": decoder_recovery_cosine(
                model.decoder.weight, data.dictionary
            ),
            "support_precision": precision,
            "support_recall": recall,
            "amplitude_shrinkage": amplitude_shrinkage(model, data.x, data.z, data.dictionary),
        }
    )


def append_baselines(
    rows: list[dict[str, float | str]], data, width: int, args: argparse.Namespace
) -> None:
    k = max(1, min(width, round(args.support_density * width)))
    dimensions = {"input_dim": data.x.shape[1], "n_latents": width}
    baselines = [
        ("l1", L1ReLUSAE(L1SAEConfig(**dimensions, l1_coefficient=1.0e-3))),
        ("topk", TopKSAE(TopKSAEConfig(**dimensions, k=k))),
        ("batchtopk", BatchTopKSAE(BatchTopKSAEConfig(**dimensions, k=float(k)))),
        ("jumprelu", JumpReLUSAE(JumpReLUSAEConfig(**dimensions))),
        ("gated", GatedSAE(GatedSAEConfig(**dimensions, l1_coefficient=1.0e-3))),
    ]
    for name, model in baselines:
        result = fit_sae(
            model,
            data.x,
            lr=args.lr,
            batch_size=args.batch_size,
            max_steps=args.steps,
            history_every=max(args.steps // 3, 1),
            seed=args.seed,
        )
        final = result.history[-1]
        rows.append(
            {
                "model": name,
                "beta_mode": args.beta_mode,
                "beta": args.beta,
                "width": width,
                "lambda": np.nan,
                "mse": final.get("reconstruction_mse", np.nan),
                "rho": final.get("rho", np.nan),
                "entropy": np.nan,
                "v_eff": np.nan,
                "susceptibility": np.nan,
                "dead_fraction": np.nan,
                "interference_energy": np.nan,
                "variance_energy": np.nan,
                "decoder_recovery_cosine": decoder_recovery_cosine(
                    model.decoder.weight, data.dictionary
                ),
                "support_precision": np.nan,
                "support_recall": np.nan,
                "amplitude_shrinkage": np.nan,
            }
        )


def add_susceptibilities(rows: list[dict[str, float | str]]) -> None:
    for model_name in sorted({row["model"] for row in rows}):
        if not str(model_name).startswith("vg"):
            continue
        widths = sorted({int(row["width"]) for row in rows if row["model"] == model_name})
        for width in widths:
            idx = [
                i
                for i, row in enumerate(rows)
                if row["model"] == model_name and int(row["width"]) == width
            ]
            lambdas = np.array([float(rows[i]["lambda"]) for i in idx])
            rhos = np.array([float(rows[i]["rho"]) for i in idx])
            chi = susceptibility(lambdas, rhos)
            for i, value in zip(idx, chi, strict=True):
                rows[i]["susceptibility"] = float(value)


def write_outputs(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "synthetic_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(6, 4))
    for model_name in sorted({row["model"] for row in rows if str(row["model"]).startswith("vg")}):
        subset = [row for row in rows if row["model"] == model_name]
        for width in sorted({int(row["width"]) for row in subset}):
            points = sorted(
                [row for row in subset if int(row["width"]) == width],
                key=lambda row: float(row["lambda"]),
            )
            plt.plot(
                [row["lambda"] for row in points],
                [row["v_eff"] for row in points],
                marker="o",
                label=f"{model_name}, N={width}",
            )
    plt.xlabel("lambda")
    plt.ylabel("V_eff")
    plt.title("VG-SAE gate-variance transition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "v_eff_vs_lambda.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | str]] = []
    for width in _int_list(args.widths):
        data = make_synthetic_sparse_coding(
            SyntheticSparseCodingConfig(
                input_dim=args.input_dim,
                n_features=width,
                n_samples=args.n_samples,
                support_density=args.support_density,
                coherence=args.coherence,
                noise_std=args.noise_std,
                frequency_skew=args.frequency_skew,
                seed=args.seed,
            )
        )
        for lambda_value in _float_list(args.lambdas):
            model = train_vg_variant(data.x, width, lambda_value, args, use_variance_term=True)
            append_vg_row(rows, model, data, width, lambda_value, "vg")
            if args.include_no_variance:
                no_var_model = train_vg_variant(
                    data.x, width, lambda_value, args, use_variance_term=False
                )
                append_vg_row(rows, no_var_model, data, width, lambda_value, "vg_no_variance")
        if args.include_baselines:
            append_baselines(rows, data, width, args)
    add_susceptibilities(rows)
    write_outputs(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
