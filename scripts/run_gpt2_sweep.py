"""Run a small VG-SAE lambda sweep on cached GPT-2 residual activations."""

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

from src.gpt2_activations import load_activation_cache
from src.sae_evaluate import susceptibility, vg_sae_observables
from src.sae_model import VGSAEConfig, VariationalGarroteSAE
from src.sae_train import fit_sae


def _float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def _int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VG-SAE sweep on cached GPT-2 activations.")
    parser.add_argument("--cache", default="outputs/gpt2/gpt2_layer8_resid.pt")
    parser.add_argument("--output-dir", default="outputs/gpt2_sweep")
    parser.add_argument("--expansion-factors", default="4")
    parser.add_argument("--lambdas", default="0.0,0.5,1.0,2.0,3.0,5.0")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--beta-mode", choices=("profiled", "learned"), default="profiled"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = load_activation_cache(args.cache)[: args.max_tokens]
    rows: list[dict[str, float | str]] = []
    for expansion in _int_list(args.expansion_factors):
        width = int(expansion * x.shape[1])
        for lambda_value in _float_list(args.lambdas):
            model = VariationalGarroteSAE(
                VGSAEConfig(
                    input_dim=x.shape[1],
                    n_latents=width,
                    lambda_sparsity=lambda_value,
                    beta=args.beta,
                    beta_mode=args.beta_mode,
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
            obs = vg_sae_observables(model, x)
            rows.append(
                {
                    "beta_mode": args.beta_mode,
                    "beta": args.beta,
                    "expansion_factor": float(expansion),
                    "width": float(width),
                    "lambda": float(lambda_value),
                    "mse": obs.mse,
                    "rho": obs.rho,
                    "entropy": obs.entropy,
                    "v_eff": obs.v_eff,
                    "susceptibility": np.nan,
                    "dead_fraction": obs.dead_fraction,
                    "interference_energy": obs.interference_energy,
                    "variance_energy": obs.variance_energy,
                }
            )

    for expansion in sorted({int(row["expansion_factor"]) for row in rows}):
        idx = [i for i, row in enumerate(rows) if int(row["expansion_factor"]) == expansion]
        lambdas = np.array([rows[i]["lambda"] for i in idx])
        rhos = np.array([rows[i]["rho"] for i in idx])
        chi = susceptibility(lambdas, rhos)
        for i, value in zip(idx, chi, strict=True):
            rows[i]["susceptibility"] = float(value)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "gpt2_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(6, 4))
    for expansion in sorted({int(row["expansion_factor"]) for row in rows}):
        points = sorted(
            [row for row in rows if int(row["expansion_factor"]) == expansion],
            key=lambda row: row["lambda"],
        )
        plt.plot(
            [row["lambda"] for row in points],
            [row["v_eff"] for row in points],
            marker="o",
            label=f"{expansion}x",
        )
    plt.xlabel("lambda")
    plt.ylabel("V_eff")
    plt.title("GPT-2 L8 VG-SAE recoverability transition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gpt2_v_eff_vs_lambda.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
