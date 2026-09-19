"""Numerically audit a constructed VG-SAE decoder-replication family on CPU.

This script evaluates known parameters; it does not optimize or train a model.
See idea-stage/pilots/replication_protocol.md for the preregistered scope.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Literal

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sae_model import VGSAEConfig, VariationalGarroteSAE


WIDTHS = (4, 16, 64, 256)
PRIOR_RULES = ("fixed_gamma", "fixed_expected_count")
BETA_MODES = ("learned", "profiled")
TOLERANCE = 1e-10


def build_replication_case(
    width: int,
    prior_rule: Literal["fixed_gamma", "fixed_expected_count"],
    beta_mode: Literal["learned", "profiled"],
    *,
    input_dim: int = 8,
    signal_amplitude: float = 1.0,
    beta: float = 2.0,
    fixed_gamma: float = 2.0,
    kappa: float = 1.0,
) -> tuple[VariationalGarroteSAE, torch.Tensor, float]:
    """Create actual encoder/decoder parameters realizing the analytic family."""
    if width <= 0 or input_dim <= 0 or signal_amplitude <= 0:
        raise ValueError("Width, input dimension, and signal amplitude must be positive.")
    if prior_rule == "fixed_gamma":
        gamma = fixed_gamma
        pi = 1.0 / (1.0 + math.exp(gamma))
    elif prior_rule == "fixed_expected_count":
        if not 0 < kappa < width:
            raise ValueError("Expected prior count must lie strictly between zero and width.")
        pi = kappa / width
        gamma = math.log((width - kappa) / kappa)
    else:
        raise ValueError(f"Unknown prior rule: {prior_rule}")
    if not 0 < pi < 0.5:
        raise ValueError("This construction requires a prior strictly below the hard threshold.")

    model = VariationalGarroteSAE(
        VGSAEConfig(
            input_dim=input_dim,
            n_latents=width,
            beta=beta,
            lambda_sparsity=gamma,
            beta_mode=beta_mode,
            decoder_bias=False,
            dtype=torch.float64,
        )
    ).cpu()
    a = signal_amplitude / (width * pi)
    # Numerically stable inverse softplus for positive a, avoiding exp(a).
    amplitude_bias = a + math.log(-math.expm1(-a))
    with torch.no_grad():
        model.decoder.weight.zero_()
        model.decoder.weight[0, :].fill_(1.0)
        model.gate_encoder.weight.zero_()
        model.gate_encoder.bias.fill_(-gamma)
        model.amplitude_encoder.weight.zero_()
        model.amplitude_encoder.bias.fill_(amplitude_bias)
    model.eval()
    x = torch.zeros(1, input_dim, dtype=torch.float64)
    x[0, 0] = signal_amplitude
    return model, x, pi


@torch.no_grad()
def evaluate_case(width: int, prior_rule: str, beta_mode: str) -> tuple[dict, dict]:
    model, x, pi = build_replication_case(width, prior_rule, beta_mode)
    result = model.free_energy(x)
    mask, _, hard_code = model.encode_inference(x)
    hard_output = model.decode(hard_code)
    expected_variance = (1.0 - pi) / (2.0 * width * pi)
    expected_beta_eff = 2.0 if beta_mode == "learned" else 8.0 / (2.0 * expected_variance)
    gaussian_constant = -4.0 * math.log(2.0 / (2.0 * math.pi))
    expected_loss = (
        2.0 * expected_variance + gaussian_constant
        if beta_mode == "learned"
        else 4.0 * math.log(2.0 * expected_variance / 8.0)
    )
    row = {
        "prior_rule": prior_rule,
        "beta_mode": beta_mode,
        "width": width,
        "input_dim": 8,
        "signal_amplitude": 1.0,
        "gamma": model.config.lambda_sparsity,
        "prior_probability": pi,
        "prior_expected_count": width * pi,
        "latent_amplitude": float(result["a"][0, 0]),
        "expected_latent_amplitude": 1.0 / (width * pi),
        "gate_probability_max_error": float((result["m"] - pi).abs().max()),
        "amplitude_max_error": float((result["a"] - 1.0 / (width * pi)).abs().max()),
        "decoder_unit_norm_max_error": float((model.decoder_column_sqnorms() - 1.0).abs().max()),
        "posterior_mean_max_error": float((result["x_hat"] - x).abs().max()),
        "recon": float(result["recon"]),
        "variance": float(result["variance"]),
        "expected_variance": expected_variance,
        "energy": float(result["energy"]),
        "prior": float(result["prior"]),
        "entropy": float(result["entropy"]),
        "kl": float(result["prior"] - result["entropy"]),
        "loss": float(result["loss"]),
        "expected_loss": expected_loss,
        "absolute_loss_error": abs(float(result["loss"]) - expected_loss),
        "beta_eff": float(result["beta_eff"]),
        "expected_beta_eff": expected_beta_eff,
        "posterior_expected_count": float(result["sparsity"]),
        "hard_active_count": int(mask.sum()),
        "soft_nmse": float((result["x_hat"] - x).square().sum() / x.square().sum()),
        "hard_nmse": float((hard_output - x).square().sum() / x.square().sum()),
        "profiled_clamps_inactive": bool(
            expected_variance > model.config.loss_eps
            and 2.0 * expected_variance / 8.0 > model.config.loss_eps
        ),
    }
    config = asdict(model.config)
    config["dtype"] = str(model.config.torch_dtype)
    return row, config


def check_results(rows: list[dict]) -> dict[str, bool]:
    close = lambda left, right: math.isclose(left, right, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
    checks = {
        "complete_grid": len(rows) == len(WIDTHS) * len(PRIOR_RULES) * len(BETA_MODES),
        "variance_matches_closed_form": all(close(r["variance"], r["expected_variance"]) for r in rows),
        "energy_matches_closed_form": all(close(r["energy"], r["expected_variance"]) for r in rows),
        "loss_matches_closed_form": all(close(r["loss"], r["expected_loss"]) for r in rows),
        "beta_matches_closed_form": all(close(r["beta_eff"], r["expected_beta_eff"]) for r in rows),
        "zero_kl": all(abs(r["kl"]) <= TOLERANCE for r in rows),
        "unit_decoder_columns": all(r["decoder_unit_norm_max_error"] <= TOLERANCE for r in rows),
        "posterior_mean_exact": all(r["posterior_mean_max_error"] <= TOLERANCE for r in rows),
        "encoder_realizes_construction": all(
            r["gate_probability_max_error"] <= TOLERANCE and r["amplitude_max_error"] <= TOLERANCE
            for r in rows
        ),
        "hard_selects_nothing": all(r["hard_active_count"] == 0 for r in rows),
        "hard_nmse_one": all(close(r["hard_nmse"], 1.0) for r in rows),
        "profiled_clamps_inactive": all(r["profiled_clamps_inactive"] for r in rows),
    }
    fixed = sorted(
        (r for r in rows if r["prior_rule"] == "fixed_gamma" and r["beta_mode"] == "profiled"),
        key=lambda r: r["width"],
    )
    matched = sorted(
        (r for r in rows if r["prior_rule"] == "fixed_expected_count" and r["beta_mode"] == "profiled"),
        key=lambda r: r["width"],
    )
    checks["fixed_gamma_inverse_width_variance"] = all(
        close(b["variance"] / a["variance"], a["width"] / b["width"])
        for a, b in zip(fixed, fixed[1:])
    )
    checks["fixed_gamma_profiled_log_width_decrease"] = all(
        close(b["loss"] - a["loss"], -4.0 * math.log(b["width"] / a["width"]))
        for a, b in zip(fixed, fixed[1:])
    )
    checks["matched_count_is_one"] = all(close(r["posterior_expected_count"], 1.0) for r in matched)
    checks["matched_count_variance_increases_toward_half"] = all(
        a["variance"] < b["variance"] < 0.5 for a, b in zip(matched, matched[1:])
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/idea_discovery_20260919/replication"))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    allowed_output_root = ROOT / "outputs" / "idea_discovery_20260919" / "replication"
    if not output_dir.is_relative_to(allowed_output_root):
        parser.error("Output directory must stay under outputs/idea_discovery_20260919/replication.")
    if any((output_dir / name).exists() for name in ("results.json", "results.csv")):
        parser.error("Refusing to overwrite existing evidence; choose a fresh output subdirectory.")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    rows, configs = [], []
    for prior_rule in PRIOR_RULES:
        for beta_mode in BETA_MODES:
            for width in WIDTHS:
                row, config = evaluate_case(width, prior_rule, beta_mode)
                rows.append(row)
                configs.append({"prior_rule": prior_rule, "config": config})
    checks = check_results(rows)
    relative_files = [
        "src/sae_model.py",
        "scripts/idea_discovery_replication_pilot.py",
        "tests/test_idea_discovery_replication.py",
        "idea-stage/pilots/replication_protocol.md",
    ]
    command = shlex.join([".venv/bin/python", "-B", str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]])
    payload = {
        "schema_version": 1,
        "pilot": "decoder_replication_constructed_feasibility",
        "status": "pass" if all(checks.values()) else "fail",
        "started_at_utc": started_at,
        "runtime_seconds": time.perf_counter() - started,
        "seed": 0,
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "gpu_hours": 0,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in relative_files},
        "commands": {
            "pilot": command,
            "tests": ".venv/bin/python -B -m pytest -p no:cacheprovider tests/test_idea_discovery_replication.py -q",
        },
        "tolerance": {"absolute": TOLERANCE, "relative": TOLERANCE},
        "checks": checks,
        "configurations": configs,
        "limitations": [
            "Constructed parameters establish feasibility; no training or optimization behavior was measured.",
            "Decoder bias is disabled; a free bias trivially fits this one constant input. Centered nonconstant data were not tested.",
            "Dropout decoder-replication reduction is established in Cavazza et al. (AISTATS 2018); this is a repository-specific audit, not a novelty claim.",
            "Matched prior expected count changes gamma with width and only blocks this constructed 1/r decrease.",
            "Absolute learned/profiled losses differ by Gaussian normalization conventions.",
            "Finite-epsilon profiled implementation eventually has a floor; tested cases do not activate clamps.",
            "A single constant positive signal direction does not establish real-data feature-recovery behavior.",
        ],
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": payload["status"], "cases": len(rows), "checks": checks, "runtime_seconds": payload["runtime_seconds"], "output_dir": str(output_dir.relative_to(ROOT))}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
