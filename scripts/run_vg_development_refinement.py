"""Preregistered fixed-checkpoint conditional VG gate refinement pilot (P3).

Run from repository root. No training; fresh test is generated after cal freeze.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Callable

import numpy as np
import scipy
from scipy.optimize import nnls
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_vg_development_holdout import (
    SOURCES, prepare_model, sample_fixed_dictionary, save_csv, save_json,
    sha256_file, tensors, utc_now,
)
from src.sae_model import VariationalGarroteSAE
from src.sae_sweep import SweepConfig, make_train_test

HOLDOUT = ROOT / "outputs/vg_sae_development_20260921/holdout"
DEFAULT_OUTPUT = ROOT / "outputs/vg_sae_development_20260921/refinement"
PROTOCOL = ROOT / "idea-stage/runs/vg-sae-development-20260921/pilots/refinement_protocol.md"
SWEEPS = (0, 1, 3)
READOUTS = ("native_a", "count_a", "count_ma", "native_nnls", "count_nnls")
CAL_SEED, TEST_SEED = 2026092192, 2026092193
N_SAMPLES, BATCH_SIZE = 512, 128


def fixed_objective(x, m, a, decoder, bias, beta: float, gamma: float):
    """Per-example conditional objective; stable entropy including m in {0,1}."""
    residual = x - bias - (m * a) @ decoder.T
    residual_half = .5 * residual.square().sum(-1)
    variance_half = .5 * (m * (1 - m) * a.square() * decoder.square().sum(0)).sum(-1)
    negative_entropy = (torch.special.xlogy(m, m) + torch.special.xlogy(1-m, 1-m)).sum(-1)
    prior = gamma * m.sum(-1) + m.shape[1] * np.logaddexp(0., -gamma)
    gaussian = -.5 * x.shape[1] * math.log(beta / (2 * math.pi))
    return {
        "free_energy": beta * (residual_half + variance_half) + prior + negative_entropy + gaussian,
        "mean_residual_half_sse": residual_half,
        "bernoulli_variance_half_sse": variance_half,
        "stochastic_half_sse": residual_half + variance_half,
    }


@torch.no_grad()
def sequential_refinement(x, m, a, decoder, bias, beta, gamma, sweeps,
                          *, scores=None, after_coordinate: Callable | None = None):
    """Gauss-Seidel Bernoulli-coordinate minimization with a frozen amplitude."""
    if sweeps < 0:
        raise ValueError("sweeps must be nonnegative")
    m = m.clone()
    if scores is None:
        scores = torch.logit(m)
    else:
        scores = scores.clone()
    residual = x - bias - (m * a) @ decoder.T
    norms = decoder.square().sum(0)
    for sweep in range(sweeps):
        for j in range(m.shape[1]):
            old = m[:, j].clone()
            atom = decoder[:, j]
            # r currently excludes no coordinates: add this coordinate back.
            r_minus = residual + (old * a[:, j]).unsqueeze(1) * atom
            score = beta * (a[:, j] * (r_minus @ atom) - .5 * a[:, j].square() * norms[j]) - gamma
            updated = score.sigmoid()
            residual = r_minus - (updated * a[:, j]).unsqueeze(1) * atom
            m[:, j] = updated
            scores[:, j] = score
            if after_coordinate is not None:
                after_coordinate(sweep, j, m, residual)
    return m, scores, residual


def exact_count_mask(scores: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Stable descending score ranking, exact cardinality including K=0."""
    if scores.ndim != 2 or counts.shape != (scores.shape[0],):
        raise ValueError("Expected BxL scores and B counts")
    if bool(((counts < 0) | (counts > scores.shape[1])).any()):
        raise ValueError("Counts outside width")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranked = torch.arange(scores.shape[1], device=scores.device)[None, :] < counts[:, None]
    return torch.zeros_like(ranked).scatter(1, order, ranked)


def support_nnls(x: np.ndarray, decoder: np.ndarray, bias: np.ndarray,
                 support: np.ndarray) -> np.ndarray:
    """Truth-free identical float64 NNLS policy for each method/readout."""
    x, decoder, bias = (np.asarray(v, dtype=np.float64) for v in (x, decoder, bias))
    code = np.zeros(support.shape, dtype=np.float64)
    for i in range(len(x)):
        selected = np.flatnonzero(support[i])
        if selected.size:
            code[i, selected] = nnls(decoder[:, selected], x[i] - bias,
                                     maxiter=max(3 * len(selected), 1))[0]
    return code


def model_dictionary(model):
    if isinstance(model, VariationalGarroteSAE):
        d = model.decoder.weight
        b = model.pre_bias if model.pre_bias is not None else d.new_zeros(d.shape[0])
    else:
        d, b = model.W_dec.T, model.b_dec
    return d, b


@torch.no_grad()
def encode_state(model, x):
    if isinstance(model, VariationalGarroteSAE):
        scores, m, a, _ = model._encode_centered(model._center(x))
        return m, a, scores
    a = model.encode(x)
    return None, a, None


@torch.no_grad()
def infer_arm(model, x, sweep: int, readout: str, beta: float | None):
    d, b = model_dictionary(model)
    m, a, scores = encode_state(model, x)
    if m is not None:
        base_counts = (m >= .5).sum(1)
        if sweep:
            m, scores, _ = sequential_refinement(x, m, a, d, b, beta,
                model.config.lambda_sparsity, sweep, scores=scores)
        support = exact_count_mask(scores, base_counts) if readout.startswith("count_") else m >= .5
        code = (m*a if readout.endswith("_ma") else a) * support
    else:
        if sweep or readout not in ("native_a", "native_nnls"):
            raise ValueError("Unsupported baseline arm")
        support = a > 0
        code = a * support
    if readout.endswith("_nnls"):
        # Transfers are part of timed inference. Preserve NNLS result to float32
        # to match the original decoder and actual production reconstruction.
        fit = support_nnls(x.cpu().numpy(), d.cpu().numpy(), b.cpu().numpy(), support.cpu().numpy())
        code = torch.as_tensor(fit, device=x.device, dtype=x.dtype)
    reconstruction = model.decode(code)
    return code, support, reconstruction, m, a


@torch.no_grad()
def train_precision(model, train_x, batch_size=512):
    if model.config.beta_mode == "learned":
        return float(model.log_beta.exp()), {"source": "frozen_learned_beta"}
    d, b = model_dictionary(model)
    d64, b64 = d.double(), b.double()
    energy_sum = 0.
    for batch in train_x.split(batch_size):
        m, a, _ = encode_state(model, batch)
        terms = fixed_objective(batch.double(), m.double(), a.double(), d64, b64, 1., 0.)
        energy_sum += float(terms["stochastic_half_sse"].sum())
    beta = train_x.numel() / (2 * max(energy_sum, model.config.loss_eps))
    return beta, {"source": "original_train_global_expected_energy", "n_train": len(train_x),
                  "train_expected_half_sse_sum": energy_sum, "denominator_energy_min": model.config.loss_eps}


@torch.no_grad()
def evaluate_arm(model, data, matching, sweep, readout, beta):
    sums = dict(sse_z=0., target_z=0., sse_x=0., actual_count=0., candidate_count=0.,
                tp=0., fp=0., fn=0., free_energy=0., mean_residual_half_sse=0.,
                bernoulli_variance_half_sse=0., stochastic_half_sse=0., expected_l0=0.)
    device = data["x"].device
    li = torch.as_tensor(matching["learned_idx"], device=device)
    ti = torch.as_tensor(matching["true_idx"], device=device)
    signs = torch.as_tensor(matching["signs"], device=device)
    d, b = model_dictionary(model)
    for begin in range(0, len(data["x"]), BATCH_SIZE):
        x, z, truth = (data[k][begin:begin+BATCH_SIZE] for k in ("x", "z", "support"))
        code, candidate, recon, m, a = infer_arm(model, x, sweep, readout, beta)
        actual = code > 0
        aligned = code[:, li].double() * signs
        target, support_truth = z[:, ti].double(), truth[:, ti]
        support_pred = actual[:, li]
        sums["sse_z"] += float((aligned-target).square().sum())
        sums["target_z"] += float(target.square().sum())
        sums["sse_x"] += float((recon.double()-x.double()).square().sum())
        sums["actual_count"] += int(actual.sum())
        sums["candidate_count"] += int(candidate.sum())
        sums["tp"] += int((support_pred & support_truth).sum())
        sums["fp"] += int((support_pred & ~support_truth).sum())
        sums["fn"] += int((~support_pred & support_truth).sum())
        if m is not None:
            terms = fixed_objective(x.double(), m.double(), a.double(), d.double(), b.double(),
                                    beta, model.config.lambda_sparsity)
            for k, value in terms.items():
                sums[k] += float(value.sum())
            sums["expected_l0"] += float(m.double().sum())
    n, numel = len(data["x"]), data["x"].numel()
    centered_energy = float((data["x"].double()-data["x"].double().mean(0)).square().sum())
    metrics = {
        "n_samples": n, "hard_latent_relative_error": math.sqrt(sums["sse_z"]/sums["target_z"]),
        "hard_support_f1": 2*sums["tp"]/max(2*sums["tp"]+sums["fp"]+sums["fn"], 1.),
        "hard_reconstruction_mse": sums["sse_x"]/numel,
        "hard_explained_variance": 1-sums["sse_x"]/centered_energy,
        "actual_nonzero_l0": sums["actual_count"]/n,
        "candidate_support_l0": sums["candidate_count"]/n,
        "decoder_recovery_cosine": matching["matched_cosine"],
        "truth_l0": float(data["support"].sum())/n,
        "tp": sums["tp"], "fp": sums["fp"], "fn": sums["fn"],
    }
    if isinstance(model, VariationalGarroteSAE):
        metrics.update({f"conditional_{k}": sums[k]/n for k in ("free_energy",
            "mean_residual_half_sse", "bernoulli_variance_half_sse", "stochastic_half_sse", "expected_l0")})
        metrics["conditional_metrics_scope"] = "refined probability state with ORIGINAL a; independent of hard readout/refit"
    return metrics


@torch.no_grad()
def benchmark_arm(model, x, sweep, readout, beta):
    device = x.device
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    infer_arm(model, x, sweep, readout, beta)
    sync()
    samples = []
    for _ in range(5):
        sync()
        start = time.perf_counter()
        infer_arm(model, x, sweep, readout, beta)
        sync()
        samples.append(time.perf_counter()-start)
    return {"median_batch_seconds": float(np.median(samples)), "samples_seconds": samples,
            "batch_size": len(x), "warmup_repetitions": 1, "timed_repetitions": 5}


def checkpoint_choices():
    selected = json.loads((HOLDOUT / "frozen_selection.json").read_text())["selections"]
    choices = {}
    for row in selected:
        if (row["condition"] == "exponential" and row["l0_cap"] in (4,8) and row["supported"]
                and row["method"] in ("vgsae", "batchtopk", "jumprelu", "topk")):
            key = row["selected_run_id"]
            if key not in choices:
                choices[key] = {**row, "caps": []}
            choices[key]["caps"].append(row["l0_cap"])
    return list(choices.values())


def select_sweep(rows):
    candidates = [r for r in rows if r["method"] == "vgsae" and r["readout"] == "count_a"]
    base = next(r for r in candidates if r["sweeps"] == 0)
    feasible = [r for r in candidates if r["hard_support_f1"] >= base["hard_support_f1"]-.01]
    return min(feasible, key=lambda r: (r["hard_latent_relative_error"], r["sweeps"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-seconds", type=float, default=7200.)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(DEFAULT_OUTPUT):
        raise ValueError("Outputs must remain in assigned refinement directory")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise FileExistsError("Refusing to overwrite a preregistered run")
    start = time.monotonic()
    def deadline():
        if time.monotonic()-start > args.max_seconds:
            raise TimeoutError("P3 hard wall-time limit")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    choices = checkpoint_choices()
    protocol = {"created_at": utc_now(), "markdown": str(PROTOCOL.relative_to(ROOT)),
        "markdown_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
        "p1_frozen_selection_sha256": sha256_file(HOLDOUT/"frozen_selection.json"),
        "checkpoint_choices": choices, "calibration_seed": CAL_SEED, "test_seed": TEST_SEED,
        "n_calibration": N_SAMPLES, "n_test": N_SAMPLES, "batch_size": BATCH_SIZE,
        "sweeps": SWEEPS, "vg_readouts": READOUTS, "device": args.device,
        "max_seconds": args.max_seconds, "training_performed": False,
        "test_access_policy": "test generated after serialized calibration sweep choice"}
    save_json(out/"protocol.json", protocol)
    save_json(out/"environment.json", {"python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device), "threads": torch.get_num_threads()})
    cfg = SweepConfig.from_dict(json.loads((ROOT/SOURCES["exponential"]/"sweep_config.json").read_text()))
    source_file = HOLDOUT/"exponential_frozen_source.npz"
    source = np.load(source_file)
    dictionary, probabilities = source["dictionary"], source["feature_probabilities"]
    train, _ = make_train_test(cfg, 0, "cpu")
    np.testing.assert_array_equal(train.dictionary.numpy(), dictionary.astype(np.float32))
    np.testing.assert_array_equal(train.feature_probabilities.numpy(), probabilities.astype(np.float32))
    sample_kwargs = {"n_samples": N_SAMPLES, "amplitude_mode": cfg.data.amplitude_mode,
        "amplitude_scale": cfg.data.amplitude_scale, "noise_std": cfg.data.noise_std, "source_offset": 0}
    cal_array = sample_fixed_dictionary(dictionary, probabilities, base_seed=CAL_SEED, **sample_kwargs)
    np.savez_compressed(out/"calibration.npz", **cal_array)
    cal = tensors(cal_array, args.device)
    models, matchings, precisions, metadata, cal_rows, latencies = {}, {}, {}, [], [], []
    for choice in choices:
        deadline()
        key = choice["selected_run_id"]
        path = ROOT/choice["selected_checkpoint"]
        model, payload, matching = prepare_model(path, torch.as_tensor(dictionary, device=args.device), args.device)
        models[key], matchings[key] = model, matching
        is_vg = isinstance(model, VariationalGarroteSAE)
        if is_vg and (not model.config.use_variance_term or not model.config.use_entropy_term or model.config.entropy_weight != 1 or not model.config.nonnegative_amplitudes):
            raise ValueError("Coordinate formula requires full variance and unit entropy")
        beta, beta_meta = train_precision(model, train.x.to(args.device)) if is_vg else (None, {})
        precisions[key] = beta
        d, b = model_dictionary(model)
        meta = {"run_id": key, "method": choice["method"], "caps": choice["caps"],
            "checkpoint": choice["selected_checkpoint"], "checkpoint_sha256": sha256_file(path),
            "checkpoint_step": payload["step"], "model_parameter_count": sum(p.numel() for p in model.parameters()),
            "added_parameter_count": 0, "beta": beta, "beta_details": beta_meta,
            "decoder_squared_norm_min": float(d.square().sum(0).min()),
            "decoder_squared_norm_max": float(d.square().sum(0).max()),
            "inherited_calibration_control_count": choice["evaluated_control_count"]}
        metadata.append(meta)
        save_json(out/"checkpoint_manifest.json", {"checkpoints": metadata,
            "source_frozen_sha256": sha256_file(source_file), "source_config": cfg.to_dict()})
        np.savez_compressed(out/f"{key}_matching.npz", **matching)
        arms = itertools.product(SWEEPS, READOUTS) if is_vg else ((0,"native_a"),(0,"native_nnls"))
        for sweep, readout in arms:
            deadline()
            row = {"method": choice["method"], "run_id": key, "sweeps": sweep, "readout": readout,
                   **evaluate_arm(model, cal, matching, sweep, readout, beta)}
            cal_rows.append(row)
            timing = benchmark_arm(model, cal["x"][:BATCH_SIZE], sweep, readout, beta)
            latencies.append({"method": choice["method"], "run_id": key, "sweeps": sweep, "readout": readout,
                **timing, "extra_dense_decoder_products": 1 if sweep else 0,
                "extra_coordinate_dot_products": sweep*d.shape[1],
                "extra_coordinate_residual_rank1_updates": 2*sweep*d.shape[1],
                "nnls_solver_internal_products": "not exposed by scipy" if readout.endswith("nnls") else 0})
        print(f"calibration+timing {key} elapsed={time.monotonic()-start:.2f}s", flush=True)
    save_json(out/"checkpoint_manifest.json", {"checkpoints": metadata,
        "source_frozen_sha256": sha256_file(source_file), "source_config": cfg.to_dict()})
    save_csv(out/"calibration_metrics.csv", cal_rows)
    save_json(out/"calibration_metrics.json", cal_rows)
    save_json(out/"latencies.json", latencies)
    selected = select_sweep(cal_rows)
    selection = {"frozen_at": utc_now(), "test_generated": False,
        "rule": "minimum calibration count_a error with F1 >= sweep0 F1-.01; tie smaller sweep",
        "selected_sweeps": selected["sweeps"], "selected_readout": "count_a", "selected_calibration_row": selected,
        "calibration_metrics_sha256": sha256_file(out/"calibration_metrics.json")}
    save_json(out/"frozen_selection.json", selection)
    freeze_hash = sha256_file(out/"frozen_selection.json")
    print(f"P3 selection frozen: sweep{selected['sweeps']} count_a; now generating test", flush=True)
    test_array = sample_fixed_dictionary(dictionary, probabilities, base_seed=TEST_SEED, **sample_kwargs)
    np.savez_compressed(out/"test.npz", **test_array)
    test = tensors(test_array, args.device)
    rows = []
    for choice in choices:
        key = choice["selected_run_id"]
        model = models[key]
        arms = itertools.product(SWEEPS, READOUTS) if isinstance(model, VariationalGarroteSAE) else ((0,"native_a"),(0,"native_nnls"))
        for sweep, readout in arms:
            deadline()
            rows.append({"method": choice["method"], "run_id": key, "sweeps": sweep, "readout": readout,
                "is_cal_selected_policy": choice["method"] == "vgsae" and sweep == selected["sweeps"] and readout == "count_a",
                **evaluate_arm(model, test, matchings[key], sweep, readout, precisions[key])})
        print(f"test {key} elapsed={time.monotonic()-start:.2f}s", flush=True)
    if sha256_file(out/"frozen_selection.json") != freeze_hash:
        raise RuntimeError("Calibration freeze changed")
    save_csv(out/"test_metrics.csv", rows)
    save_json(out/"test_metrics.json", rows)
    save_json(out/"completion.json", {"completed_at": utc_now(), "status": "complete",
        "wall_seconds": time.monotonic()-start, "device_allocation_hours": (time.monotonic()-start)/3600,
        "unique_checkpoints": len(choices), "test_arms": len(rows), "selection_sha256": freeze_hash,
        "script_sha256": sha256_file(Path(__file__)), "test_selected": False,
        "test_npz_sha256": sha256_file(out/"test.npz"), "calibration_npz_sha256": sha256_file(out/"calibration.npz")})


if __name__ == "__main__":
    main()
