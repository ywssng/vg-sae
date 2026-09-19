"""Known-model B05 pilot: exact posterior, current VG gate, and mean-field.

The registered protocol fixes every training and inference budget. No learned
dictionary/amplitude or real-activation calibration claims follow from this run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sae_model import VGSAEConfig, VariationalGarroteSAE

PI = .15
BETA = 25.
GAMMA = math.log((1 - PI) / PI)
SWEEPS = 100


def supports(k: int, *, dtype=torch.float64) -> torch.Tensor:
    return ((torch.arange(2**k)[:, None] >> torch.arange(k)) & 1).to(dtype)


def exact_posterior(x: torch.Tensor, d: torch.Tensor, beta=BETA, pi=PI) -> dict:
    """Enumerate the normalized generative joint, independently of VG loss."""
    states = supports(d.shape[1], dtype=x.dtype).to(x.device)
    signal = states @ d.T
    sq_error = (x[:, None, :] - signal[None, :, :]).square().sum(-1)
    log_prior = (states * math.log(pi) + (1 - states) * math.log1p(-pi)).sum(-1)
    log_joint = -.5 * beta * sq_error + .5 * x.shape[1] * math.log(beta / (2 * math.pi)) + log_prior
    log_evidence = torch.logsumexp(log_joint, -1)
    log_p = log_joint - log_evidence[:, None]
    p = log_p.exp()
    m = p @ states
    signal_mean = m @ d.T
    signal_variance = (p @ signal.square().sum(-1) - signal_mean.square().sum(-1)).clamp_min(0)
    return dict(states=states, log_joint=log_joint, log_evidence=log_evidence,
                log_p=log_p, p=p, m=m, signal_variance=signal_variance)


def meanfield_energy(x: torch.Tensor, d: torch.Tensor, m: torch.Tensor,
                     beta=BETA, pi=PI) -> torch.Tensor:
    """Full normalized negative ELBO, one scalar per sample."""
    expected_sq_error = (x - m @ d.T).square().sum(-1) + (m * (1 - m) * d.square().sum(0)).sum(-1)
    prior = -(m * math.log(pi) + (1 - m) * math.log1p(-pi)).sum(-1)
    neg_entropy = (torch.special.xlogy(m, m) + torch.special.xlogy(1 - m, 1 - m)).sum(-1)
    return .5 * beta * expected_sq_error - .5 * x.shape[1] * math.log(beta / (2 * math.pi)) + prior + neg_entropy


def fixed_point(x: torch.Tensor, d: torch.Tensor, m: torch.Tensor,
                beta=BETA, pi=PI) -> torch.Tensor:
    gram = d.T @ d
    diag = gram.diagonal()
    return torch.sigmoid(beta * (x @ d - m @ gram + m * diag) - .5 * beta * diag - math.log((1 - pi) / pi))


def coordinate_meanfield(x: torch.Tensor, d: torch.Tensor, start: torch.Tensor,
                         sweeps=SWEEPS, beta=BETA, pi=PI) -> tuple[torch.Tensor, torch.Tensor]:
    m = start.clone()
    gram = d.T @ d
    projected = x @ d
    gamma = math.log((1 - pi) / pi)
    for _ in range(sweeps):
        for j in range(d.shape[1]):
            excluding_j = projected[:, j] - m @ gram[:, j] + m[:, j] * gram[j, j]
            m[:, j] = torch.sigmoid(beta * excluding_j - .5 * beta * gram[j, j] - gamma)
    residual = (m - fixed_point(x, d, m, beta, pi)).abs().amax(-1)
    return m, residual


def make_oracle_model(d: torch.Tensor) -> VariationalGarroteSAE:
    model = VariationalGarroteSAE(VGSAEConfig(
        input_dim=d.shape[0], n_latents=d.shape[1], beta=BETA,
        lambda_sparsity=GAMMA, beta_mode="learned", decoder_bias=False,
        normalize_decoder=False, dtype=d.dtype,
    )).to(d.device)
    with torch.no_grad():
        model.decoder.weight.copy_(d)
        model.amplitude_encoder.weight.zero_()
        model.amplitude_encoder.bias.fill_(math.log(math.expm1(1.)))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.gate_encoder.parameters():
        p.requires_grad_(True)
    return model


def train_gate(d: torch.Tensor, x: torch.Tensor, device: str) -> tuple[dict, list[dict], dict]:
    torch.manual_seed(0)
    model = make_oracle_model(d.to(device=device, dtype=torch.float32))
    x = x.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.gate_encoder.parameters(), lr=.003, weight_decay=0.)
    rng = torch.Generator().manual_seed(0)
    history = []
    order = torch.empty(0, dtype=torch.long)
    offset = 0
    for step in range(1500):
        if offset >= len(order):
            order = torch.randperm(len(x), generator=rng)
            offset = 0
        idx = order[offset:offset + 256].to(device)
        offset += 256
        optimizer.zero_grad(set_to_none=True)
        loss = model.free_energy(x[idx])["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.gate_encoder.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == 1499:
            with torch.no_grad():
                full = model.free_energy(x)
            history.append(dict(step=step + 1, train_free_energy=float(full["loss"]), minibatch_free_energy=float(loss.detach())))
    weights = {k: v.detach().cpu().double() for k, v in model.state_dict().items()}
    checks = dict(
        decoder_max_error=float((weights["decoder.weight"] - d).abs().max()),
        amplitude_weight_max_abs=float(weights["amplitude_encoder.weight"].abs().max()),
        amplitude_value=float(torch.nn.functional.softplus(weights["amplitude_encoder.bias"])[0]),
        frozen_beta=float(weights["log_beta"].exp()),
        trained_parameter_names=[k for k, p in model.named_parameters() if p.requires_grad],
    )
    return weights, history, checks


def score_method(x, truth, d, exact, m, name, residual):
    safe = m.clamp(1e-15, 1 - 1e-15)
    marginal_nll = -(truth * safe.log() + (1 - truth) * torch.log1p(-safe)).sum(-1)
    mf_f = meanfield_energy(x, d, m)
    if name == "exact_joint":
        idx = (truth.long() * (2**torch.arange(truth.shape[1]))).sum(-1)
        joint_nll = -exact["log_p"][torch.arange(len(x)), idx]
        f = -exact["log_evidence"]
    else:
        joint_nll = marginal_nll
        f = mf_f
    brier = (m - truth).square()
    mean_signal = m @ d.T
    hard_signal = (m > .5).to(m.dtype) @ d.T
    clean = truth @ d.T
    exact_signal = exact["m"] @ d.T
    dim = x.shape[1]
    per = dict(
        brier=brier.mean(-1),
        marginal_nll_per_atom=marginal_nll / m.shape[1],
        joint_support_nll=joint_nll,
        marginal_mse_to_exact=(m - exact["m"]).square().mean(-1),
        posterior_kl=f + exact["log_evidence"],
        free_energy=f,
        product_marginal_free_energy=mf_f,
        product_marginal_posterior_kl=mf_f + exact["log_evidence"],
        predicted_prevalence=m.mean(-1),
        sampled_prevalence=truth.mean(-1),
        mean_observed_mse=(x - mean_signal).square().mean(-1),
        hard_observed_mse=(x - hard_signal).square().mean(-1),
        q_expected_observed_mse=((x - mean_signal).square().sum(-1) + (m * (1 - m) * d.square().sum(0)).sum(-1)) / dim,
        mean_sampled_clean_mse=(clean - mean_signal).square().mean(-1),
        hard_sampled_clean_mse=(clean - hard_signal).square().mean(-1),
        mean_posterior_clean_risk=((mean_signal - exact_signal).square().sum(-1) + exact["signal_variance"]) / dim,
        hard_posterior_clean_risk=((hard_signal - exact_signal).square().sum(-1) + exact["signal_variance"]) / dim,
        fixed_point_residual=residual,
    )
    # For the exact joint E[||x-Ds||^2] contains posterior covariances.
    if name == "exact_joint":
        per["q_expected_observed_mse"] = ((x - exact_signal).square().sum(-1) + exact["signal_variance"]) / dim
    summary = {key: float(value.mean()) for key, value in per.items()}
    summary.update(
        brier_active=float(brier[truth.bool()].mean()),
        brier_inactive=float(brier[~truth.bool()].mean()),
        active_prevalence=float(truth.mean()),
        fixed_point_residual_max=float(residual.max()),
        marginal_max_abs_to_exact=float((m - exact["m"]).abs().max()),
    )
    return summary, per


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/idea_discovery_20260919/posterior"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    start_time = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("Existing pilot evidence is not overwritten; choose a fresh output directory.")
    rng = torch.Generator().manual_seed(20260919)
    all_truth = (torch.rand(10240, 8, generator=rng, dtype=torch.float64) < PI).double()
    noise = .2 * torch.randn(10240, 8, generator=rng, dtype=torch.float64)
    truth = all_truth[8192:]
    summaries, samples, atoms, histories = [], [], [], []
    results, raw = {}, {}
    for condition in ("orthogonal", "coherent095"):
        d = torch.eye(8, dtype=torch.float64)
        if condition == "coherent095":
            d[:, 1] = 0.
            d[0, 1], d[1, 1] = .95, math.sqrt(1 - .95**2)
        data = all_truth @ d.T + noise
        train, x = data[:8192], data[8192:]
        weights, history, checks = train_gate(d, train, args.device)
        histories.extend(dict(condition=condition, **row) for row in history)
        amortized = torch.sigmoid(x @ weights["gate_encoder.weight"].T + weights["gate_encoder.bias"])
        exact = exact_posterior(x, d)
        random_start = torch.rand(x.shape, generator=torch.Generator().manual_seed(20260920), dtype=x.dtype)
        starts = {"prior": torch.full_like(x, PI), "amortized": amortized, "random": random_start}
        candidates, residuals, energies = [], [], []
        for name, initial in starts.items():
            m, residual = coordinate_meanfield(x, d, initial)
            candidates.append(m)
            residuals.append(residual)
            energies.append(meanfield_energy(x, d, m))
        energies = torch.stack(energies, 1)
        chosen = energies.argmin(1)
        refined = torch.stack(candidates, 1)[torch.arange(len(x)), chosen]
        refined_residual = torch.stack(residuals, 1)[torch.arange(len(x)), chosen]
        methods = dict(exact_joint=exact["m"], amortized=amortized, refined_meanfield=refined)
        for name, m in methods.items():
            residual = (m - fixed_point(x, d, m)).abs().amax(-1)
            summary, per_sample = score_method(x, truth, d, exact, m, name, residual)
            summaries.append(dict(condition=condition, method=name, **summary))
            for i in range(len(x)):
                samples.append(dict(condition=condition, method=name, sample=i, **{k: float(v[i]) for k, v in per_sample.items()}))
        for i in range(len(x)):
            for j in range(8):
                atoms.append(dict(condition=condition, sample=i, atom=j, truth=int(truth[i, j]), exact=float(exact["m"][i, j]), amortized=float(amortized[i, j]), refined=float(refined[i, j])))
        analytic = torch.sigmoid(BETA * (x @ d) - .5 * BETA - GAMMA)
        mf_kl = meanfield_energy(x, d, refined) + exact["log_evidence"]
        results[condition] = dict(
            frozen_parameter_checks=checks,
            weights={key: value.tolist() for key, value in weights.items()},
            selected_starts={name: int((chosen == i).sum()) for i, name in enumerate(starts)},
            final_start_mean_free_energy={name: float(energies[:, i].mean()) for i, name in enumerate(starts)},
            final_start_max_residual={name: float(residuals[i].max()) for i, name in enumerate(starts)},
            refined_max_fixed_point_residual=float(refined_residual.max()),
            orthogonal_analytic_max_error=float((exact["m"] - analytic).abs().max()) if condition == "orthogonal" else None,
            exact_bayes_atom_error=float(torch.minimum(exact["m"], 1 - exact["m"]).mean()),
            exact_joint_entropy=float(-(exact["p"] * exact["log_p"]).sum(-1).mean()),
            exact_pair_covariance_mean=float(((exact["p"] * (exact["states"][:, 0] * exact["states"][:, 1])).sum(-1) - exact["m"][:, 0] * exact["m"][:, 1]).mean()),
            refined_mean_abs_kl=float(mf_kl.abs().mean()),
        )
        raw.update({f"{condition}_{key}": value.numpy() for key, value in dict(dictionary=d, train_x=train, train_truth=all_truth[:8192], test_x=x, test_truth=truth, exact_p=exact["p"], exact_m=exact["m"], amortized_m=amortized, refined_m=refined, all_start_free_energy=energies, chosen_start=chosen).items()})
        print(json.dumps({"condition": condition, "finished": True, "seconds": time.perf_counter() - start_time}), flush=True)
    indexed = {(row["condition"], row["method"]): row for row in summaries}
    a, r = [indexed[("coherent095", name)] for name in ("amortized", "refined_meanfield")]
    reduction = 1 - r["marginal_mse_to_exact"] / max(a["marginal_mse_to_exact"], 1e-30)
    f_drop = a["free_energy"] - r["free_energy"]
    converged = r["fixed_point_residual_max"] < 1e-6
    valid = results["orthogonal"]["orthogonal_analytic_max_error"] < 1e-10 and results["orthogonal"]["refined_mean_abs_kl"] < 1e-8
    if not valid:
        verdict = "invalid_control"
    elif converged and f_drop >= .1 and reduction >= .25 and r["brier"] <= a["brier"] + .002:
        verdict = "positive"
    elif converged and f_drop < .01 and reduction < .05:
        verdict = "negative"
    else:
        verdict = "null_or_inconclusive"
    protocol = ROOT / "idea-stage/pilots/posterior_protocol.md"
    payload = dict(
        config=dict(data_seed=20260919, optimizer_seed=0, random_start_seed=20260920, train_samples=8192, test_samples=2048, steps=1500, lr=.003, batch_size=256, sweeps=SWEEPS, beta=BETA, pi=PI, amplitude=1, d=8, k=8, device=args.device, train_dtype="float32", evaluation_dtype="float64", deterministic_algorithms=True),
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else None),
        provenance=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest()),
        conditions=results, summary=summaries,
        preregistered_verdict=dict(control_valid=valid, refinement=verdict, coherent_free_energy_drop=f_drop, coherent_marginal_mse_fraction_reduction=reduction, coherent_brier_change=r["brier"] - a["brier"], achieved_meanfield_limitation=valid and converged and r["posterior_kl"] >= .05 and r["marginal_mse_to_exact"] >= .001),
        elapsed_seconds=time.perf_counter() - start_time,
        peak_cuda_memory_mib=torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else 0.,
        limitations=["One seed; fixed budget, no global mean-field certificate.", "Amortization gap includes optimization and finite-data effects.", "Known D, amplitude and beta only; no learned-model calibration or OOD claims."],
    )
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "samples.csv", samples)
    write_csv(out / "marginals.csv", atoms)
    write_csv(out / "training.csv", histories)
    np.savez_compressed(out / "raw_arrays.npz", **raw)
    (out / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["preregistered_verdict"]), flush=True)


if __name__ == "__main__":
    main()
