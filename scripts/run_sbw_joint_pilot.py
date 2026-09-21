"""P3 fixed-precision, point-amplitude conditional-support mechanism pilot.

No runs on import. Test data are constructed only after selection is frozen.
The optimized product arm is a three-start local solver, not a global certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sbw_pilot_utils import geometry_metrics, orthogonal_dictionary, sample_toy, tensor_hash
from src.sae_model import VGSAEConfig, VariationalGarroteSAE

ARMS = ("amortized", "optimized_mf", "exact32")
WORLDS = (210, 211, 212)
RHOS = (-.4, .4)
GAMMAS = (0., 4., 8.)
BETA = 10.


def states_for(width: int, like: torch.Tensor) -> torch.Tensor:
    """Ascending binary-index states; coordinate zero is the least significant bit."""
    integers = torch.arange(2**width, device=like.device)
    shifts = torch.arange(width, device=like.device)
    return ((integers[:, None] >> shifts) & 1).to(like.dtype)


def state_risks(y: torch.Tensor, a: torch.Tensor, decoder: torch.Tensor,
                states: torch.Tensor) -> torch.Tensor:
    reconstruction = (a[:, None, :] * states[None]) @ decoder.T
    return (y[:, None, :] - reconstruction).square().sum(-1)


def exact_posterior(y: torch.Tensor, a: torch.Tensor, decoder: torch.Tensor,
                    beta: float, gamma: float) -> dict:
    states = states_for(a.shape[-1], a)
    risks = state_risks(y, a, decoder, states)
    energy = .5 * beta * risks + gamma * states.sum(-1)
    q = torch.softmax(-energy, dim=-1)
    normalizer = a.shape[-1] * F.softplus(a.new_tensor(-gamma))
    gaussian = -.5 * y.shape[-1] * math.log(beta / (2 * math.pi))
    return {"free_energy": -torch.logsumexp(-energy, dim=-1) + normalizer + gaussian,
            "q": q, "m": q @ states, "states": states, "risk": (q*risks).sum(-1),
            "map": states[energy.argmin(-1)], "energy": energy}


def product_risk(y: torch.Tensor, a: torch.Tensor, decoder: torch.Tensor,
                 m: torch.Tensor) -> torch.Tensor:
    mean_risk = (y - (m*a) @ decoder.T).square().sum(-1)
    variance = (m*(1-m)*a.square()*decoder.square().sum(0)).sum(-1)
    return mean_risk + variance


def product_free_energy(y: torch.Tensor, a: torch.Tensor, decoder: torch.Tensor,
                        m: torch.Tensor, beta: float, gamma: float) -> torch.Tensor:
    # xlogy defines 0 log 0 = 0; solved q is detached for training, so endpoint
    # derivatives of xlogy with respect to q are intentionally not used.
    neg_entropy = (torch.xlogy(m, m) + torch.xlogy(1-m, 1-m)).sum(-1)
    prior = gamma*m.sum(-1) + m.shape[-1]*F.softplus(m.new_tensor(-gamma))
    gaussian = -.5*y.shape[-1]*math.log(beta/(2*math.pi))
    return .5*beta*product_risk(y, a, decoder, m) + neg_entropy + prior + gaussian


def mf_statistics(y, a, decoder, beta, gamma):
    gram = decoder.T @ decoder
    hessian = a[:, :, None]*a[:, None, :]*gram
    diagonal = hessian.diagonal(dim1=-2, dim2=-1)
    linear = beta*a*(y @ decoder) - .5*beta*diagonal - gamma
    off = beta*(hessian - torch.diag_embed(diagonal))
    return linear, off


def coordinate_logits(m, linear, off):
    return linear - torch.einsum("...bj,bjk->...bk", m, off)


@torch.no_grad()
def solve_mean_field(y: torch.Tensor, a: torch.Tensor, decoder: torch.Tensor,
                     beta: float, gamma: float, *, max_sweeps: int = 50,
                     iterate_tol: float = 1e-5, stationarity_tol: float = 5e-5,
                     keep_history: bool = False) -> dict:
    linear, off = mf_statistics(y, a, decoder, beta, gamma)
    m = a.new_tensor((.1, .5, .9))[:, None, None].expand(3, *a.shape).clone()
    history = [product_free_energy(y, a, decoder, m, beta, gamma)] if keep_history else []
    for sweep in range(max_sweeps):
        old = m.clone()
        for j in range(a.shape[-1]):
            # New coordinates are immediately visible to the next coordinate.
            logit = linear[:, j] - (m*off[None, :, :, j]).sum(-1)
            m[:, :, j] = logit.sigmoid()
        delta = (m-old).abs().amax(-1)
        residual = (m-coordinate_logits(m, linear, off).sigmoid()).abs().amax(-1)
        converged = (delta < iterate_tol) & (residual < stationarity_tol)
        if keep_history:
            history.append(product_free_energy(y, a, decoder, m, beta, gamma))
        if bool(converged.all()):
            break
    objective = product_free_energy(y, a, decoder, m, beta, gamma)
    winning_start = objective.argmin(0)
    index = torch.arange(a.shape[0], device=a.device)
    return {"m": m[winning_start, index], "converged": converged[winning_start, index],
            "delta": delta[winning_start, index], "residual": residual[winning_start, index],
            "winning_start": winning_start, "per_start_converged": converged,
            "per_start_delta": delta, "per_start_residual": residual,
            "sweeps": sweep+1, "objective": objective[winning_start, index],
            "history": history}


def make_model(arm: str, gamma: float, world: int, device: str = "cpu"):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(world)
    model = VariationalGarroteSAE(VGSAEConfig(
        input_dim=20, n_latents=5, beta=BETA, beta_mode="learned",
        lambda_sparsity=gamma, gate_bias_init=-2., amplitude_bias_init=0.,
        decoder_bias=True, normalize_decoder=True)).to(device)
    model.log_beta.requires_grad_(False)
    if arm != "amortized":
        model.gate_encoder.requires_grad_(False)
    return model


def conditional_inputs(model, x):
    y = x-model.pre_bias
    a = F.softplus(model.amplitude_encoder(y))
    return y, a, model.decoder.weight


def training_loss(model, arm, x):
    gamma = model.config.lambda_sparsity
    if arm == "amortized":
        return model.free_energy(x)["loss"], None
    y, a, decoder = conditional_inputs(model, x)
    if arm == "exact32":
        return exact_posterior(y, a, decoder, BETA, gamma)["free_energy"].mean(), None
    solved = solve_mean_field(y, a, decoder, BETA, gamma)
    # Retain all paired examples. For nonconverged q this is a fixed-q partial
    # derivative, and must not be described as an optimized envelope gradient.
    value = product_free_energy(y, a, decoder, solved["m"].detach(), BETA, gamma)
    return value.mean(), solved


def solver_summary(parts: list[dict]) -> dict:
    if not parts:
        return {}
    selected = torch.cat([p["converged"].cpu() for p in parts])
    per_start = torch.cat([p["per_start_converged"].cpu() for p in parts], dim=1)
    winning = torch.cat([p["winning_start"].cpu() for p in parts])
    return {"sample_count": len(selected), "failed_samples": int((~selected).sum()),
            "sample_convergence_fraction": float(selected.float().mean()),
            "batch_count": len(parts), "failed_batches": sum(not bool(p["converged"].all()) for p in parts),
            "batch_convergence_fraction": sum(bool(p["converged"].all()) for p in parts)/len(parts),
            "per_start_convergence_fraction": per_start.float().mean(1).tolist(),
            "winning_start_counts": torch.bincount(winning, minlength=3).tolist(),
            "max_delta": max(float(p["delta"].max()) for p in parts),
            "max_stationarity_residual": max(float(p["residual"].max()) for p in parts),
            "mean_full_sweeps": sum(p["sweeps"] for p in parts)/len(parts)}


@torch.no_grad()
def evaluate(model, arm, data, chunk: int = 128) -> dict:
    model.eval()
    geometry, (li, ti, _) = geometry_metrics(model.decoder.weight, data.dictionary)
    codes, marginal_codes, means, probs, risks, solves, energies = [], [], [], [], [], [], []
    for x in data.x.split(chunk):
        y, a, decoder = conditional_inputs(model, x)
        if arm == "exact32":
            out = exact_posterior(y, a, decoder, BETA, model.config.lambda_sparsity)
            m, code, risk, energy = out["m"], a*out["map"], out["risk"], out["free_energy"]
        elif arm == "optimized_mf":
            out = solve_mean_field(y, a, decoder, BETA, model.config.lambda_sparsity)
            solves.append(out)
            m = out["m"]
            code = a*(m > .5)
            risk = product_risk(y, a, decoder, m)
            energy = out["objective"]
        else:
            m, a, _ = model.encode(x)
            code = a*(m > .5)
            risk = product_risk(y, a, decoder, m)
            energy = product_free_energy(y, a, decoder, m, BETA, model.config.lambda_sparsity)
        codes.append(code)
        marginal_codes.append(a*(m > .5))
        means.append(a*m)
        probs.append(m)
        risks.append(risk)
        energies.append(energy)
    code, marginal, mean, m = map(torch.cat, (codes, marginal_codes, means, probs))
    native = code_metrics(code, model.decode(code), data, li, ti)
    result = {**geometry, **native, "expected_l0": float(m.sum(1).mean()),
              "alignment_learned_indices": li.tolist(), "alignment_truth_indices": ti.tolist(),
              "mean_mse": float((model.decode(mean)-data.x).square().mean()),
              "expected_sse": float(torch.cat(risks).mean()),
              "sampled_expected_mse": float(torch.cat(risks).mean()/data.x.shape[1]),
              "free_energy": float(torch.cat(energies).mean()),
              "fixed_beta": BETA, "beta_requires_grad": model.log_beta.requires_grad,
              "precision_floor_hits": 0,
              "readout": "joint_MAP" if arm == "exact32" else "marginal_strictly_gt_0.5"}
    if arm == "exact32":
        result["common_marginal_readout"] = code_metrics(marginal, model.decode(marginal), data, li, ti)
    if solves:
        result["solver"] = solver_summary(solves)
    return result


def code_metrics(code, reconstruction, data, li, ti):
    aligned = torch.zeros_like(data.z)
    aligned[:, ti] = code[:, li]
    support = aligned > 0
    tp = (support & data.support).sum(0)
    pred = support.sum(0)
    actual = data.support.sum(0)
    f1 = 2*tp/(pred+actual).clamp_min(1)
    recall = tp/actual.clamp_min(1)
    l0 = support.sum(1).float()
    true_l0 = data.support.sum(1).float()
    sse = (reconstruction-data.x).square().sum(1).mean()
    variance = (data.x-data.x.mean(0)).square().sum(1).mean().clamp_min(1e-12)
    return {"support_f1": float(2*tp.sum()/(pred.sum()+actual.sum()).clamp_min(1)),
            "support_macro_f1": float(f1.mean()), "per_feature_support_f1": f1.tolist(),
            "per_feature_recall": recall.tolist(), "hard_l0": float(l0.mean()),
            "true_l0": float(true_l0.mean()),
            "l0_absolute_sample_error": float((l0-true_l0).abs().mean()),
            "hard_latent_relative_error": float(((aligned-data.z).square().sum()/data.z.square().sum().clamp_min(1e-12)).sqrt()),
            "hard_mse": float(sse/data.x.shape[1]), "hard_sse": float(sse), "hard_ev": float(1-sse/variance),
            "active_latent_fraction": float(support.any(0).float().mean())}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def dataset_manifest(data):
    support = data.support.double().cpu()
    return {"x_sha256": tensor_hash(data.x), "z_sha256": tensor_hash(data.z),
            "dictionary_sha256": tensor_hash(data.dictionary), "count": len(data.x),
            "true_l0": float(support.sum(1).mean()),
            "empirical_support_correlation": torch.corrcoef(support.T).tolist()}


def train_fit(model, arm, x, world, steps=2000, checkpoint_steps=(), snapshot=None):
    from src.sae_train import _CyclingTensorBatches
    provider = _CyclingTensorBatches(x, 128, 40000+world)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=.003, betas=(.9, .999), weight_decay=0)
    started = time.perf_counter()
    trace, failed_batches, summaries = [], [], []
    first_hash = None
    for update in range(1, steps+1):
        model.train()
        batch = next(provider)
        if first_hash is None:
            first_hash = tensor_hash(batch)
        optimizer.zero_grad(set_to_none=True)
        loss, solved = training_loss(model, arm, batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite {arm} loss on update {update}")
        loss.backward()
        model.remove_decoder_parallel_grad()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        model.normalize_decoder_columns()
        if solved is not None:
            summaries.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in solved.items() if k != "history"})
            if not bool(solved["converged"].all()):
                failed_batches.append({"update": update, "failed_samples": int((~solved["converged"]).sum()),
                                       "max_delta": float(solved["delta"].max()),
                                       "max_residual": float(solved["residual"].max())})
        if update == 1 or update % 50 == 0:
            trace.append({"update": update, "batch_free_energy": float(loss.detach()),
                          "elapsed_seconds": time.perf_counter()-started})
        if update in checkpoint_steps and snapshot:
            snapshot(model, update)
        if model.log_beta.requires_grad or model.log_beta.grad is not None:
            raise AssertionError("The controlled precision must remain explicitly frozen")
    return {"updates": steps, "sample_presentations": steps*128, "first_batch_sha256": first_hash,
            "training_and_snapshot_seconds": time.perf_counter()-started, "trace": trace,
            "failed_batches": failed_batches, "solver": solver_summary(summaries),
            "gradient_semantics": "detached q; envelope for converged samples, fixed-q partial gradient otherwise",
            "excluded_samples": 0, "skipped_updates": 0}


def select_calibration(candidates):
    selected = []
    for rho in RHOS:
        for world in WORLDS:
            for arm in ARMS:
                group = [r for r in candidates if r["rho"] == rho and r["world"] == world and r["arm"] == arm and r["update"] == 2000]
                for target in (1.8, 2., 2.2):
                    row = min(group, key=lambda r: (abs(r["calibration"]["hard_l0"]-target), r["calibration"]["hard_mse"], r["gamma_index"]))
                    distance = abs(row["calibration"]["hard_l0"]-target)
                    selected.append({"rho": rho, "world": world, "arm": arm, "target": target,
                                     "candidate_id": row["candidate_id"], "calibration_l0": row["calibration"]["hard_l0"],
                                     "target_distance": distance, "within_target_tolerance": distance <= .15})
    return selected


def pair_differences(test_rows, selected, timings):
    by_id = {r["candidate_id"]: r for r in test_rows}
    fit_timing = {r["fit_id"]: r for r in timings}
    metrics = ("matched_positive_cosine", "recovered_fraction_cosine_095", "mixing_energy", "support_f1", "hard_l0", "hard_ev", "c_dec")
    pairs = []
    for rho in RHOS:
        for world in WORLDS:
            for comparator in ("optimized_mf", "amortized"):
                for target in (1.8, 2., 2.2):
                    exact = next(r for r in selected if r["rho"] == rho and r["world"] == world and r["arm"] == "exact32" and r["target"] == target)
                    comp = next(r for r in selected if r["rho"] == rho and r["world"] == world and r["arm"] == comparator and r["target"] == target)
                    e, c = by_id[exact["candidate_id"]], by_id[comp["candidate_id"]]
                    diff = {m: e["test"][m]-c["test"][m] for m in metrics}
                    paired = exact["within_target_tolerance"] and comp["within_target_tolerance"] and abs(exact["calibration_l0"]-comp["calibration_l0"]) <= .10
                    solver_ok = True
                    if comparator == "optimized_mf":
                        solver_ok = (fit_timing[c["fit_id"]]["solver"]["batch_convergence_fraction"] >= .95
                                     and c["test"]["solver"]["failed_samples"] == 0
                                     and c["calibration_solver"]["failed_samples"] == 0)
                    joint = (diff["matched_positive_cosine"] >= .05 or diff["recovered_fraction_cosine_095"] >= .20) and diff["support_f1"] >= .03 and diff["mixing_energy"] <= -.01
                    marginal_f1 = e["test"]["common_marginal_readout"]["support_f1"]-c["test"]["support_f1"]
                    pairs.append({"rho": rho, "world": world, "comparator": comparator, "target": target,
                                  "exact_candidate_id": exact["candidate_id"], "comparator_candidate_id": comp["candidate_id"],
                                  "matched_l0": paired, "solver_attribution_eligible": solver_ok,
                                  "exact_minus_comparator": diff, "common_marginal_f1_difference": marginal_f1,
                                  "common_marginal_test_l0_difference": e["test"]["common_marginal_readout"]["hard_l0"]-c["test"]["hard_l0"],
                                  "joint_threshold_pass": bool(paired and solver_ok and joint and marginal_f1 >= .03),
                                  "fidelity_tradeoff": diff["hard_ev"] < -.02})
    same_gamma = []
    for row in test_rows:
        if row["arm"] != "exact32":
            continue
        for comparator in ("optimized_mf", "amortized"):
            comp = next(r for r in test_rows if r["rho"] == row["rho"] and r["world"] == row["world"] and r["gamma"] == row["gamma"] and r["arm"] == comparator)
            same_gamma.append({"rho": row["rho"], "world": row["world"], "gamma": row["gamma"], "comparator": comparator,
                               "exact_minus_comparator": {m: row["test"][m]-comp["test"][m] for m in metrics}})
    return {"matched_selection_pairs": pairs, "same_gamma_total_effect_pairs": same_gamma,
            "interpretation": "Coverage or MF convergence failures leave covariance attribution unresolved; three worlds are screening only."}


def run_probe(output):
    output.mkdir(parents=True, exist_ok=True)
    dictionary = orthogonal_dictionary(210)
    data = sample_toy(dictionary, 32768, 10210, correlation=-.4)
    results = []
    for arm in ("optimized_mf", "exact32", "amortized"):
        model = make_model(arm, 0., 210)
        timing = train_fit(model, arm, data.x, 210, steps=100)
        results.append({"arm": arm, **timing})
    seconds = sum(r["training_and_snapshot_seconds"]*20*18 for r in results)
    report = {"device": "cpu", "torch_threads": torch.get_num_threads(), "probe_updates_per_arm": 100,
              "gamma": 0, "world": 210, "rho": -.4, "fits": results,
              "predicted_training_seconds": seconds, "predicted_with_50_percent_eval_overhead_seconds": 1.5*seconds,
              "gpu_hours": 0., "test_accessed": False}
    write_json(output/"runtime_probe.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "fits"}), flush=True)


def run_grid(output):
    if (output/"selection_manifest.json").exists():
        raise RuntimeError("Selection already frozen; use a new approved run path rather than rerunning tests")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    protocol = ROOT/"idea-stage/runs/vg-sae-sparse-but-wrong-20260922/pilots/joint_protocol.md"
    config = {"protocol_version": 1, "worlds": WORLDS, "rho": RHOS, "gamma": GAMMAS, "beta": BETA,
              "arms": ARMS, "steps": 2000, "batch_size": 128, "checkpoint_steps": [500, 1000, 2000],
              "device": "cpu", "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
              "optimizer": "Adam", "lr": .003, "gradient_clip_norm": 1., "beta_frozen": True,
              "finite_data": "shuffled cycling without replacement within epoch", "secondary_mixed_init": "deferred before test access",
              "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gpu_hours": 0., "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output/"config.json", config)
    candidates, timings, datasets = [], [], []
    for rho in RHOS:
        for world in WORLDS:
            dictionary = orthogonal_dictionary(world)
            train = sample_toy(dictionary, 32768, 10000+world, correlation=rho)
            calibration = sample_toy(dictionary, 8192, 20000+world, correlation=rho)
            datasets.append({"rho": rho, "world": world, "train": dataset_manifest(train), "calibration": dataset_manifest(calibration)})
            for gamma_index, gamma in enumerate(GAMMAS):
                for arm in ARMS:
                    fit_id = f"rho{rho:+.1f}_world{world}_{arm}_g{gamma_index}"
                    model = make_model(arm, gamma, world)
                    initial_hashes = {"decoder": tensor_hash(model.decoder.weight), "amplitude_weight": tensor_hash(model.amplitude_encoder.weight),
                                      "amplitude_bias": tensor_hash(model.amplitude_encoder.bias), "decoder_bias": tensor_hash(model.pre_bias)}
                    def snapshot(current, update):
                        candidate_id = f"{fit_id}_s{update}"
                        checkpoint = output/"checkpoints"/f"{candidate_id}.pt"
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(current.state_dict(), checkpoint)
                        candidates.append({"candidate_id": candidate_id, "fit_id": fit_id, "rho": rho, "world": world,
                                           "arm": arm, "gamma": gamma, "gamma_index": gamma_index, "update": update,
                                           "checkpoint": str(checkpoint.relative_to(ROOT)),
                                           "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                           "calibration": evaluate(current, arm, calibration)})
                        write_json(output/"calibration_results.json", candidates)
                    try:
                        timing = train_fit(model, arm, train.x, world, checkpoint_steps=(500, 1000, 2000), snapshot=snapshot)
                    except Exception as exc:
                        write_json(output/"failed_fits.json", [{"fit_id": fit_id, "error_type": type(exc).__name__,
                                   "error": str(exc), "completed_fits": len(timings), "test_accessed": False}])
                        raise
                    timings.append({"fit_id": fit_id, "initial_hashes": initial_hashes, **timing})
                    write_json(output/"training_timing_solver.json", timings)
                    print(json.dumps({"phase": "train", "completed_fits": len(timings), "total_fits": 54,
                                      "fit_id": fit_id, "seconds": timing["training_and_snapshot_seconds"],
                                      "solver": timing["solver"]}), flush=True)
    selections = select_calibration(candidates)
    final_candidates = [r for r in candidates if r["update"] == 2000]
    manifest = {"config": config, "datasets": datasets, "calibration_candidates": candidates,
                "selected": selections, "predeclared_test_ids": [r["candidate_id"] for r in final_candidates],
                "test_accessed_at_freeze": False, "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output/"selection_manifest.json", manifest)
    manifest_digest = hashlib.sha256((output/"selection_manifest.json").read_bytes()).hexdigest()
    test_rows, test_datasets = [], []
    for rho in RHOS:
        for world in WORLDS:
            test = sample_toy(orthogonal_dictionary(world), 16384, 30000+world, correlation=rho)
            test_datasets.append({"rho": rho, "world": world, **dataset_manifest(test)})
            for row in final_candidates:
                if row["rho"] != rho or row["world"] != world:
                    continue
                model = make_model(row["arm"], row["gamma"], world)
                model.load_state_dict(torch.load(ROOT/row["checkpoint"], map_location="cpu", weights_only=True))
                test_rows.append({k: row[k] for k in ("candidate_id", "fit_id", "rho", "world", "arm", "gamma", "gamma_index", "update")})
                test_rows[-1]["test"] = evaluate(model, row["arm"], test)
                for key in ("alignment_learned_indices", "alignment_truth_indices"):
                    if test_rows[-1]["test"][key] != row["calibration"][key]:
                        raise AssertionError("Dictionary alignment changed after calibration freeze")
                if row["arm"] == "optimized_mf":
                    test_rows[-1]["calibration_solver"] = row["calibration"]["solver"]
                write_json(output/"test_results.json", test_rows)
            print(json.dumps({"phase": "frozen_test", "rho": rho, "world": world, "completed_test_fits": len(test_rows)}), flush=True)
    write_json(output/"paired_comparisons.json", pair_differences(test_rows, selections, timings))
    write_json(output/"test_datasets.json", test_datasets)
    elapsed = time.perf_counter()-started
    write_json(output/"completion.json", {"complete": True, "primary_fits": len(timings), "test_evaluations": len(test_rows),
               "selection_manifest_sha256": manifest_digest, "elapsed_wall_seconds": elapsed,
               "cpu_wall_hours": elapsed/3600, "requested_thread_hours_upper_proxy": elapsed*torch.get_num_threads()/3600,
               "gpu_hours": 0., "finished_at_utc": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "run"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/sbw_20260922/joint")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the project")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    (run_probe if args.mode == "probe" else run_grid)(output)


if __name__ == "__main__":
    main()
