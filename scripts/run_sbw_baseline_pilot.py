"""Preregistered, bounded Sparse-but-Wrong-inspired current-VG pilot.

Run from the repository root. Calibration and test are separate phases, and a
frozen JSON selector receipt must exist before the test stream is constructed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.optimize import lsq_linear

from scripts.sbw_pilot_utils import (
    ToyBatch, evaluate_model, geometry_metrics, orthogonal_dictionary,
    sample_toy, tensor_hash, train_shared_batches,
)
from src.sae_baselines import (
    BatchTopKSAE, BatchTopKSAEConfig, GatedSAE, GatedSAEConfig,
    JumpReLUSAE, JumpReLUSAEConfig, StandardSAE, StandardSAEConfig,
    to_inference_sae,
)
from src.sae_model import VGSAEConfig, VariationalGarroteSAE

OUT = ROOT / "outputs/sbw_20260922/baseline"
PROTOCOL = ROOT / "idea-stage/runs/vg-sae-sparse-but-wrong-20260922/pilots/baseline_protocol.md"
JURY = ROOT / "idea-stage/runs/vg-sae-sparse-but-wrong-20260922/evidence/pilot_jury.json"
CONTROLS = {
    "vg": [-2., 0., 2., 4., 6.],
    "l1": [0., .03, .1, .3, 1.],
    "gated": [0., .03, .1, .3, 1.],
    "jumprelu_anthropic": [.03, .1, .3, 1., 3.],
    "batchtopk": [1., 1.5, 2., 2.5, 3.],
}
SEEDS = [210, 211, 212]
CORRELATIONS = [-.4, 0., .4]
TARGETS = [1.8, 2., 2.2]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenSelection:
    candidate_id: str | None
    selector: str
    target: float | None
    status: str


def select_calibration(candidates: list[dict], target: float | None,
                       selector: str = "oracle_recovery", tolerance: float = .15) -> FrozenSelection:
    """Consume calibration metrics only; no extrapolation through coverage gaps."""
    feasible = [row for row in candidates if target is None or
                abs(row["calibration"]["hard_l0"] - target) <= tolerance]
    if not feasible:
        return FrozenSelection(None, selector, target, "coverage_failure")
    def ranking(row):
        m = row["calibration"]
        tie = (row["completed_updates"], row["control_index"], row["candidate_id"])
        if selector == "matched_l0":
            return (abs(m["hard_l0"] - target), m["hard_mse"], row["control_index"], row["candidate_id"])
        if selector == "oracle_recovery":
            return (-m["matched_positive_cosine"], m["mixing_energy"], -m["support_f1"],
                    -row["completed_updates"], row["control_index"], row["candidate_id"])
        if selector == "deployment_c_dec":
            return (m["c_dec"], m["hard_mse"], *tie)
        if selector == "deployment_mse":
            return (m["hard_mse"], m["c_dec"], *tie)
        raise ValueError(f"Unknown selector: {selector}")
    best = min(feasible, key=ranking)
    return FrozenSelection(best["candidate_id"], selector, target, "selected")


def fit_amplitude_rescaling(code: torch.Tensor, decoder: torch.Tensor,
                            centered_inputs: torch.Tensor) -> torch.Tensor:
    """Fit positive per-code gains using only observed calibration inputs.

    The decoder, encoder, and binary support are fixed. No true support,
    coefficients, or dictionary is accepted by this API.
    """
    design = (code.double().unsqueeze(1) * decoder.double().unsqueeze(0)).reshape(-1, code.shape[1])
    target = centered_inputs.double().reshape(-1)
    result = lsq_linear(design.detach().cpu().numpy(), target.detach().cpu().numpy(),
                        bounds=(1e-8, np.inf), tol=1e-10, max_iter=500)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Calibration gain fit failed: {result.message}")
    return torch.as_tensor(result.x, dtype=code.dtype, device=code.device)


def make_model(method: str, control: float, world: int, device: str):
    torch.manual_seed(world)
    torch.cuda.manual_seed_all(world)
    if method == "vg":
        return VariationalGarroteSAE(VGSAEConfig(
            input_dim=20, n_latents=5, lambda_sparsity=control,
            beta_mode="profiled", use_variance_term=True, use_entropy_term=True,
        )).to(device)
    common = dict(d_in=20, d_sae=5, device=device, dtype="float32",
                  apply_b_dec_to_input=True, normalize_activations="none", decoder_init_norm=.1)
    if method == "l1":
        return StandardSAE(StandardSAEConfig(**common, l1_coefficient=control, l1_warm_up_steps=0))
    if method == "gated":
        return GatedSAE(GatedSAEConfig(**common, l1_coefficient=control, l1_warm_up_steps=0))
    if method == "jumprelu_anthropic":
        return JumpReLUSAE(JumpReLUSAEConfig(
            **common, l0_coefficient=control, jumprelu_sparsity_loss_mode="tanh",
            pre_act_loss_coefficient=3e-6, jumprelu_init_threshold=.1,
            jumprelu_bandwidth=2., jumprelu_tanh_scale=4., l0_warm_up_steps=1000))
    if method == "batchtopk":
        return BatchTopKSAE(BatchTopKSAEConfig(**common, k=control))
    raise ValueError(method)


def model_config(model) -> dict:
    if isinstance(model, VariationalGarroteSAE):
        result = asdict(model.config)
        result["dtype"] = str(result["dtype"])
        return result
    return model.cfg.to_dict()


@torch.no_grad()
def extended_metrics(model, data: ToyBatch, gains: torch.Tensor | None = None) -> dict:
    result = evaluate_model(model, data)
    if isinstance(model, VariationalGarroteSAE):
        m, amplitude, _ = model.encode(data.x)
        code = amplitude * (m > .5)
        decoder = model.decoder.weight
        reconstruction = model.decode(code)
        terms = model.free_energy(data.x)
        expected_risk = result["sampled_expected_mse"]
        result.update(calibration_or_evaluation_objective=float(terms["loss"]),
                      profiled_risk_floor_ratio=expected_risk / model.config.loss_eps,
                      profiled_risk_near_floor=expected_risk <= 10 * model.config.loss_eps)
    else:
        inference = to_inference_sae(model, fold_decoder_norm=True).to(data.x.device)
        code = inference.encode(data.x)
        decoder = inference.W_dec.T
        if gains is not None:
            code = code * gains
        reconstruction = inference.decode(code)
    training_decoder = model.decoder.weight if isinstance(model, VariationalGarroteSAE) else model.W_dec.T
    norms = training_decoder.norm(dim=0)
    result.update(training_decoder_norm_min=float(norms.min()),
                  training_decoder_norm_mean=float(norms.mean()), training_decoder_norm_max=float(norms.max()))
    _, (li, ti, _) = geometry_metrics(decoder, data.dictionary)
    aligned = torch.zeros_like(data.z)
    aligned[:, ti] = code[:, li]
    truth, predicted = data.support, aligned > 0
    tp = (truth & predicted).sum(0)
    actual = truth.sum(0)
    estimated = predicted.sum(0)
    matched_active = truth & predicted
    error = aligned - data.z
    nonzero = data.x.square().sum(1) > 0
    result.update({
        "support_macro_f1": float((2 * tp / (actual + estimated).clamp_min(1)).mean()),
        "per_feature_f1": (2 * tp / (actual + estimated).clamp_min(1)).tolist(),
        "per_feature_recall": (tp / actual.clamp_min(1)).tolist(),
        "coefficient_nmse": float(error.square().sum() / data.z.square().sum().clamp_min(1e-12)),
        "active_tp_coefficient_bias": float(error[matched_active].mean()) if matched_active.any() else None,
        "active_tp_coefficient_mae": float(error[matched_active].abs().mean()) if matched_active.any() else None,
        "zero_input_count": int((~nonzero).sum()),
        "output_input_norm_ratio_nonzero_only": float((reconstruction[nonzero].norm(dim=1) / data.x[nonzero].norm(dim=1)).mean()),
        "alignment_learned_indices": li.tolist(), "alignment_truth_indices": ti.tolist(),
    })
    if gains is not None:
        residual = (reconstruction - data.x).square().sum(1).mean()
        variance = (data.x - data.x.mean(0)).square().sum(1).mean().clamp_min(1e-12)
        result.update(hard_mse=float(residual / data.x.shape[1]), hard_ev=float(1 - residual / variance),
                      hard_latent_relative_error=float(result["coefficient_nmse"] ** .5),
                      rescaling_gains=gains.tolist())
    return result


def data_manifest(data: ToyBatch) -> dict:
    support = data.support.float()
    return {"count": data.x.shape[0], "x_sha256": tensor_hash(data.x),
            "z_sha256": tensor_hash(data.z), "dictionary_sha256": tensor_hash(data.dictionary),
            "marginal_firing": support.mean(0).tolist(), "true_l0": float(support.sum(1).mean()),
            "support_pearson": torch.corrcoef(support.T).tolist()}


def configure() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("P1 exclusively requires CUDA_VISIBLE_DEVICES=1 (physical GPU 1).")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def throughput(args) -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("Save protocol before any pilot operation.")
    started = time.perf_counter()
    dictionary = orthogonal_dictionary(999210)
    train = sample_toy(dictionary, 32768, 999211, device="cuda:0")
    rows = []
    for method, controls in CONTROLS.items():
        model = make_model(method, controls[2], 999210, "cuda:0")
        timing = train_shared_batches(model, train.x, steps=args.benchmark_steps,
            seed=999212, lr=.003, batch_size=256)
        row = {"method": method, **timing}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
    projected_seconds = sum(row["training_and_snapshot_seconds"] for row in rows) / args.benchmark_steps * args.steps * 45
    write_json(OUT / "throughput.json", {"phase": "engineering_throughput_only", "no_scientific_test": True,
        "rows": rows, "projected_training_seconds": projected_seconds,
        "projected_with_25_percent_overhead_seconds": projected_seconds * 1.25,
        "wall_seconds_including_setup": time.perf_counter() - started,
        "proposed_updates": args.steps, "total_models": 225, "protocol_sha256": file_hash(PROTOCOL)})


def run(args) -> None:
    started = time.perf_counter()
    if not JURY.exists():
        raise RuntimeError("Final pilot jury is required before scientific launch.")
    if not PROTOCOL.exists() or (OUT / "frozen_selection.json").exists():
        raise RuntimeError("Missing preregistration or output already scientifically frozen.")
    if args.steps != 3000:
        raise ValueError("Budget changes require editing preregistration before launch.")
    checkpoints = (500, 1500, 3000)
    output_checkpoints = OUT / "checkpoints"
    output_checkpoints.mkdir(parents=True, exist_ok=True)
    config = {"world_seeds": SEEDS, "correlations": CORRELATIONS, "controls": CONTROLS,
        "targets": TARGETS, "tolerance": .15, "checkpoints": checkpoints,
        "steps": args.steps, "lr": .003, "batch_size": 256,
        "counts": {"train": 32768, "calibration": 8192, "test": 16384},
        "seed_contract": {"dictionary_and_init": "world", "train": "10000+world", "calibration": "20000+world", "test": "30000+world", "batches": "40000+world"},
        "protocol_sha256": file_hash(PROTOCOL), "jury_sha256": file_hash(JURY),
        "source_sha256": {path: file_hash(ROOT / path) for path in ["scripts/run_sbw_baseline_pilot.py", "scripts/sbw_pilot_utils.py", "src/sae_model.py", "src/sae_baselines.py"]},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
            "sae_lens": importlib.metadata.version("sae-lens"), "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(), "physical_gpu": 1, "threads": 2,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
    write_json(OUT / "config.json", config)
    candidates, timing_rows, datasets = [], [], []
    for correlation in CORRELATIONS:
        for world in SEEDS:
            dictionary = orthogonal_dictionary(world)
            train = sample_toy(dictionary, 32768, 10000 + world, correlation=correlation, device="cuda:0")
            cal = sample_toy(dictionary, 8192, 20000 + world, correlation=correlation, device="cuda:0")
            world_key = f"corr{correlation:+.1f}_world{world}"
            datasets.append({"world_key": world_key, "train": data_manifest(train), "calibration": data_manifest(cal)})
            for method, controls in CONTROLS.items():
                for control_index, control in enumerate(controls):
                    model = make_model(method, control, world, "cuda:0")
                    name = f"{world_key}_{method}_c{control_index}"
                    cfg = model_config(model)
                    def snapshot(current, completed):
                        candidate_id = f"{name}_step{completed}"
                        state_file = output_checkpoints / f"{candidate_id}.pt"
                        torch.save({"model_state": {k: v.detach().cpu() for k, v in current.state_dict().items()},
                                    "model_config": cfg, "completed_updates": completed,
                                    "identity": {"world": world, "correlation": correlation, "method": method,
                                                 "control": control, "control_index": control_index}}, state_file)
                        metrics = extended_metrics(current, cal)
                        row = {"candidate_id": candidate_id, "world": world, "correlation": correlation,
                               "world_key": world_key, "method": method, "control": control,
                               "control_index": control_index, "completed_updates": completed,
                               "model_config": cfg, "state_file": str(state_file.relative_to(ROOT)),
                               "state_sha256": file_hash(state_file), "calibration": metrics}
                        if method == "l1":
                            inference = to_inference_sae(current, fold_decoder_norm=True).to("cuda:0")
                            with torch.no_grad():
                                gains = fit_amplitude_rescaling(inference.encode(cal.x), inference.W_dec.T, cal.x - inference.b_dec)
                            row["calibration_rescaling_gains"] = gains.tolist()
                            row["calibration_rescaled"] = extended_metrics(current, cal, gains)
                        candidates.append(row)
                    timing = train_shared_batches(model, train.x, steps=args.steps, seed=40000 + world,
                        lr=.003, batch_size=256, checkpoints=checkpoints, snapshot=snapshot)
                    timing_rows.append({"name": name, "method": method, **timing})
                    write_json(OUT / "calibration_candidates.json", candidates)
                    write_json(OUT / "training_timings.json", timing_rows)
                    elapsed = time.perf_counter() - started
                    print(json.dumps({"phase": "train_calibration", "completed_models": len(timing_rows),
                          "total_models": 225, "name": name, "elapsed_seconds": elapsed}), flush=True)
                    if elapsed > 6900:
                        raise TimeoutError("P1 2 GPUh cap approached; preserving partial data without test selection.")
                    del model
            del train, cal
    write_json(OUT / "dataset_manifests_train_cal.json", datasets)
    selections = []
    for correlation in CORRELATIONS:
        for world in SEEDS:
            for method in CONTROLS:
                group = [r for r in candidates if r["correlation"] == correlation and r["world"] == world and r["method"] == method]
                for target in TARGETS:
                    chosen = select_calibration(group, target)
                    selections.append({"correlation": correlation, "world": world, "method": method, **asdict(chosen)})
                    final_group = [r for r in group if r["completed_updates"] == args.steps]
                    chosen = select_calibration(final_group, target, "matched_l0")
                    selections.append({"correlation": correlation, "world": world, "method": method, **asdict(chosen)})
                for selector in ["deployment_c_dec", "deployment_mse"]:
                    chosen = select_calibration(final_group, None, selector)
                    selections.append({"correlation": correlation, "world": world, "method": method, **asdict(chosen)})
    receipt = {"phase": "calibration_frozen_before_test_generation", "selections": selections,
               "calibration_sha256": file_hash(OUT / "calibration_candidates.json"),
               "config_sha256": file_hash(OUT / "config.json"),
               "test_generation_started": False,
               "elapsed_seconds_at_freeze": time.perf_counter() - started}
    write_json(OUT / "frozen_selection.json", receipt)
    frozen_hash = file_hash(OUT / "frozen_selection.json")
    chosen_ids = {r["candidate_id"] for r in selections if r["candidate_id"] is not None}
    test_rows, test_manifests = [], []
    for correlation in CORRELATIONS:
        for world in SEEDS:
            test = sample_toy(orthogonal_dictionary(world), 16384, 30000 + world,
                              correlation=correlation, device="cuda:0")
            test_manifests.append({"world": world, "correlation": correlation, **data_manifest(test)})
            for row in candidates:
                if row["world"] != world or row["correlation"] != correlation:
                    continue
                if row["completed_updates"] != args.steps and row["candidate_id"] not in chosen_ids:
                    continue
                model = make_model(row["method"], row["control"], world, "cuda:0")
                state_path = ROOT / row["state_file"]
                if file_hash(state_path) != row["state_sha256"]:
                    raise RuntimeError("Checkpoint changed after calibration.")
                model.load_state_dict(torch.load(state_path, map_location="cuda:0", weights_only=True)["model_state"])
                model.eval()
                metrics = extended_metrics(model, test)
                test_row = {k: row[k] for k in ["candidate_id", "world", "correlation", "world_key", "method", "control", "control_index", "completed_updates"]}
                test_row.update(test=metrics, final_grid=row["completed_updates"] == args.steps,
                                chosen_by_calibration=row["candidate_id"] in chosen_ids)
                if row["method"] == "l1":
                    gains = torch.tensor(row["calibration_rescaling_gains"], device="cuda:0")
                    test_row["test_rescaled"] = extended_metrics(model, test, gains)
                test_rows.append(test_row)
                del model
            write_json(OUT / "test_metrics.json", test_rows)
            print(json.dumps({"phase": "frozen_test", "world": world, "correlation": correlation,
                              "evaluated_checkpoints": len(test_rows)}), flush=True)
    if frozen_hash != file_hash(OUT / "frozen_selection.json"):
        raise RuntimeError("Frozen selection receipt was modified during test.")
    write_json(OUT / "dataset_manifests_test.json", test_manifests)
    scalar_keys = sorted({key for row in test_rows for key, value in row["test"].items() if isinstance(value, (int, float))})
    with (OUT / "test_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "world", "correlation", "method", "control", "completed_updates", "final_grid", "chosen_by_calibration", *scalar_keys])
        writer.writeheader()
        for row in test_rows:
            writer.writerow({**{k: row[k] for k in writer.fieldnames[:8]}, **{k: row["test"].get(k) for k in scalar_keys}})
    write_json(OUT / "completion.json", {"status": "complete", "trained_models": len(timing_rows),
        "calibration_checkpoints": len(candidates), "unique_test_checkpoints": len(test_rows),
        "full_final_grid_count": sum(r["final_grid"] for r in test_rows),
        "coverage_failures": sum(r["status"] == "coverage_failure" for r in selections),
        "frozen_selection_sha256": frozen_hash,
        "training_and_snapshot_seconds": sum(r["training_and_snapshot_seconds"] for r in timing_rows),
        "wall_seconds_including_training_calibration_io_test": time.perf_counter() - started})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["throughput", "run"], required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--benchmark-steps", type=int, default=150)
    args = parser.parse_args()
    configure()
    (throughput if args.mode == "throughput" else run)(args)


if __name__ == "__main__":
    main()
