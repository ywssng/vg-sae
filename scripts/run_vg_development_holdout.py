"""Fresh calibration/test evaluation of frozen Stage-1 checkpoint grids.

No training or test-set model selection is performed. Run from the project root.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sae_baselines import to_inference_sae
from src.sae_data import feature_probabilities, make_unit_dictionary
from src.sae_model import VariationalGarroteSAE
from src.sae_sweep import SweepConfig, build_specs, load_checkpoint, make_train_test
from src.sae_sweep_eval import _l1_threshold

DEFAULT_OUTPUT = ROOT / "outputs/vg_sae_development_20260921/holdout"
SOURCES = {
    "exponential": "outputs/runs/stage1_beta_profiled_din128_gt1024_sae1024_sd001_seed0",
    "constant": "outputs/runs/stage1_ablation2_constant_beta_profiled_din128_gt1024_sae1024_sd001_seed0",
}
CAPS = (4.0, 8.0, 16.0)
CAL_SEED, TEST_SEED = 2026092101, 2026092102


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sample_fixed_dictionary(
    dictionary: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_samples: int,
    amplitude_mode: str,
    amplitude_scale: float,
    noise_std: float,
    base_seed: int,
    source_offset: int,
) -> dict[str, np.ndarray]:
    """Draw new examples without drawing or modifying a dictionary.

    SeedSequence components separate support, amplitudes, and observation noise.
    """
    shape = (n_samples, dictionary.shape[1])
    rngs = {
        name: np.random.default_rng(np.random.SeedSequence([base_seed, source_offset, offset]))
        for name, offset in (("support", 11), ("amplitude", 17), ("noise", 23))
    }
    support = rngs["support"].binomial(1, probabilities[None, :], size=shape).astype(np.float64)
    if amplitude_mode == "exponential":
        amplitude = rngs["amplitude"].exponential(amplitude_scale, size=shape)
    elif amplitude_mode == "constant":
        amplitude = np.full(shape, math.sqrt(2) * amplitude_scale)
    else:
        raise ValueError("This preregistered pilot supports exponential/constant only.")
    z = support * amplitude
    clean_x = z @ dictionary.T
    x = clean_x + rngs["noise"].normal(0, noise_std, size=clean_x.shape)
    return {"x": x.astype(np.float32), "z": z.astype(np.float32), "support": support.astype(bool)}


def resolve_l1_threshold(
    saved_metric: dict, original_train_activations: Callable[[], torch.Tensor]
) -> tuple[float, str]:
    """Only persisted train-fit thresholds or original training activations enter."""
    saved = saved_metric.get("l1_gmm_threshold")
    if saved not in (None, ""):
        threshold = float(saved)
        if not math.isnan(threshold):
            return threshold, "saved_train_fit_gmm"
    return _l1_threshold(original_train_activations()), "refit_original_train_only"


def select_calibration_frontier(rows: list[dict], caps=CAPS) -> list[dict]:
    """Choose a control using only explicitly named calibration fields."""
    selections = []
    groups = sorted({(row["condition"], row["method"]) for row in rows})
    for condition, method in groups:
        group = [r for r in rows if (r["condition"], r["method"]) == (condition, method)]
        for cap in caps:
            supported = [r for r in group if r["cal_average_l0"] <= cap]
            selected = min(
                supported,
                key=lambda r: (
                    r["cal_hard_generalization_error"],
                    r["cal_hard_reconstruction_mse"],
                    r["cal_average_l0"],
                    r["control_order"],
                ),
                default=None,
            )
            selections.append({
                "condition": condition, "method": method, "l0_cap": cap,
                "supported": selected is not None,
                "eligible_control_count": len(supported),
                "evaluated_control_count": len(group),
                "selected_run_id": selected["run_id"] if selected else None,
                "selected_checkpoint": selected["checkpoint"] if selected else None,
                "control_value": selected["control_value"] if selected else None,
                "control_order": selected["control_order"] if selected else None,
                "cal_average_l0": selected["cal_average_l0"] if selected else None,
                "cal_hard_generalization_error": selected["cal_hard_generalization_error"] if selected else None,
                "cal_hard_reconstruction_mse": selected["cal_hard_reconstruction_mse"] if selected else None,
                "cal_hard_support_f1": selected["cal_hard_support_f1"] if selected else None,
            })
    return selections


@torch.no_grad()
def prepare_model(checkpoint: Path, dictionary: torch.Tensor, device: str):
    model, payload = load_checkpoint(checkpoint, device="cpu")
    is_vg = isinstance(model, VariationalGarroteSAE)
    inference = model if is_vg else to_inference_sae(model, fold_decoder_norm=True)
    inference = inference.to(device).eval()
    decoder = inference.decoder.weight if is_vg else inference.W_dec.T
    learned = torch.nn.functional.normalize(decoder.float(), dim=0)
    true = torch.nn.functional.normalize(dictionary.float(), dim=0)
    cosine = (learned.T @ true).cpu().numpy()
    learned_idx, true_idx = linear_sum_assignment(-np.abs(cosine))
    signs = np.sign(cosine[learned_idx, true_idx])
    signs[signs == 0] = 1
    matching = {
        "learned_idx": learned_idx, "true_idx": true_idx, "signs": signs,
        "matched_cosine": float(np.abs(cosine[learned_idx, true_idx]).mean()),
    }
    return inference, payload, matching


@torch.no_grad()
def evaluate_hard(
    model, data: dict[str, torch.Tensor], matching: dict, *,
    method: str, l1_threshold: float | None, threshold: float = .5,
    batch_size: int = 256,
) -> dict[str, float]:
    device = data["x"].device
    learned_idx = torch.as_tensor(matching["learned_idx"], device=device)
    true_idx = torch.as_tensor(matching["true_idx"], device=device)
    signs = torch.as_tensor(matching["signs"], device=device)
    width, n_true = len(learned_idx), data["z"].shape[1]
    if width != n_true:
        raise ValueError("This pilot preregisters equal-width complete Hungarian matching.")
    totals = dict(sse_z=0., target_z=0., sse_x=0., l0=0., tp=0., fp=0., fn=0., expected_l0=0., mean_sse_x=0.)
    n = len(data["x"])
    for start in range(0, n, batch_size):
        x, z, truth = (data[key][start:start + batch_size] for key in ("x", "z", "support"))
        if isinstance(model, VariationalGarroteSAE):
            m, a, native = model.encode(x)
            support = m >= threshold  # Same boundary as Stage-1 evaluator.
            h = a * support
            expected_l0 = m.sum()
        else:
            native = model.encode(x)
            support = native > (l1_threshold if method == "l1" else 0.)
            h = native * support
            expected_l0 = support.sum()
        reconstructed = model.decode(h)
        mean_reconstructed = model.decode(native)
        aligned_h = h[:, learned_idx].double() * signs
        aligned_z = z[:, true_idx].double()
        aligned_support = support[:, learned_idx]
        aligned_truth = truth[:, true_idx]
        totals["sse_z"] += float((aligned_h - aligned_z).square().sum())
        totals["target_z"] += float(aligned_z.square().sum())
        totals["sse_x"] += float((reconstructed - x).double().square().sum())
        totals["mean_sse_x"] += float((mean_reconstructed - x).double().square().sum())
        totals["l0"] += float(support.sum())
        totals["expected_l0"] += float(expected_l0)
        totals["tp"] += float((aligned_support & aligned_truth).sum())
        totals["fp"] += float((aligned_support & ~aligned_truth).sum())
        totals["fn"] += float((~aligned_support & aligned_truth).sum())
    x = data["x"].double()
    centered_energy = float((x - x.mean(0)).square().sum())
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1.)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1.)
    return {
        "n_samples": n,
        "hard_generalization_error": math.sqrt(totals["sse_z"] / max(totals["target_z"], 1e-12)),
        "hard_reconstruction_mse": totals["sse_x"] / x.numel(),
        "hard_explained_variance": 1 - totals["sse_x"] / max(centered_energy, 1e-12),
        "hard_support_precision": precision,
        "hard_support_recall": recall,
        "hard_support_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "average_l0": totals["l0"] / n,
        "expected_l0": totals["expected_l0"] / n,
        "mean_code_reconstruction_mse": totals["mean_sse_x"] / x.numel(),
        "decoder_recovery_cosine": matching["matched_cosine"],
        "ground_truth_empirical_l0": float(data["support"].double().sum() / n),
        "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
    }


def recover_source(root: Path, condition: str, out: Path) -> dict:
    """Recover the original seed-0 dictionary, and verify persisted old labels."""
    cfg = SweepConfig.from_dict(json.loads((root / "sweep_config.json").read_text()))
    train, old_test = make_train_test(cfg, 0, "cpu")
    saved = np.load(root / "runs/vgsae/vgsae_gamma=3.0_seed=0/eval/last/cache.npz")
    if not np.array_equal(saved["ground_truth_support"], old_test.support.numpy()):
        raise RuntimeError("Regenerated original support disagrees with saved artifact.")
    if not np.array_equal(saved["ground_truth_latents"], old_test.z.numpy()):
        raise RuntimeError("Regenerated original latent coefficients disagree with saved artifact.")
    dictionary = make_unit_dictionary(cfg.data.input_dim, cfg.data.ground_truth_num_features,
                                     np.random.default_rng(0), cfg.data.coherence)
    probabilities = feature_probabilities(cfg.data.ground_truth_num_features,
                                          cfg.data.support_density, cfg.data.frequency_skew)
    np.testing.assert_array_equal(dictionary.astype(np.float32), train.dictionary.numpy())
    np.testing.assert_array_equal(probabilities.astype(np.float32), train.feature_probabilities.numpy())
    frozen_path = out / f"{condition}_frozen_source.npz"
    np.savez_compressed(frozen_path, dictionary=dictionary, feature_probabilities=probabilities)
    metric_path = root / "summary/last/final_metrics.csv"
    with metric_path.open() as f:
        saved_metrics = {r["run_id"]: r for r in csv.DictReader(f)}
    return {"config": cfg, "train": train, "dictionary": dictionary, "probabilities": probabilities,
            "metrics": saved_metrics, "frozen_path": frozen_path,
            "source_verification": "Original saved config/seed0 reproduced cached support and latents bitwise; dictionary/probabilities frozen before fresh sampling."}


def tensors(data: dict[str, np.ndarray], device: str):
    return {k: torch.from_numpy(v).to(device) for k, v in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=7200.)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output must remain inside this repository.")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    start = time.monotonic()
    def check_deadline():
        if time.monotonic() - start > args.max_seconds:
            raise TimeoutError("Preregistered runtime limit reached.")
    protocol = {
        "created_at": utc_now(), "phase": "benchmark" if args.benchmark_only else "full",
        "sources": SOURCES, "checkpoint_kind": "last", "training_seed": 0,
        "conditions": ["exponential", "constant"], "source_offsets": {"exponential": 0, "constant": 1},
        "calibration_seed": CAL_SEED, "test_seed": TEST_SEED,
        "rng_components": {"support": 11, "amplitude": 17, "noise": 23},
        "rng_rule": "NumPy SeedSequence([base_seed, source_offset, component_offset])",
        "calibration_samples": 2048, "test_samples": 4096, "l0_caps": CAPS,
        "selection": "Per condition/method/cap: minimum calibration hard latent relative error subject to calibration mean L0 <= cap; tie calibration hard MSE, calibration L0, fixed control order.",
        "test_access": "Generate and evaluate test only after every calibration decision is serialized and hashed.",
        "comparison": "Same L0-cap frontier, not exact matched L0; test cap excess is reported without reselection.",
        "candidate_scope": "Complete original 273-control grid per condition; no old-metric shortlist.",
        "limitations": ["One frozen training/dictionary seed; fresh examples do not provide training-seed robustness.", "Unequal method control grids and optimizer/parameter budgets; descriptive pilot only.", "Source conditions were chosen after observing historical curves; not an untouched problem-selection test."],
        "script_sha256": sha256_file(Path(__file__)),
        "device": args.device, "max_seconds": args.max_seconds,
    }
    protocol_path = out / ("benchmark_protocol.json" if args.benchmark_only else "protocol.json")
    if protocol_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing preregistration: {protocol_path}")
    save_json(protocol_path, protocol)
    environment = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                   "gpu": torch.cuda.get_device_name(args.device), "cuda": torch.version.cuda,
                   "device": args.device, "torch_threads": torch.get_num_threads()}
    save_json(out / ("benchmark_environment.json" if args.benchmark_only else "environment.json"), environment)
    source_data = {}
    all_cal = []
    matching_cache = {}
    threshold_cache = {}
    for source_offset, (condition, relative_root) in enumerate(SOURCES.items()):
        check_deadline()
        source = recover_source(ROOT / relative_root, condition, out)
        source_data[condition] = source
        config = source["config"]
        source["source_offset"] = source_offset
        cal_array = sample_fixed_dictionary(source["dictionary"], source["probabilities"],
            n_samples=2048, amplitude_mode=config.data.amplitude_mode, amplitude_scale=config.data.amplitude_scale,
            noise_std=config.data.noise_std, base_seed=CAL_SEED, source_offset=source_offset)
        np.savez_compressed(out / f"{condition}_calibration.npz", **cal_array)
        cal = tensors(cal_array, args.device)
        dictionary = torch.as_tensor(source["dictionary"], dtype=torch.float32, device=args.device)
        specs = build_specs(config)
        if len(specs) != 273:
            raise RuntimeError("Expected the complete preregistered 273-control grid.")
        for order, spec in enumerate(specs):
            check_deadline()
            checkpoint = ROOT / relative_root / "runs" / spec.method / spec.run_id / "checkpoints/last.pt"
            tick = time.monotonic()
            model, payload, matching = prepare_model(checkpoint, dictionary, args.device)
            threshold, threshold_source = None, "not_applicable"
            if spec.method == "l1":
                threshold, threshold_source = resolve_l1_threshold(
                    source["metrics"].get(spec.run_id, {}),
                    lambda: model.encode(source["train"].x.to(args.device)),
                )
            threshold_cache[(condition, spec.run_id)] = threshold
            matching_cache[(condition, spec.run_id)] = matching
            metrics = evaluate_hard(model, cal, matching, method=spec.method,
                                    l1_threshold=threshold, threshold=config.training.mask_threshold)
            elapsed = time.monotonic() - tick
            row = {"condition": condition, "method": spec.method, "run_id": spec.run_id,
                   "control_name": spec.control_name, "control_value": spec.control_value,
                   "control_order": order, "checkpoint": str(checkpoint.relative_to(ROOT)),
                   "checkpoint_sha256": sha256_file(checkpoint),
                   "checkpoint_step": payload["step"], "training_seed": spec.seed,
                   "l1_threshold": threshold if threshold is None or math.isfinite(threshold) else "inf",
                   "l1_threshold_source": threshold_source, "evaluation_seconds": elapsed,
                   **{f"cal_{k}": v for k, v in metrics.items()}}
            all_cal.append(row)
            del model
            if args.benchmark_only:
                save_json(out / "benchmark.json", {"row": row, "checkpoint_seconds": elapsed,
                    "projected_546_checkpoint_seconds": elapsed * 546,
                    "total_wall_seconds": time.monotonic() - start})
                print(json.dumps({"benchmark_checkpoint_seconds": elapsed,
                    "projected_full_gpu_hours": elapsed * 582 / 3600}), flush=True)
                return
            if len(all_cal) % 25 == 0:
                save_csv(out / "calibration_metrics.csv", all_cal)
                print(f"calibration {len(all_cal)}/546 elapsed={time.monotonic()-start:.1f}s", flush=True)
        del cal
    save_csv(out / "calibration_metrics.csv", all_cal)
    selected = select_calibration_frontier(all_cal)
    selection_path = out / "frozen_selection.json"
    save_json(selection_path, {"frozen_at": utc_now(), "test_generated": False,
        "calibration_metrics_sha256": sha256_file(out / "calibration_metrics.csv"),
        "selection_rule": protocol["selection"], "selections": selected})
    frozen_sha256 = sha256_file(selection_path)
    print(f"selection frozen sha256={frozen_sha256}; test generation starts now", flush=True)
    source_manifest = {}
    for condition, source in source_data.items():
        source_manifest[condition] = {"frozen_source": str(source["frozen_path"].relative_to(ROOT)),
            "sha256": sha256_file(source["frozen_path"]), "verification": source["source_verification"],
            "resolved_config": source["config"].to_dict()}
    save_json(out / "source_manifest.json", source_manifest)
    test_by_key = {}
    for condition, source in source_data.items():
        check_deadline()
        config = source["config"]
        array = sample_fixed_dictionary(source["dictionary"], source["probabilities"],
            n_samples=4096, amplitude_mode=config.data.amplitude_mode, amplitude_scale=config.data.amplitude_scale,
            noise_std=config.data.noise_std, base_seed=TEST_SEED, source_offset=source["source_offset"])
        np.savez_compressed(out / f"{condition}_test.npz", **array)
        test = tensors(array, args.device)
        for choice in selected:
            key = (condition, choice["selected_run_id"])
            if choice["condition"] != condition or not choice["supported"] or key in test_by_key:
                continue
            check_deadline()
            checkpoint = ROOT / choice["selected_checkpoint"]
            model, _ = load_checkpoint(checkpoint, device="cpu")
            if not isinstance(model, VariationalGarroteSAE):
                model = to_inference_sae(model, fold_decoder_norm=True)
            model = model.to(args.device).eval()
            matching = matching_cache[key]
            map_path = out / f"{condition}_{choice['selected_run_id']}_matching.npz"
            np.savez_compressed(map_path, **matching)
            test_by_key[key] = {f"test_{k}": v for k, v in evaluate_hard(model, test, matching,
                method=choice["method"], l1_threshold=threshold_cache[key],
                threshold=config.training.mask_threshold).items()}
            test_by_key[key]["matching_path"] = str(map_path.relative_to(ROOT))
            del model
        del test
    final = []
    for choice in selected:
        row = {**choice, "frozen_selection_sha256": frozen_sha256}
        if choice["supported"]:
            row.update(test_by_key[(choice["condition"], choice["selected_run_id"])])
            row["test_l0_cap_excess"] = max(0., row["test_average_l0"] - row["l0_cap"])
            row["test_l0_exceeds_cap"] = row["test_average_l0"] > row["l0_cap"]
        final.append(row)
    if sha256_file(selection_path) != frozen_sha256:
        raise RuntimeError("Frozen selection changed during test evaluation.")
    save_csv(out / "selected_test_metrics.csv", final)
    save_json(out / "selected_test_metrics.json", final)
    save_json(out / "completion.json", {"completed_at": utc_now(), "wall_seconds": time.monotonic() - start,
        "calibration_checkpoints": len(all_cal), "selected_rows": len(selected),
        "unique_test_checkpoints": len(test_by_key), "frozen_selection_sha256": frozen_sha256,
        "test_selection_performed": False})
    print(f"complete: {len(all_cal)} calibration checkpoints, {len(test_by_key)} unique test checkpoints; {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
