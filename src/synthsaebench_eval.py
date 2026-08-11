"""Streaming official-style metrics for fixed-generator SynthSAEBench sweeps."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sae_lens.evals import ExplainedVarianceCalculator
from sae_lens.synthetic import SyntheticModel
from sae_lens.util import cosine_similarities
from scipy.optimize import linear_sum_assignment

from .sae_baselines import to_inference_sae
from .saelens_vg import VGSAE
from .synthsaebench_sweep import (
    METHOD_LABELS,
    SAELENS_REVISION,
    SynthSAEBenchRunSpec,
    SynthSAEBenchSweepConfig,
    temporary_seed_for_device,
)


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    safe_denominator = denominator.masked_fill(denominator <= 0, 1.0)
    return torch.where(
        denominator > 0,
        numerator / safe_denominator,
        torch.zeros_like(numerator),
    )


@torch.no_grad()
def evaluate_model(
    training_model: torch.nn.Module,
    synthetic: SyntheticModel,
    config: SynthSAEBenchSweepConfig,
    spec: SynthSAEBenchRunSpec,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate without materializing the 25M-sample held-out stream.

    MCC and classifier matching follow SAELens' official definitions.  Extra
    reconstruction and AP/AUC diagnostics are accumulated from the same stream.
    Only ``heatmap_samples`` rows are retained for notebook-10-style masks.
    """

    model = to_inference_sae(training_model, fold_decoder_norm=True).to(
        synthetic.feature_dict.feature_vectors.device
    )
    model.eval()
    decoder = model.W_dec.float()
    ground_truth_dictionary = synthetic.feature_dict.feature_vectors.float()
    cosine = cosine_similarities(decoder, ground_truth_dictionary).abs()
    best_matches = cosine.argmax(dim=1)
    learned_idx, true_idx = linear_sum_assignment(
        cosine.detach().cpu().numpy(), maximize=True
    )
    matched_cosines = cosine[learned_idx, true_idx]
    mcc = float(matched_cosines.mean().cpu())
    uniqueness = float(best_matches.unique().numel() / decoder.shape[0])

    device = decoder.device
    width = decoder.shape[0]
    n_true = ground_truth_dictionary.shape[0]
    tp = torch.zeros(width, dtype=torch.float64, device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    tn = torch.zeros_like(tp)
    ever_fired = torch.zeros(width, dtype=torch.bool, device=device)
    explained_variance = ExplainedVarianceCalculator()
    expected_explained_variance = ExplainedVarianceCalculator()
    true_l0_sum = sae_l0_sum = 0.0
    input_sq_sum = reconstruction_sq_error = 0.0
    expected_sq_error = 0.0
    reconstruction_element_count = 0
    shrinkage_sum = 0.0
    shrinkage_count = 0
    latent_sq_error = latent_target_sq = 0.0
    posterior_sum = posterior_variance_sum = 0.0
    preview_masks: list[torch.Tensor] = []
    preview_support: list[torch.Tensor] = []
    preview_latents: list[torch.Tensor] = []
    preview_true_latents: list[torch.Tensor] = []
    preview_probabilities: list[torch.Tensor] = []
    preview_remaining = config.training.heatmap_samples
    is_vg = isinstance(model, VGSAE)

    processed = 0
    with temporary_seed_for_device(spec.eval_stream_seed, device):
        while processed < config.data.n_test:
            current = min(config.training.batch_size, config.data.n_test - processed)
            feature_acts = synthetic.activation_generator.sample(current)
            hidden = synthetic.feature_dict(feature_acts)
            if is_vg:
                posterior = model.posterior(hidden)
                probabilities = posterior["m"]
                hard = posterior["a"] * (
                    probabilities > model.cfg.inference_threshold
                ).to(posterior["a"])
                latents = model.hook_sae_acts_post(hard)
            else:
                posterior = None
                probabilities = None
                latents = model.encode(hidden)
            reconstruction = model.decode(latents)
            true_matched = feature_acts[:, best_matches]
            fires = latents > 0
            truth = true_matched > 0

            true_l0_sum += float((feature_acts > 0).sum().cpu())
            sae_l0_sum += float(fires.sum().cpu())
            ever_fired |= fires.any(dim=0)
            tp += (fires & truth).sum(dim=0)
            fp += (fires & ~truth).sum(dim=0)
            fn += (~fires & truth).sum(dim=0)
            tn += (~fires & ~truth).sum(dim=0)
            explained_variance.add_batch(reconstruction, hidden)

            difference = (reconstruction.float() - hidden.float()).pow(2)
            reconstruction_sq_error += float(difference.sum().cpu())
            input_sq_sum += float(hidden.float().pow(2).sum().cpu())
            reconstruction_element_count += hidden.numel()
            input_norm = hidden.float().norm(dim=-1)
            valid_norm = input_norm > 1.0e-6
            if valid_norm.any():
                shrinkage_sum += float(
                    (
                        reconstruction.float().norm(dim=-1)[valid_norm]
                        / input_norm[valid_norm]
                    )
                    .sum()
                    .cpu()
                )
                shrinkage_count += int(valid_norm.sum().cpu())
            latent_sq_error += float(
                (latents.float() - true_matched.float()).pow(2).sum().cpu()
            )
            latent_target_sq += float(true_matched.float().pow(2).sum().cpu())

            if posterior is not None and probabilities is not None:
                expected = model.decode(posterior["expected_code"])
                posterior_sum += float(probabilities.sum().cpu())
                posterior_variance_sum += float(
                    (probabilities * (1.0 - probabilities)).sum().cpu()
                )
                expected_explained_variance.add_batch(expected, hidden)
                expected_sq_error += float(
                    (expected.float() - hidden.float()).pow(2).sum().cpu()
                )

            if preview_remaining:
                take = min(preview_remaining, current)
                preview_masks.append(fires[:take].detach().cpu())
                preview_support.append(truth[:take].detach().cpu())
                preview_latents.append(latents[:take].detach().float().cpu())
                preview_true_latents.append(
                    true_matched[:take].detach().float().cpu()
                )
                if probabilities is not None:
                    preview_probabilities.append(
                        probabilities[:take].detach().float().cpu()
                    )
                preview_remaining -= take
            processed += current

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, tp + fp + fn + tn)
    prevalence = _safe_divide(tp + fn, tp + fp + fn + tn)
    average_precision = precision * recall + prevalence * (1.0 - recall)
    specificity = _safe_divide(tn, tn + fp)
    roc_auc = 0.5 * (recall + specificity)
    selection_error = _safe_divide(fp + fn, tp + fp + fn + tn)

    sae_l0 = sae_l0_sum / processed
    true_l0 = true_l0_sum / processed
    expected_l0 = posterior_sum / processed if is_vg else sae_l0
    posterior_rho = posterior_sum / (processed * width) if is_vg else np.nan
    posterior_variance = (
        posterior_variance_sum / (processed * width) if is_vg else np.nan
    )
    expected_ev = expected_explained_variance.compute() if is_vg else np.nan
    expected_relative_error = (
        math_sqrt(expected_sq_error / max(input_sq_sum, 1.0e-12))
        if is_vg
        else np.nan
    )
    reconstruction_error = math_sqrt(
        reconstruction_sq_error / max(input_sq_sum, 1.0e-12)
    )
    generalization_error = math_sqrt(
        latent_sq_error / max(latent_target_sq, 1.0e-12)
    )
    metadata = config.data
    row: dict[str, Any] = {
        "run_id": spec.run_id,
        "seed": spec.seed,
        "init_seed": spec.init_seed,
        "calibration_seed": spec.calibration_seed,
        "train_stream_seed": spec.train_stream_seed,
        "eval_stream_seed": spec.eval_stream_seed,
        "method": spec.method,
        "method_label": METHOD_LABELS[spec.method],
        "control_name": spec.control_name,
        "control_value": spec.control_value,
        "benchmark_model_id": metadata.model_id,
        "benchmark_revision": metadata.revision,
        "benchmark_model_config_sha256": metadata.model_config_sha256,
        "benchmark_scale_children_by_parent": metadata.scale_children_by_parent,
        "saelens_revision": SAELENS_REVISION,
        "input_dim": metadata.input_dim,
        "ground_truth_num_features": n_true,
        "sae_width": width,
        "n_training_samples": metadata.n_train,
        "n_evaluation_samples": processed,
        "sae_l0": sae_l0,
        "true_l0": true_l0,
        "rho_model": sae_l0 / width,
        "true_l0_over_d_sae": true_l0 / width,
        "true_feature_density": true_l0 / n_true,
        "target_model_density": true_l0 / width,
        "average_l0": sae_l0,
        "expected_l0": expected_l0,
        "dead_latents": int((~ever_fired).sum().cpu()),
        "dead_fraction": float((~ever_fired).float().mean().cpu()),
        "shrinkage": shrinkage_sum / max(shrinkage_count, 1),
        "explained_variance": explained_variance.compute(),
        "reconstruction_error": reconstruction_error,
        "reconstruction_mse": reconstruction_sq_error
        / max(reconstruction_element_count, 1),
        "generalization_error": generalization_error,
        "mcc": mcc,
        "uniqueness": uniqueness,
        # These aliases retain notebook-10's tabular seam; Synth plots label and
        # interpret them using the official definitions rather than Stage-1 union metrics.
        "decoder_recovery_cosine": mcc,
        "support_precision": float(precision.mean().cpu()),
        "support_recall": float(recall.mean().cpu()),
        "support_f1": float(f1.mean().cpu()),
        "support_average_precision": float(average_precision.mean().cpu()),
        "support_roc_auc": float(roc_auc.mean().cpu()),
        "classification_precision": float(precision.mean().cpu()),
        "classification_recall": float(recall.mean().cpu()),
        "classification_f1": float(f1.mean().cpu()),
        "classification_accuracy": float(accuracy.mean().cpu()),
        "selection_error": float(selection_error.mean().cpu()),
        "matching_policy": "official_mcc_and_per_latent_best_cosine_classifier",
        "vg_posterior_rho": posterior_rho,
        "vg_expected_l0": expected_l0 if is_vg else np.nan,
        "vg_posterior_variance": posterior_variance,
        "vg_expected_explained_variance": expected_ev,
        "vg_expected_relative_error": expected_relative_error,
    }
    if is_vg:
        probability_preview = torch.cat(preview_probabilities)
        quantile_levels = {
            "q10": 0.10,
            "q25": 0.25,
            "q50": 0.50,
            "q75": 0.75,
            "q90": 0.90,
        }
        row.update(
            {
                f"vg_posterior_probability_{name}": float(
                    torch.quantile(probability_preview, level)
                )
                for name, level in quantile_levels.items()
            }
        )
        row["vg_expected_to_hard_l0_ratio"] = expected_l0 / max(sae_l0, 1.0e-12)
        row["vg_expected_hard_ev_gap"] = expected_ev - row["explained_variance"]
    cache = {
        "mask": torch.cat(preview_masks).numpy(),
        "true_support": torch.cat(preview_support).numpy(),
        "h": torch.cat(preview_latents).numpy(),
        "true_latents": torch.cat(preview_true_latents).numpy(),
        "matched_true_idx": best_matches.detach().cpu().numpy(),
        "preview_sample_count": np.asarray(
            config.training.heatmap_samples, dtype=np.int64
        ),
    }
    if is_vg:
        cache["posterior_probability"] = probability_preview.numpy()
    return row, cache


def math_sqrt(value: float) -> float:
    """Keep scalar square roots independent of NumPy scalar serialization."""

    return float(value**0.5)


__all__ = ["evaluate_model"]
