"""Experiment-07 evaluation, shared by the evaluator and plotting pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sae_lens import TrainingSAE

from .evaluate import selection_error, selection_uncertainty
from .sae_baselines import to_inference_sae
from .sae_model import StandardSAE, VariationalGarroteSAE
from .sae_sweep import METHOD_LABELS, RunSpec, SweepConfig


def _relative_error(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.sqrt((estimate - target).pow(2).sum() / target.pow(2).sum().clamp_min(eps)))


def _explained_variance(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    residual = (target - estimate).pow(2).sum(dim=1).mean()
    variance = (target.pow(2).sum(dim=1).mean() - target.mean(dim=0).pow(2).sum()).clamp_min(eps)
    return float(1.0 - residual / variance)


def _decoder_matching(model: torch.nn.Module, true_dictionary: torch.Tensor):
    from scipy.optimize import linear_sum_assignment

    inference = to_inference_sae(model, fold_decoder_norm=True) if isinstance(model, TrainingSAE) else model
    weight = inference.W_dec.T if hasattr(inference, "W_dec") else inference.decoder.weight
    learned = weight.detach().cpu()
    true = true_dictionary.detach().cpu()
    learned = learned / learned.norm(dim=0, keepdim=True).clamp_min(1e-12)
    true = true / true.norm(dim=0, keepdim=True).clamp_min(1e-12)
    signed_cosines = (learned.T @ true).numpy()
    learned_idx, true_idx = linear_sum_assignment(-np.abs(signed_cosines))
    signs = np.sign(signed_cosines[learned_idx, true_idx])
    signs[signs == 0] = 1
    recovery = float(np.abs(signed_cosines[learned_idx, true_idx]).mean())
    return learned_idx, true_idx, signs, recovery


def _inference_model(model: TrainingSAE, x: torch.Tensor) -> torch.nn.Module:
    return to_inference_sae(model, fold_decoder_norm=True).to(device=x.device, dtype=x.dtype)


def _align(values: torch.Tensor, learned_idx, true_idx, n_true: int, signs=None) -> np.ndarray:
    array = values.detach().cpu().numpy()
    aligned = np.zeros((array.shape[0], n_true), dtype=np.float64)
    selected = array[:, learned_idx]
    if signs is not None:
        selected *= np.asarray(signs)[None, :]
    aligned[:, true_idx] = selected
    return aligned


def _rectangular_union(
    values: torch.Tensor,
    target: torch.Tensor,
    learned_idx,
    true_idx,
    signs=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align matches, then retain unmatched truth (FN) and learned columns (FP)."""

    array = values.detach().cpu().numpy()
    truth = target.detach().cpu().numpy()
    learned_idx = np.asarray(learned_idx, dtype=np.int64)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    unmatched = np.setdiff1d(np.arange(array.shape[1]), learned_idx, assume_unique=True)
    union_width = truth.shape[1] + unmatched.size

    aligned = np.zeros((array.shape[0], union_width), dtype=np.float64)
    union_truth = np.zeros((truth.shape[0], union_width), dtype=np.float64)
    union_truth[:, : truth.shape[1]] = truth
    selected = array[:, learned_idx]
    if signs is not None:
        selected = selected * np.asarray(signs)[None, :]
    aligned[:, true_idx] = selected
    aligned[:, truth.shape[1] :] = array[:, unmatched]

    union_learned_idx = np.full(union_width, -1, dtype=np.int64)
    union_true_idx = np.full(union_width, -1, dtype=np.int64)
    union_learned_idx[true_idx] = learned_idx
    union_learned_idx[truth.shape[1] :] = unmatched
    union_true_idx[: truth.shape[1]] = np.arange(truth.shape[1])
    return aligned, union_truth, union_learned_idx, union_true_idx


def _l1_threshold(h: torch.Tensor) -> float:
    from sklearn.mixture import GaussianMixture

    values = h.detach().cpu().numpy().reshape(-1, 1)
    if values.size == 0 or np.nanmax(values) <= 0:
        return np.inf
    logged = np.log1p(values)
    if float(np.ptp(logged)) < 1e-8:
        return 0.0
    try:
        model = GaussianMixture(
            n_components=2,
            covariance_type="full",
            means_init=np.array([[0.0], [float(np.percentile(logged, 95))]]),
            random_state=0,
        ).fit(logged)
        means = np.sort(model.means_.reshape(-1))
        return float(max(np.expm1(0.5 * (means[0] + means[-1])), 0.0))
    except Exception:
        positive = values[values[:, 0] > 0, 0]
        return float(np.percentile(positive, 50)) if positive.size else np.inf


@torch.no_grad()
def _latents_and_masks(model: torch.nn.Module, x: torch.Tensor, l1_threshold=None):
    model.eval()
    if isinstance(model, VariationalGarroteSAE):
        output = model(x)
        return output["h"], output["m"], {"mask_family": "bernoulli_probability"}
    if isinstance(model, StandardSAE):
        h = _inference_model(model, x).encode(x)
        threshold = _l1_threshold(h) if l1_threshold is None else float(l1_threshold)
        return h, (h > threshold).to(h), {
            "mask_family": "gmm_relu_activation",
            "l1_gmm_threshold": threshold,
            "l1_raw_relu_density": float((h > 0).to(h).mean()),
        }
    if isinstance(model, TrainingSAE):
        h = _inference_model(model, x).encode(x)
        return h, h.gt(0).to(h), {"mask_family": model.cfg.architecture()}
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


@torch.no_grad()
def _reconstruct(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, TrainingSAE):
        return _inference_model(model, x)(x)
    return model(x)["x_hat"]


@torch.no_grad()
def _hard_latents(
    model: torch.nn.Module,
    x: torch.Tensor,
    native_h: torch.Tensor,
    mask_values: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return the operational hard code paired with thresholded mask density.

    VG's native code is the posterior mean ``m * a``.  Its hard inference code
    must instead be ``1[m >= threshold] * a``.  The L1 baseline's mask is the
    train-fitted GMM decision, so multiplying by that mask removes sub-threshold
    ReLU activations.  Other SAELens baselines already return zero outside their
    native hard support; the multiplication is therefore an explicit no-op.
    """

    hard_mask = (mask_values >= threshold).to(native_h)
    if isinstance(model, VariationalGarroteSAE):
        _, amplitudes, _ = model.encode(x)
        return hard_mask * amplitudes
    return hard_mask * native_h


@torch.no_grad()
def _decode_latents(model: torch.nn.Module, h: torch.Tensor) -> torch.Tensor:
    """Decode externally supplied latents with the evaluation-time decoder."""

    if isinstance(model, TrainingSAE):
        return _inference_model(model, h).decode(h)
    if isinstance(model, VariationalGarroteSAE):
        return model.decode(h)
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def _support_metrics(mask: np.ndarray, target: np.ndarray, threshold: float):
    from sklearn.metrics import average_precision_score, roc_auc_score

    pred, truth = mask >= threshold, target >= 0.5
    tp = np.logical_and(pred, truth).sum()
    fp = np.logical_and(pred, ~truth).sum()
    fn = np.logical_and(~pred, truth).sum()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    flat_truth = truth.reshape(-1).astype(int)
    if flat_truth.min() == flat_truth.max():
        average_precision = roc_auc = np.nan
    else:
        average_precision = average_precision_score(flat_truth, mask.reshape(-1))
        roc_auc = roc_auc_score(flat_truth, mask.reshape(-1))
    return map(float, (precision, recall, f1, average_precision, roc_auc))


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    train_data: Any,
    test_data: Any,
    config: SweepConfig,
    spec: RunSpec,
    run_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one trained checkpoint with notebook-07's exact conventions."""

    threshold = config.training.mask_threshold
    train_h, _, train_info = _latents_and_masks(model, train_data.x)
    test_h, test_mask, test_info = _latents_and_masks(
        model, test_data.x, l1_threshold=train_info.get("l1_gmm_threshold")
    )
    hard_test_h = _hard_latents(
        model,
        test_data.x,
        test_h,
        test_mask,
        threshold,
    )
    reconstruction = _reconstruct(model, test_data.x)
    hard_reconstruction = _decode_latents(model, hard_test_h)
    learned_idx, true_idx, signs, recovery = _decoder_matching(model, test_data.dictionary)
    mask, support, union_learned_idx, union_true_idx = _rectangular_union(
        test_mask, test_data.support, learned_idx, true_idx
    )
    h, z, _, _ = _rectangular_union(test_h, test_data.z, learned_idx, true_idx, signs)
    hard_h, _, _, _ = _rectangular_union(
        hard_test_h, test_data.z, learned_idx, true_idx, signs
    )
    raw_mask = test_mask.detach().cpu().numpy()
    raw_h = test_h.detach().cpu().numpy()
    ground_truth_support = test_data.support.detach().cpu().numpy()
    ground_truth_z = test_data.z.detach().cpu().numpy()
    sae_width = raw_mask.shape[1]
    ground_truth_num_features = ground_truth_support.shape[1]
    matched_latent_count = len(learned_idx)
    union_width = mask.shape[1]
    data_config = getattr(config, "data", None)
    probabilities = getattr(test_data, "feature_probabilities", None)
    expected_true_l0 = (
        float(probabilities.sum())
        if probabilities is not None
        else float(ground_truth_support.sum(1).mean())
    )
    empirical_true_l0 = float(ground_truth_support.sum(1).mean())
    precision, recall, f1, average_precision, roc_auc = _support_metrics(mask, support, threshold)
    active = z > 1e-8
    amplitude_ratio = h[active] / np.maximum(z[active], 1e-8) if active.any() else np.array([])
    hard_amplitude_ratio = (
        hard_h[active] / np.maximum(z[active], 1e-8)
        if active.any()
        else np.array([])
    )
    hard_mask = (mask >= threshold).astype(np.float64)
    (
        hard_precision,
        hard_recall,
        hard_f1,
        hard_average_precision,
        hard_roc_auc,
    ) = _support_metrics(hard_mask, support, 0.5)
    hard_generalization_error = float(
        np.sqrt(np.sum((hard_h - z) ** 2) / max(np.sum(z**2), 1e-12))
    )
    hard_code_kind = {
        "vgsae": "posterior_thresholded_amplitude",
        "l1": "gmm_thresholded_relu",
    }.get(spec.method, "native_nonzero_activation")

    row: dict[str, Any] = {
        "run_id": run_id or spec.run_id,
        "seed": spec.seed,
        "init_seed": spec.init_seed,
        "method": spec.method,
        "method_label": METHOD_LABELS[spec.method],
        "control_name": spec.control_name,
        "control_value": spec.control_value,
        "input_dim": int(test_data.x.shape[1]),
        "support_density": (
            float(data_config.support_density)
            if data_config is not None
            else float(ground_truth_support.mean())
        ),
        "amplitude_mode": (
            str(data_config.amplitude_mode)
            if data_config is not None
            else "unknown"
        ),
        "amplitude_scale": (
            float(data_config.amplitude_scale)
            if data_config is not None
            else float("nan")
        ),
        "frequency_skew": (
            float(data_config.frequency_skew)
            if data_config is not None
            else float("nan")
        ),
        "train_steps": config.training.train_steps,
        "dead_feature_window": config.training.dead_feature_window,
        "rho_model": float(raw_mask.mean()),
        "sae_width": sae_width,
        "ground_truth_num_features": ground_truth_num_features,
        "ground_truth_expected_l0": expected_true_l0,
        "ground_truth_empirical_l0": empirical_true_l0,
        "target_model_density": expected_true_l0 / sae_width,
        "target_model_density_expected": expected_true_l0 / sae_width,
        "target_model_density_empirical": empirical_true_l0 / sae_width,
        "matched_latent_count": matched_latent_count,
        "union_width": union_width,
        "unmatched_ground_truth_features": ground_truth_num_features - matched_latent_count,
        "unmatched_sae_latents": sae_width - matched_latent_count,
        "ground_truth_match_coverage": matched_latent_count / ground_truth_num_features,
        "sae_latent_match_coverage": matched_latent_count / sae_width,
        "matching_policy": "rectangular_hungarian_union",
        "generalization_error": float(np.sqrt(np.sum((h - z) ** 2) / max(np.sum(z**2), 1e-12))),
        "hard_generalization_error": hard_generalization_error,
        "reconstruction_error": _relative_error(reconstruction, test_data.x),
        "clean_reconstruction_error": _relative_error(reconstruction, test_data.clean_x),
        "explained_variance": _explained_variance(reconstruction, test_data.x),
        "reconstruction_mse": float((reconstruction - test_data.x).pow(2).mean()),
        "selection_error": selection_error(mask, support),
        "hard_reconstruction_error": _relative_error(hard_reconstruction, test_data.x),
        "hard_clean_reconstruction_error": _relative_error(
            hard_reconstruction, test_data.clean_x
        ),
        "hard_explained_variance": _explained_variance(
            hard_reconstruction, test_data.x
        ),
        "hard_reconstruction_mse": float(
            (hard_reconstruction - test_data.x).pow(2).mean()
        ),
        "hard_selection_error": selection_error(hard_mask, support),
        "hard_support_precision": hard_precision,
        "hard_support_recall": hard_recall,
        "hard_support_f1": hard_f1,
        "hard_support_average_precision": hard_average_precision,
        "hard_support_roc_auc": hard_roc_auc,
        "hard_paper_style_sigma_sel": selection_uncertainty(hard_mask),
        "hard_metric_schema_version": 1,
        "hard_code_threshold": float(threshold),
        "hard_code_kind": hard_code_kind,
        "mask_uncertainty": float(np.mean(mask * (1 - mask))),
        "paper_style_sigma_sel": selection_uncertainty(mask),
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "support_average_precision": average_precision,
        "support_roc_auc": roc_auc,
        "decoder_recovery_cosine": recovery * matched_latent_count / union_width,
        "matched_decoder_recovery_cosine": recovery,
        "decoder_recovery_ground_truth": (
            recovery * matched_latent_count / ground_truth_num_features
        ),
        "decoder_recovery_sae": recovery * matched_latent_count / sae_width,
        "dead_fraction": float((train_h.mean(0) <= config.training.dead_threshold).float().mean()),
        "amplitude_shrinkage": float(amplitude_ratio.mean()) if amplitude_ratio.size else np.nan,
        "hard_amplitude_shrinkage": (
            float(hard_amplitude_ratio.mean()) if hard_amplitude_ratio.size else np.nan
        ),
        "mean_activation": float(test_h.mean()),
        "hard_mean_activation": float(hard_test_h.mean()),
        "average_l0": float((raw_mask >= threshold).sum(1).mean()),
        "rho_model_hard": float((raw_mask >= threshold).mean()),
        "expected_l0": float(raw_mask.sum(1).mean()),
    }
    row.update({key: value for key, value in train_info.items() if not isinstance(value, str)})
    row.update({key: value for key, value in test_info.items() if not isinstance(value, str)})
    if spec.method == "l1":
        raw = (test_h > 0).to(test_h)
        raw_union, _, _, _ = _rectangular_union(
            raw, test_data.support, learned_idx, true_idx
        )
        row.update(
            l1_raw_relu_rho_model=float(raw.float().mean()),
            l1_raw_relu_selection_error=selection_error(raw_union, support),
            l1_raw_relu_sigma_sel=selection_uncertainty(raw_union),
        )
    else:
        row.update(
            l1_raw_relu_rho_model=np.nan,
            l1_raw_relu_selection_error=np.nan,
            l1_raw_relu_sigma_sel=np.nan,
        )
    return row, {
        "mask": mask,
        "true_support": support,
        "h": h,
        "hard_mask": hard_mask.astype(np.uint8),
        "true_latents": z,
        "raw_mask": raw_mask,
        "raw_h": raw_h,
        "ground_truth_support": ground_truth_support,
        "ground_truth_latents": ground_truth_z,
        "matched_learned_idx": np.asarray(learned_idx, dtype=np.int64),
        "matched_true_idx": np.asarray(true_idx, dtype=np.int64),
        "matched_signs": np.asarray(signs),
        "union_learned_idx": union_learned_idx,
        "union_true_idx": union_true_idx,
    }
