"""Streaming Stage-3 metrics for SAEs trained on real model activations.

Unlike the synthetic evaluators, this module never assumes that a ground-truth
dictionary, support, or latent code exists.  It consumes flat activation batches
until an exact token budget is reached and retains only a bounded preview.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sae_lens import TrainingSAE
from sae_lens.evals import ExplainedVarianceCalculator

from .sae_baselines import to_inference_sae
from .sae_evaluate import decoder_pairwise_cosine_similarity
from .saelens_vg import VGSAE


_MISSING = object()
_QUANTILES = {
    "q10": 0.10,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q90": 0.90,
}
_METHOD_LABELS = {
    "vgsae": "VG-SAE",
    "l1": "L1/ReLU SAE",
    "batchtopk": "BatchTopK SAE",
    "jumprelu": "JumpReLU SAE",
}

# These Stage-1/2 columns require synthetic labels or a known feature dictionary.
# Keeping them in the real-activation row avoids silently changing the table schema.
_GROUND_TRUTH_ONLY_METRICS = (
    "support_density",
    "amplitude_mode",
    "amplitude_scale",
    "frequency_skew",
    "true_l0",
    "ground_truth_num_features",
    "ground_truth_expected_l0",
    "ground_truth_empirical_l0",
    "true_l0_over_d_sae",
    "true_feature_density",
    "target_model_density",
    "target_model_density_expected",
    "target_model_density_empirical",
    "matched_latent_count",
    "union_width",
    "unmatched_ground_truth_features",
    "unmatched_sae_latents",
    "ground_truth_match_coverage",
    "sae_latent_match_coverage",
    "support_precision",
    "support_recall",
    "support_f1",
    "support_average_precision",
    "support_roc_auc",
    "hard_support_precision",
    "hard_support_recall",
    "hard_support_f1",
    "hard_support_average_precision",
    "hard_support_roc_auc",
    "selection_error",
    "hard_selection_error",
    "paper_style_sigma_sel",
    "hard_paper_style_sigma_sel",
    "decoder_recovery_cosine",
    "matched_decoder_recovery_cosine",
    "decoder_recovery_ground_truth",
    "decoder_recovery_sae",
    "mcc",
    "uniqueness",
    "classification_precision",
    "classification_recall",
    "classification_f1",
    "classification_accuracy",
    "classifier_precision",
    "classifier_recall",
    "classifier_f1",
    "classifier_accuracy",
    "generalization_error",
    "hard_generalization_error",
    "clean_reconstruction_error",
    "hard_clean_reconstruction_error",
    "amplitude_shrinkage",
    "hard_amplitude_shrinkage",
    "l1_raw_relu_selection_error",
    "l1_raw_relu_sigma_sel",
    "matching_policy",
)


def _get_path(source: Any, path: str) -> Any:
    value = source
    for part in path.split("."):
        if value is None:
            return _MISSING
        if isinstance(value, Mapping):
            value = value.get(part, _MISSING)
        else:
            value = getattr(value, part, _MISSING)
        if value is _MISSING:
            return _MISSING
    return value


def _first_value(*candidates: tuple[Any, str]) -> Any:
    for source, path in candidates:
        value = _get_path(source, path)
        if value is not _MISSING and value is not None:
            return value
    return None


def _json_safe(value: Any) -> Any:
    """Convert config/spec scalars to values accepted by strict ``json.dumps``."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return _json_safe(value.detach().cpu().item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    enum_value = getattr(value, "value", _MISSING)
    if enum_value is not _MISSING:
        return _json_safe(enum_value)
    return str(value)


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        lower_bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {lower_bound} integer.")
    result = int(value)
    if result < 0 or (result == 0 and not allow_zero):
        lower_bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {lower_bound} integer.")
    return result


def _resolve_budgets(
    config: Any,
    n_eval_tokens: int | None,
    preview_tokens: int | None,
) -> tuple[int, int]:
    if n_eval_tokens is None:
        n_eval_tokens = _first_value(
            (config, "data.n_eval"),
            (config, "data.n_eval_tokens"),
            (config, "evaluation.n_eval_tokens"),
            (config, "n_eval_tokens"),
        )
    if preview_tokens is None:
        preview_tokens = _first_value(
            (config, "training.preview_tokens"),
            (config, "training.heatmap_samples"),
            (config, "evaluation.preview_tokens"),
            (config, "preview_tokens"),
        )
    if n_eval_tokens is None:
        raise ValueError(
            "n_eval_tokens is required (or provide config.data.n_eval)."
        )
    if preview_tokens is None:
        raise ValueError(
            "preview_tokens is required (or provide config.training.preview_tokens)."
        )
    return (
        _positive_int(n_eval_tokens, "n_eval_tokens"),
        _positive_int(preview_tokens, "preview_tokens", allow_zero=True),
    )


def _identity_row(
    training_model: TrainingSAE[Any],
    config: Any,
    spec: Any,
    identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    explicit = {} if identity is None else dict(identity)
    row = {str(key): _json_safe(value) for key, value in explicit.items()}

    def resolve(name: str, *paths: str) -> Any:
        candidates: list[tuple[Any, str]] = [(explicit, name)]
        candidates.extend((explicit, path) for path in paths)
        candidates.extend((spec, path) for path in (name, *paths))
        candidates.extend((config, path) for path in (name, *paths))
        return _json_safe(_first_value(*candidates))

    model_id = _json_safe(
        _first_value(
            (explicit, "model_id"),
            (explicit, "model_name"),
            (explicit, "model"),
            (spec, "model_id"),
            (spec, "model_name"),
            (spec, "model"),
            (config, "model_id"),
            (config, "model_name"),
            (config, "data.model_id"),
            (config, "data.model_name"),
            (config, "model.model_id"),
            (config, "model.name"),
        )
    )
    if model_id is None:
        model_id = _json_safe(
            _first_value(
                (training_model, "cfg.metadata.model_name"),
                (training_model, "cfg.metadata.model_id"),
            )
        )
    hook_name = resolve(
        "hook_name",
        "hook",
        "data.hook_name",
        "model.hook_name",
    )
    if hook_name is None:
        hook_name = _json_safe(
            _first_value((training_model, "cfg.metadata.hook_name"))
        )

    canonical = {
        "run_id": resolve("run_id"),
        "method": resolve("method"),
        "method_label": resolve("method_label"),
        "model": model_id,
        "model_id": model_id,
        "model_name": model_id,
        "target_name": resolve("target_name", "data.target_name"),
        "model_revision": resolve("model_revision", "data.model_revision"),
        "layer": resolve(
            "layer", "layer_index", "hook_layer", "data.layer", "model.layer"
        ),
        "hook_name": hook_name,
        "paper_hook_name": resolve("paper_hook_name", "data.paper_hook_name"),
        "dataset_id": resolve("dataset_id", "data.dataset_id"),
        "dataset_revision": resolve("dataset_revision", "data.dataset_revision"),
        "control_name": resolve("control_name"),
        "control_value": resolve("control_value"),
        "seed": resolve("seed"),
        "init_seed": resolve("init_seed"),
        "calibration_seed": resolve("calibration_seed"),
        "train_stream_seed": resolve("train_stream_seed"),
        "eval_stream_seed": resolve("eval_stream_seed", "evaluation_seed"),
        "beta_mode": resolve("beta_mode", "training.beta_mode"),
        "beta_initial": resolve("beta_initial", "beta", "training.beta"),
        "saelens_revision": resolve("saelens_revision"),
    }
    for key, value in canonical.items():
        row[key] = value
    if row["method_label"] is None:
        row["method_label"] = _METHOD_LABELS.get(row["method"], row["method"])
    return row


def _ground_truth_unavailable_row() -> dict[str, Any]:
    row = {metric: None for metric in _GROUND_TRUTH_ONLY_METRICS}
    row.update(
        ground_truth_available=False,
        ground_truth_unavailable_reason=(
            "Real model activations do not provide a ground-truth sparse "
            "dictionary, support labels, or latent coefficients."
        ),
        support_metrics_available=False,
        support_metrics_unavailable_reason=(
            "Ground-truth feature support is unavailable for real activations."
        ),
        decoder_recovery_available=False,
        decoder_recovery_unavailable_reason=(
            "A ground-truth feature dictionary is unavailable for real activations."
        ),
        mcc_available=False,
        mcc_unavailable_reason=(
            "MCC requires matching learned decoders to a ground-truth dictionary."
        ),
        uniqueness_available=False,
        uniqueness_unavailable_reason=(
            "Uniqueness requires ground-truth decoder matches."
        ),
        classifier_metrics_available=False,
        classifier_metrics_unavailable_reason=(
            "Matched ground-truth feature labels are unavailable."
        ),
        generalization_metrics_available=False,
        generalization_metrics_unavailable_reason=(
            "Ground-truth sparse latent coefficients are unavailable."
        ),
    )
    return row


def _model_device_and_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device, parameter.dtype
    device = torch.device(getattr(model, "device", "cpu"))
    dtype = getattr(model, "dtype", torch.float32)
    return device, dtype if isinstance(dtype, torch.dtype) else torch.float32


def _cat_preview(
    values: list[torch.Tensor], *, shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    if values:
        return torch.cat(values, dim=0).numpy()
    return np.empty((0, *shape), dtype=dtype)


@torch.no_grad()
def evaluate_model(
    training_model: TrainingSAE[Any],
    data_provider: Iterable[torch.Tensor],
    config: Any | None = None,
    spec: Any | None = None,
    *,
    n_eval_tokens: int | None = None,
    preview_tokens: int | None = None,
    identity: Mapping[str, Any] | None = None,
    decoder_pairwise_block_size: int = 256,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate a loaded training SAE over exactly ``n_eval_tokens`` activations.

    ``config`` and ``spec`` are intentionally duck typed so the evaluator stays
    independent of the Stage-3 runner.  The preferred config fallbacks are
    ``config.data.n_eval`` and ``config.training.preview_tokens``; callers may
    instead pass both budgets explicitly.
    """

    n_eval_tokens, preview_tokens = _resolve_budgets(
        config, n_eval_tokens, preview_tokens
    )
    decoder_pairwise_block_size = _positive_int(
        decoder_pairwise_block_size, "decoder_pairwise_block_size"
    )
    if not isinstance(training_model, TrainingSAE):
        raise TypeError("training_model must be an SAELens TrainingSAE.")

    # Exporting through SAELens is important for BatchTopK: its learned batch
    # threshold becomes the inference JumpReLU threshold before evaluation.
    model = to_inference_sae(training_model, fold_decoder_norm=True)
    model.eval()
    device, dtype = _model_device_and_dtype(model)
    width = int(model.cfg.d_sae)
    input_dim = int(model.cfg.d_in)
    decoder_pairwise_cosine = decoder_pairwise_cosine_similarity(
        model.W_dec, block_size=decoder_pairwise_block_size
    )
    is_vg = isinstance(model, VGSAE)

    explained_variance = ExplainedVarianceCalculator()
    expected_explained_variance = ExplainedVarianceCalculator()
    ever_fired = torch.zeros(width, dtype=torch.bool, device=device)

    reconstruction_sq_error = 0.0
    expected_sq_error = 0.0
    input_sq_sum = 0.0
    output_sq_sum = 0.0
    input_output_dot_sum = 0.0
    hard_l0_sum = hard_l1_sum = 0.0
    input_l2_sum = output_l2_sum = l2_ratio_sum = 0.0
    reconstruction_cosine_sum = 0.0
    expected_cosine_sum = 0.0
    posterior_sum = posterior_variance_sum = 0.0

    preview_inputs: list[torch.Tensor] = []
    preview_reconstructions: list[torch.Tensor] = []
    preview_latents: list[torch.Tensor] = []
    preview_masks: list[torch.Tensor] = []
    preview_probabilities: list[torch.Tensor] = []
    preview_expected_codes: list[torch.Tensor] = []
    preview_expected_reconstructions: list[torch.Tensor] = []
    preview_remaining = min(preview_tokens, n_eval_tokens)

    iterator = iter(data_provider)
    processed = 0
    while processed < n_eval_tokens:
        try:
            raw_batch = next(iterator)
        except StopIteration as exc:
            raise ValueError(
                "data_provider was exhausted after "
                f"{processed} tokens; exactly {n_eval_tokens} were requested."
            ) from exc
        if not isinstance(raw_batch, torch.Tensor):
            raise TypeError("data_provider must yield torch.Tensor batches.")
        if raw_batch.ndim != 2:
            raise ValueError(
                "Activation batches must be flat with shape (tokens, d_in); "
                f"received {tuple(raw_batch.shape)}."
            )
        if raw_batch.shape[0] == 0:
            raise ValueError("data_provider yielded an empty activation batch.")
        if raw_batch.shape[1] != input_dim:
            raise ValueError(
                f"Expected activation d_in={input_dim}, got {raw_batch.shape[1]}."
            )

        current = min(int(raw_batch.shape[0]), n_eval_tokens - processed)
        batch = raw_batch[:current].to(device=device, dtype=dtype)

        # Always use the exported inference encode path for the operational hard
        # code.  In particular, VGSAE.encode applies its posterior threshold and
        # BatchTopK has already exported to its calibrated JumpReLU inference SAE.
        hard_latents = model.encode(batch)
        if hard_latents.layout != torch.strided:
            hard_latents = hard_latents.to_dense()
        hard_reconstruction = model.decode(hard_latents)
        if hard_latents.shape != (current, width):
            raise ValueError(
                "SAE encode returned shape "
                f"{tuple(hard_latents.shape)}, expected {(current, width)}."
            )
        if hard_reconstruction.shape != batch.shape:
            raise ValueError(
                "SAE decode returned shape "
                f"{tuple(hard_reconstruction.shape)}, expected {tuple(batch.shape)}."
            )

        metric_input = batch.float()
        metric_output = hard_reconstruction.float()
        metric_latents = hard_latents.float()
        fires = metric_latents > 0
        residual = metric_output - metric_input
        input_l2 = metric_input.norm(dim=-1)
        output_l2 = metric_output.norm(dim=-1)
        ratio_denominator = input_l2.clone()
        ratio_denominator[ratio_denominator.abs() < 1.0e-4] = 1.0

        explained_variance.add_batch(metric_output, metric_input)
        reconstruction_sq_error += float(residual.pow(2).sum().cpu())
        input_sq_sum += float(metric_input.pow(2).sum().cpu())
        output_sq_sum += float(metric_output.pow(2).sum().cpu())
        input_output_dot_sum += float((metric_input * metric_output).sum().cpu())
        hard_l0_sum += float(fires.sum().cpu())
        hard_l1_sum += float(metric_latents.abs().sum().cpu())
        ever_fired |= fires.any(dim=0)
        input_l2_sum += float(input_l2.sum().cpu())
        output_l2_sum += float(output_l2.sum().cpu())
        l2_ratio_sum += float((output_l2 / ratio_denominator).sum().cpu())
        reconstruction_cosine_sum += float(
            F.cosine_similarity(metric_input, metric_output, dim=-1, eps=1.0e-8)
            .sum()
            .cpu()
        )

        probabilities: torch.Tensor | None = None
        expected_code: torch.Tensor | None = None
        expected_reconstruction: torch.Tensor | None = None
        if is_vg:
            posterior = model.posterior(batch)
            probabilities = posterior["m"]
            expected_code = posterior["expected_code"]
            expected_reconstruction = model.decode(expected_code)
            metric_probabilities = probabilities.float()
            metric_expected = expected_reconstruction.float()
            expected_residual = metric_expected - metric_input

            posterior_sum += float(metric_probabilities.sum().cpu())
            posterior_variance_sum += float(
                (metric_probabilities * (1.0 - metric_probabilities)).sum().cpu()
            )
            expected_sq_error += float(expected_residual.pow(2).sum().cpu())
            expected_explained_variance.add_batch(metric_expected, metric_input)
            expected_cosine_sum += float(
                F.cosine_similarity(
                    metric_input, metric_expected, dim=-1, eps=1.0e-8
                )
                .sum()
                .cpu()
            )

        if preview_remaining:
            take = min(preview_remaining, current)
            preview_inputs.append(metric_input[:take].detach().cpu())
            preview_reconstructions.append(metric_output[:take].detach().cpu())
            preview_latents.append(metric_latents[:take].detach().cpu())
            preview_masks.append(fires[:take].detach().cpu())
            if (
                probabilities is not None
                and expected_code is not None
                and expected_reconstruction is not None
            ):
                preview_probabilities.append(
                    probabilities[:take].detach().float().cpu()
                )
                preview_expected_codes.append(
                    expected_code[:take].detach().float().cpu()
                )
                preview_expected_reconstructions.append(
                    expected_reconstruction[:take].detach().float().cpu()
                )
            preview_remaining -= take
        processed += current

    element_count = processed * input_dim
    hard_l0 = hard_l0_sum / processed
    hard_l1 = hard_l1_sum / processed
    hard_ev = float(explained_variance.compute())
    relative_error = math.sqrt(
        reconstruction_sq_error / max(input_sq_sum, 1.0e-12)
    )
    relative_bias = (
        output_sq_sum / input_output_dot_sum
        if abs(input_output_dot_sum) > 1.0e-12
        else None
    )

    row = _identity_row(training_model, config, spec, identity)
    row.update(
        {
            "input_dim": input_dim,
            "sae_width": width,
            "n_evaluation_tokens": processed,
            "n_evaluation_samples": processed,
            "n_training_tokens": _json_safe(
                _first_value((config, "data.n_train_tokens"))
            ),
            "n_training_samples": _json_safe(
                _first_value((config, "data.n_train_tokens"))
            ),
            "train_steps": _json_safe(
                _first_value((config, "total_training_steps"))
            ),
            "dead_feature_window": _json_safe(
                _first_value((config, "training.dead_feature_window"))
            ),
            "hard_code_kind": "inference_nonzero_activation",
            "hard_code_threshold": (
                float(model.cfg.inference_threshold) if is_vg else None
            ),
            "hard_metric_schema_version": 1,
            "reconstruction_mse": reconstruction_sq_error / element_count,
            "hard_reconstruction_mse": reconstruction_sq_error / element_count,
            "reconstruction_error": relative_error,
            "reconstruction_relative_error": relative_error,
            "hard_reconstruction_error": relative_error,
            "hard_reconstruction_relative_error": relative_error,
            "explained_variance": hard_ev,
            "hard_explained_variance": hard_ev,
            "reconstruction_cosine": reconstruction_cosine_sum / processed,
            "reconstruction_cosine_similarity": (
                reconstruction_cosine_sum / processed
            ),
            "hard_reconstruction_cosine": reconstruction_cosine_sum / processed,
            "sae_l0": hard_l0,
            "average_l0": hard_l0,
            "l0": hard_l0,
            "hard_l0": hard_l0,
            "sae_l1": hard_l1,
            "average_l1": hard_l1,
            "l1": hard_l1,
            "hard_l1": hard_l1,
            "mean_activation": hard_l1 / width,
            "hard_mean_activation": hard_l1 / width,
            "rho_model": hard_l0 / width,
            "density": hard_l0 / width,
            "rho_model_hard": hard_l0 / width,
            "hard_density": hard_l0 / width,
            "dead_latents": int((~ever_fired).sum().cpu()),
            "dead_fraction": float((~ever_fired).float().mean().cpu()),
            "dead_latent_basis": "evaluation_hard_support",
            "input_l2_mean": input_l2_sum / processed,
            "output_l2_mean": output_l2_sum / processed,
            "l2_norm_in": input_l2_sum / processed,
            "l2_norm_out": output_l2_sum / processed,
            "l2_ratio": l2_ratio_sum / processed,
            "output_input_l2_ratio": l2_ratio_sum / processed,
            "shrinkage": l2_ratio_sum / processed,
            "relative_reconstruction_bias": relative_bias,
            "decoder_pairwise_cosine_similarity": decoder_pairwise_cosine,
            "mask_uncertainty": (
                posterior_variance_sum / (processed * width) if is_vg else 0.0
            ),
            "l1_raw_relu_rho_model": (
                hard_l0 / width if row.get("method") == "l1" else None
            ),
            # The official Stage-3 Standard SAE uses its native positive ReLU
            # code at inference; no Stage-1 fitted GMM threshold is involved.
            "l1_gmm_threshold": None,
            "l1_raw_relu_density": (
                hard_l0 / width if row.get("method") == "l1" else None
            ),
            "l1_gmm_threshold_unavailable_reason": (
                "Stage-3 evaluates the official native ReLU code without a "
                "fitted Stage-1 GMM support threshold."
            ),
        }
    )
    row.update(_ground_truth_unavailable_row())

    vg_fields = {
        "vg_expected_reconstruction_mse": None,
        "vg_expected_mse": None,
        "vg_expected_reconstruction_error": None,
        "vg_expected_reconstruction_relative_error": None,
        "vg_expected_relative_error": None,
        "vg_expected_explained_variance": None,
        "vg_expected_reconstruction_cosine": None,
        "vg_expected_l0": None,
        "vg_expected_density": None,
        "vg_posterior_rho": None,
        "vg_posterior_variance": None,
        "vg_expected_to_hard_l0_ratio": None,
        "vg_expected_hard_ev_gap": None,
        "vg_posterior_probability_quantile_sample_count": None,
        "vg_posterior_probability_quantile_source": None,
    }
    vg_fields.update(
        {f"vg_posterior_probability_{name}": None for name in _QUANTILES}
    )
    row.update(vg_fields)

    probability_preview: torch.Tensor | None = None
    if is_vg:
        expected_l0 = posterior_sum / processed
        expected_ev = float(expected_explained_variance.compute())
        expected_relative_error = math.sqrt(
            expected_sq_error / max(input_sq_sum, 1.0e-12)
        )
        row.update(
            {
                "vg_expected_reconstruction_mse": expected_sq_error / element_count,
                "vg_expected_mse": expected_sq_error / element_count,
                "vg_expected_reconstruction_error": expected_relative_error,
                "vg_expected_reconstruction_relative_error": (
                    expected_relative_error
                ),
                "vg_expected_relative_error": expected_relative_error,
                "vg_expected_explained_variance": expected_ev,
                "vg_expected_reconstruction_cosine": expected_cosine_sum / processed,
                "vg_expected_l0": expected_l0,
                "vg_expected_density": expected_l0 / width,
                "vg_posterior_rho": expected_l0 / width,
                "vg_posterior_variance": posterior_variance_sum / (processed * width),
                "vg_expected_to_hard_l0_ratio": (
                    expected_l0 / hard_l0 if hard_l0 > 1.0e-12 else None
                ),
                "vg_expected_hard_ev_gap": expected_ev - hard_ev,
            }
        )
        if preview_probabilities:
            probability_preview = torch.cat(preview_probabilities, dim=0)
            flattened_probability_preview = probability_preview.flatten()
            row.update(
                {
                    f"vg_posterior_probability_{name}": float(
                        torch.quantile(flattened_probability_preview, level)
                    )
                    for name, level in _QUANTILES.items()
                }
            )
            row["vg_posterior_probability_quantile_sample_count"] = int(
                flattened_probability_preview.numel()
            )
            row["vg_posterior_probability_quantile_source"] = "capped_preview"

    # Stage-2's schema uses expected_l0 for every method; for deterministic
    # baseline posteriors it is exactly the operational hard L0.
    row["expected_l0"] = row["vg_expected_l0"] if is_vg else hard_l0
    row["expected_density"] = (
        row["vg_expected_density"] if is_vg else hard_l0 / width
    )
    row = {key: _json_safe(value) for key, value in row.items()}

    preview_count = min(preview_tokens, processed)
    cache: dict[str, np.ndarray] = {
        "input": _cat_preview(
            preview_inputs, shape=(input_dim,), dtype=np.dtype(np.float32)
        ),
        "reconstruction": _cat_preview(
            preview_reconstructions,
            shape=(input_dim,),
            dtype=np.dtype(np.float32),
        ),
        "h": _cat_preview(
            preview_latents, shape=(width,), dtype=np.dtype(np.float32)
        ),
        "mask": _cat_preview(
            preview_masks, shape=(width,), dtype=np.dtype(np.bool_)
        ),
        "preview_token_count": np.asarray(preview_count, dtype=np.int64),
        "preview_sample_count": np.asarray(preview_count, dtype=np.int64),
    }
    if is_vg:
        cache.update(
            posterior_probability=(
                probability_preview.numpy()
                if probability_preview is not None
                else np.empty((0, width), dtype=np.float32)
            ),
            expected_h=_cat_preview(
                preview_expected_codes,
                shape=(width,),
                dtype=np.dtype(np.float32),
            ),
            expected_reconstruction=_cat_preview(
                preview_expected_reconstructions,
                shape=(input_dim,),
                dtype=np.dtype(np.float32),
            ),
        )
    return row, cache


evaluate_real_activations = evaluate_model

__all__ = ["evaluate_model", "evaluate_real_activations"]
