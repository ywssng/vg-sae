"""Training utilities for VG-SAE and baseline sparse autoencoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .sae_loss import sae_loss_terms
from .sae_model import (
    BatchTopKSAE,
    BatchTopKSAEConfig,
    GatedSAE,
    GatedSAEConfig,
    JumpReLUSAE,
    JumpReLUSAEConfig,
    L1ReLUSAE,
    L1SAEConfig,
    TopKSAE,
    TopKSAEConfig,
    VGSAEConfig,
    VariationalGarroteSAE,
)
from .utils import set_seed


@dataclass
class SAETrainResult:
    model: torch.nn.Module
    history: list[dict[str, float]]


def _history_row(step: int, terms: Any, lr: float) -> dict[str, float]:
    row = {"step": float(step), "loss": float(terms.loss.detach().cpu()), "lr": float(lr)}
    for name in (
        "reconstruction_mse",
        "reconstruction_loss",
        "variance_loss",
        "prior_loss",
        "entropy_loss",
        "sparsity_loss",
        "auxiliary_loss",
        "entropy",
        "rho",
        "v_eff",
        "beta",
    ):
        if hasattr(terms, name):
            value = getattr(terms, name)
            row[name] = (
                float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            )
    model_threshold = getattr(terms, "threshold", None)
    if model_threshold is not None:
        row["threshold"] = float(model_threshold.detach().mean().cpu())
    return row


def fit_sae(
    model: torch.nn.Module,
    x: torch.Tensor,
    lr: float = 1.0e-3,
    batch_size: int = 256,
    max_steps: int = 1000,
    weight_decay: float = 0.0,
    gradient_clip_norm: float | None = 1.0,
    history_every: int = 50,
    dead_feature_window: int = 1000,
    seed: int = 0,
    verbose: bool = False,
) -> SAETrainResult:
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (n_samples, input_dim), got {tuple(x.shape)}.")
    if dead_feature_window <= 0:
        raise ValueError("dead_feature_window must be positive.")
    set_seed(seed)
    model.to(device=x.device, dtype=x.dtype)
    if isinstance(model, BatchTopKSAE):
        model.topk_threshold = model.topk_threshold.double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(x),
        batch_size=min(batch_size, x.shape[0]),
        shuffle=True,
        drop_last=False,
    )
    iterator = iter(loader)
    history: list[dict[str, float]] = []
    last_terms = None
    steps_since_fired = torch.zeros(
        model.config.n_latents, dtype=torch.long, device=x.device  # type: ignore[attr-defined]
    )
    model.train()

    for step in range(max_steps):
        try:
            (batch,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (batch,) = next(iterator)

        optimizer.zero_grad(set_to_none=True)
        dead_feature_mask = steps_since_fired > dead_feature_window
        terms = sae_loss_terms(model, batch, dead_feature_mask)
        feature_acts = getattr(terms, "feature_acts", None)
        if feature_acts is not None:
            with torch.no_grad():
                steps_since_fired.add_(1)
                steps_since_fired[feature_acts.gt(0).any(dim=0)] = 0
                if isinstance(model, BatchTopKSAE):
                    model.update_topk_threshold(feature_acts)
        terms.loss.backward()
        should_normalize_decoder = hasattr(model, "normalize_decoder_columns") and (
            not isinstance(model, VariationalGarroteSAE) or model.config.normalize_decoder
        )
        if should_normalize_decoder and hasattr(model, "remove_decoder_parallel_grad"):
            model.remove_decoder_parallel_grad()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
        optimizer.step()
        if should_normalize_decoder:
            model.normalize_decoder_columns()
        last_terms = terms

        if step % history_every == 0 or step == max_steps - 1:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                full_terms = sae_loss_terms(model, x, steps_since_fired > dead_feature_window)
            if was_training:
                model.train()
            row = _history_row(step, full_terms, float(optimizer.param_groups[0]["lr"]))
            if isinstance(model, BatchTopKSAE):
                row["threshold"] = float(model.topk_threshold.cpu())
            history.append(row)
            if verbose:
                print(
                    f"step={step:5d} loss={row['loss']:.6g} "
                    f"mse={row.get('reconstruction_mse', np.nan):.6g} "
                    f"rho={row.get('rho', np.nan):.4f}"
                )

    if not history and last_terms is not None:
        lr_value = float(optimizer.param_groups[0]["lr"])
        history.append(_history_row(max_steps - 1, last_terms, lr_value))
    model.eval()
    return SAETrainResult(model=model, history=history)


def build_sae(kind: str, input_dim: int, n_latents: int, **kwargs: Any) -> torch.nn.Module:
    normalized = kind.lower()
    if normalized in {"vg", "vg-sae", "vg_sae"}:
        return VariationalGarroteSAE(
            VGSAEConfig(input_dim=input_dim, n_latents=n_latents, **kwargs)
        )
    if normalized in {"l1", "l1-relu", "l1_relu"}:
        return L1ReLUSAE(L1SAEConfig(input_dim=input_dim, n_latents=n_latents, **kwargs))
    if normalized in {"topk", "top-k", "top_k"}:
        return TopKSAE(TopKSAEConfig(input_dim=input_dim, n_latents=n_latents, **kwargs))
    if normalized in {"batchtopk", "batch-topk", "batch_topk"}:
        return BatchTopKSAE(BatchTopKSAEConfig(input_dim=input_dim, n_latents=n_latents, **kwargs))
    if normalized in {"jumprelu", "jump-relu", "jump_relu"}:
        return JumpReLUSAE(JumpReLUSAEConfig(input_dim=input_dim, n_latents=n_latents, **kwargs))
    if normalized in {"gated", "gated-sae", "gated_sae"}:
        return GatedSAE(GatedSAEConfig(input_dim=input_dim, n_latents=n_latents, **kwargs))
    raise ValueError("kind must be one of: vg, l1, topk, batchtopk, jumprelu, gated.")
