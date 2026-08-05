"""Training utilities for VG-SAE and sparse-autoencoder baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE
from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.sae_trainer import SAETrainer

from .sae_baselines import to_inference_sae
from .sae_loss import sae_loss_terms, saelens_sae_loss_terms
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


class _CyclingTensorBatches:
    """Deterministic fixed-size provider matching ``SAETrainer``'s contract."""

    def __init__(self, x: torch.Tensor, batch_size: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.loader = DataLoader(
            TensorDataset(x),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
        )
        self.iterator = iter(self.loader)

    def __iter__(self) -> _CyclingTensorBatches:
        return self

    def __next__(self) -> torch.Tensor:
        try:
            (batch,) = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            (batch,) = next(self.iterator)
        return batch


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
    for name, value in getattr(terms, "details", {}).items():
        row[name] = (
            float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        )
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
    if x.shape[0] == 0:
        raise ValueError("x must contain at least one sample.")
    if batch_size <= 0 or max_steps <= 0 or history_every <= 0:
        raise ValueError("batch_size, max_steps, and history_every must be positive.")
    if dead_feature_window <= 0:
        raise ValueError("dead_feature_window must be positive.")
    set_seed(seed)

    if isinstance(model, TrainingSAE):
        return _fit_saelens_sae(
            model,
            x,
            lr=lr,
            batch_size=batch_size,
            max_steps=max_steps,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            history_every=history_every,
            dead_feature_window=dead_feature_window,
            seed=seed,
            verbose=verbose,
        )

    model.to(device=x.device, dtype=x.dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(x),
        batch_size=min(batch_size, x.shape[0]),
        shuffle=True,
        drop_last=False,
    )
    iterator = iter(loader)
    history: list[dict[str, float]] = []
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
        terms = sae_loss_terms(model, batch, steps_since_fired > dead_feature_window)
        feature_acts = getattr(terms, "feature_acts", None)
        if feature_acts is not None:
            with torch.no_grad():
                steps_since_fired.add_(1)
                steps_since_fired[feature_acts.gt(0).any(dim=0)] = 0
        terms.loss.backward()
        should_normalize_decoder = hasattr(model, "normalize_decoder_columns") and (
            not isinstance(model, VariationalGarroteSAE) or model.config.normalize_decoder
        )
        if should_normalize_decoder and hasattr(model, "remove_decoder_parallel_grad"):
            model.remove_decoder_parallel_grad()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
        optimizer.step()
        if should_normalize_decoder:
            model.normalize_decoder_columns()

        if step % history_every == 0 or step == max_steps - 1:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                full_terms = sae_loss_terms(model, x, steps_since_fired > dead_feature_window)
            if was_training:
                model.train()
            row = _history_row(step, full_terms, optimizer.param_groups[0]["lr"])
            history.append(row)
            if verbose:
                print(
                    f"step={step:5d} loss={row['loss']:.6g} "
                    f"mse={row.get('reconstruction_mse', np.nan):.6g} "
                    f"rho={row.get('rho', np.nan):.4f}"
                )

    model.eval()
    return SAETrainResult(model=model, history=history)


def _fit_saelens_sae(
    model: TrainingSAE[Any],
    x: torch.Tensor,
    *,
    lr: float,
    batch_size: int,
    max_steps: int,
    weight_decay: float,
    gradient_clip_norm: float | None,
    history_every: int,
    dead_feature_window: int,
    seed: int,
    verbose: bool,
) -> SAETrainResult:
    """Train through the official optimizer, schedulers, and ``step`` method."""

    if weight_decay != 0:
        raise ValueError("Official SAELens SAETrainer uses Adam without weight decay.")
    if gradient_clip_norm != 1.0:
        raise ValueError("Official SAELens SAETrainer fixes gradient clipping at 1.0.")

    model.to(device=x.device)
    effective_batch_size = min(batch_size, x.shape[0])
    provider = _CyclingTensorBatches(x, effective_batch_size, seed)
    trainer_type: type[SAETrainer] = SAETrainer
    if model.cfg.architecture() == "vg":
        from .saelens_vg import VGSAETrainer

        trainer_type = VGSAETrainer
    trainer = trainer_type(
        cfg=SAETrainerConfig(
            total_training_samples=max_steps * effective_batch_size,
            train_batch_size_samples=effective_batch_size,
            lr=lr,
            lr_end=lr,
            lr_scheduler_name="constant",
            device=str(x.device),
            dead_feature_window=dead_feature_window,
            logger=LoggingConfig(log_to_wandb=False),
        ),
        sae=model,
        data_provider=provider,
    )
    if model.cfg.normalize_activations == "expected_average_only_in":
        trainer.activation_scaler.estimate_scaling_factor(
            d_in=model.cfg.d_in,
            data_provider=provider,
            n_batches_for_norm_estimate=trainer.cfg.n_batches_for_norm_estimate,
        )

    history: list[dict[str, float]] = []
    for step in range(max_steps):
        trainer.maybe_reset_sparsity()
        trainer.step(next(provider))

        if step % history_every == 0 or step == max_steps - 1:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                terms = saelens_sae_loss_terms(
                    model,
                    trainer.activation_scaler(x),
                    trainer.dead_neurons,
                    coefficients=trainer.get_coefficients(),
                    n_training_steps=trainer.n_training_steps,
                )
            if was_training:
                model.train()
            row = _history_row(step, terms, trainer.optimizer.param_groups[0]["lr"])
            if isinstance(model, BatchTopKTrainingSAE):
                row["threshold"] = float(model.topk_threshold.detach().cpu())
            history.append(row)
            if verbose:
                print(
                    f"step={step:5d} loss={row['loss']:.6g} "
                    f"mse={row.get('reconstruction_mse', np.nan):.6g} "
                    f"rho={row.get('rho', np.nan):.4f}"
                )
        trainer.n_training_steps += 1

    if trainer.activation_scaler.scaling_factor is not None:
        model.fold_activation_norm_scaling_factor(trainer.activation_scaler.scaling_factor)
        trainer.activation_scaler.scaling_factor = None
    trainer.set_final_sae_metadata()
    model.eval()
    return SAETrainResult(model=model, history=history)


def build_sae(kind: str, input_dim: int, n_latents: int, **kwargs: Any) -> torch.nn.Module:
    """Build an SAE; only this factory translates legacy dimension names."""

    normalized = kind.lower()
    if normalized in {"vg", "vg-sae", "vg_sae"}:
        return VariationalGarroteSAE(
            VGSAEConfig(input_dim=input_dim, n_latents=n_latents, **kwargs)
        )
    dimensions = {"d_in": input_dim, "d_sae": n_latents}
    if normalized in {"l1", "l1-relu", "l1_relu"}:
        return L1ReLUSAE(L1SAEConfig(**dimensions, **kwargs))
    if normalized in {"topk", "top-k", "top_k"}:
        return TopKSAE(TopKSAEConfig(**dimensions, **kwargs))
    if normalized in {"batchtopk", "batch-topk", "batch_topk"}:
        return BatchTopKSAE(BatchTopKSAEConfig(**dimensions, **kwargs))
    if normalized in {"jumprelu", "jump-relu", "jump_relu"}:
        return JumpReLUSAE(JumpReLUSAEConfig(**dimensions, **kwargs))
    if normalized in {"gated", "gated-sae", "gated_sae"}:
        return GatedSAE(GatedSAEConfig(**dimensions, **kwargs))
    raise ValueError("kind must be one of: vg, l1, topk, batchtopk, jumprelu, gated.")
