"""Training loop and config entrypoint for Variational Garrote."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import SyntheticConfig, make_synthetic_train_test
from .evaluate import EvaluationResult, evaluate_model
from .loss import VGLossTerms, vg_loss_terms
from .model import VGConfig, VariationalGarrote
from .utils import dtype_from_string, load_yaml, set_seed


@dataclass
class TrainHistory:
    steps: list[int]
    free_energy: list[float]
    energy: list[float]
    entropy: list[float]
    sparsity_penalty: list[float]
    rho_model: list[float]
    lr: list[float]

    @classmethod
    def empty(cls) -> "TrainHistory":
        return cls([], [], [], [], [], [], [])

    def append(self, step: int, terms: VGLossTerms, lr: float) -> None:
        self.steps.append(step)
        self.free_energy.append(float(terms.free_energy.detach().cpu()))
        self.energy.append(float(terms.energy.detach().cpu()))
        self.entropy.append(float(terms.entropy.detach().cpu()))
        self.sparsity_penalty.append(float(terms.sparsity_penalty.detach().cpu()))
        self.rho_model.append(float(terms.rho_model.detach().cpu()))
        self.lr.append(float(lr))


@dataclass
class TrainResult:
    model: VariationalGarrote
    history: TrainHistory
    evaluation: EvaluationResult | None = None

    def __iter__(self):
        yield self.model
        yield self.history


def build_model(config: dict[str, Any], dtype: torch.dtype, device: torch.device) -> VariationalGarrote:
    model_config = VGConfig(
        n_features=int(config["model"]["n_features"]),
        gamma=float(config["model"]["gamma"]),
        mask_init=float(config["model"]["mask_init"]),
        weight_init_std=float(config["model"].get("weight_init_std", 1.0)),
        loss_eps=float(config["model"].get("loss_eps", config["model"].get("mask_eps", 1.0e-12))),
        dtype=dtype,
    )
    return VariationalGarrote(model_config).to(device=device)


def configure_optimizer(
    model: VariationalGarrote,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def configure_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float = 0.5,
    patience: int = 100,
    threshold: float = 1.0e-7,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        threshold=threshold,
    )


def fit_vg(
    model: VariationalGarrote,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.03,
    lr_stop: float = 1.0e-6,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
    weight_decay: float = 0.0,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 100,
    scheduler_threshold: float = 1.0e-7,
    max_steps: int = 5000,
    gradient_clip_norm: float | None = None,
    history_every: int = 25,
    verbose: bool = False,
) -> TrainResult:
    optimizer = configure_optimizer(model, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    scheduler = configure_scheduler(
        optimizer,
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
    )
    history = TrainHistory.empty()
    model.train()

    last_terms: VGLossTerms | None = None
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        terms = vg_loss_terms(model, x, y)
        terms.free_energy.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
        optimizer.step()

        loss_value = float(terms.free_energy.detach().cpu())
        scheduler.step(loss_value)
        current_lr = float(optimizer.param_groups[0]["lr"])
        last_terms = terms

        should_record = step % history_every == 0 or current_lr < lr_stop or step == max_steps - 1
        if should_record:
            history.append(step, terms, current_lr)
            if verbose:
                print(
                    f"step={step:6d} loss={loss_value:.6g} lr={current_lr:.2e} "
                    f"rho={float(terms.rho_model.detach().cpu()):.4f}"
                )
        if current_lr < lr_stop:
            break

    if not history.steps and last_terms is not None:
        history.append(max_steps - 1, last_terms, float(optimizer.param_groups[0]["lr"]))
    model.eval()
    return TrainResult(model=model, history=history)


def train_vg(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    config: VGConfig,
    **kwargs: Any,
) -> TrainResult:
    model = VariationalGarrote(config).to(device=x_train.device)
    return fit_vg(model, x_train, y_train, **kwargs)


def sweep_gamma(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    gamma_values: np.ndarray,
    base_config: VGConfig,
    **train_kwargs: Any,
) -> list[tuple[float, TrainResult]]:
    results: list[tuple[float, TrainResult]] = []
    for gamma in gamma_values:
        cfg = replace(base_config, gamma=float(gamma))
        results.append((float(gamma), train_vg(x_train, y_train, cfg, **train_kwargs)))
    return results


def _training_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    return {
        "lr": float(training["lr"]),
        "lr_stop": float(training.get("lr_stop", training.get("min_lr", 1.0e-6))),
        "betas": tuple(float(v) for v in training.get("betas", [0.9, 0.999])),
        "eps": float(training.get("eps", 1.0e-8)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "scheduler_factor": float(training.get("scheduler_factor", training.get("lr_factor", 0.5))),
        "scheduler_patience": int(training.get("scheduler_patience", training.get("lr_patience", 100))),
        "scheduler_threshold": float(training.get("scheduler_threshold", 1.0e-7)),
        "max_steps": int(training.get("max_steps", 5000)),
        "gradient_clip_norm": training.get("gradient_clip_norm", training.get("grad_clip")),
        "history_every": int(training.get("history_every", 25)),
    }


def run_from_config(config_path: str | Path = "configs/base.yaml", verbose: bool = False) -> TrainResult:
    config = load_yaml(config_path)
    training_config = config["training"]
    data_config = config["data"]
    dtype = dtype_from_string(config["model"].get("dtype", "float32"))
    device = torch.device(training_config.get("device", "cpu"))
    set_seed(int(training_config.get("seed", 0)))

    if data_config.get("kind", "synthetic") != "synthetic":
        raise ValueError("run_from_config currently supports data.kind='synthetic'.")

    synthetic_config = SyntheticConfig(
        n_features=int(data_config["n_features"]),
        n_samples=int(data_config["n_train"]),
        rho_data=float(data_config["rho_data"]),
        snr=float(data_config["snr"]),
        seed=int(training_config.get("seed", 0)),
        dtype=dtype,
    )
    train_data, test_data = make_synthetic_train_test(
        synthetic_config,
        n_test=int(data_config["n_test"]),
        test_seed_offset=int(data_config.get("test_seed_offset", 10000)),
        device=device,
    )
    model = build_model(config, dtype=dtype, device=device)
    result = fit_vg(model, train_data.x, train_data.y, verbose=verbose, **_training_kwargs(config))
    result.evaluation = evaluate_model(result.model, test_data.x, test_data.y, true_selection=test_data.teacher_mask)

    if bool(training_config.get("save_checkpoint", False)):
        checkpoint_path = Path(training_config.get("checkpoint_path", "outputs/vg_model.pt"))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": result.model.state_dict(),
                "config": config,
                "history": result.history.__dict__,
                "evaluation": result.evaluation.__dict__ if result.evaluation else None,
            },
            checkpoint_path,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Variational Garrote model from YAML config.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = run_from_config(args.config, verbose=args.verbose)
    if result.evaluation is not None:
        print(result.evaluation)


if __name__ == "__main__":
    main()
