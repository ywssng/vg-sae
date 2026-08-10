"""Reusable configuration, model, data, and artifact seams for SAE sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
from .sae_model import (
    BatchTopKSAE,
    BatchTopKSAEConfig,
    GatedSAE,
    GatedSAEConfig,
    JumpReLUSAE,
    JumpReLUSAEConfig,
    StandardSAE,
    StandardSAEConfig,
    TopKSAE,
    TopKSAEConfig,
    VGSAEConfig,
    VariationalGarroteSAE,
)


METHOD_ORDER = ("vgsae", "l1", "topk", "batchtopk", "jumprelu", "gated")
METHOD_LABELS = {
    "vgsae": "VG-SAE",
    "l1": "L1-ReLU",
    "topk": "TopK",
    "batchtopk": "BatchTopK",
    "jumprelu": "JumpReLU",
    "gated": "Gated",
}

FULL_CONTROLS: dict[str, list[float | int]] = {
    "vgsae": [
        12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5,
        2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.35, 0.25, 0.125, 0.0,
        -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1,
    ],
    "l1": [5.0, 4.5, 3.5, 3.0, 2.0, 1.0, 0.5, 0.3, 0.25, 0.2, 0.1,
           0.05, 0.02, 0.002, 0.0003, 0.0],
    "topk": list(range(1, 33)),
    "batchtopk": [
        0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.5, 0.75, 1.0, 1.5,
        2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0,
        20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 31.0, 32.0,
    ],
    "jumprelu": [
        30.0, 10.0, 5.0, 4.3, 4.25, 4.2, 4.1, 4.0, 3.5, 3.0, 2.0,
        1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03,
        0.02, 0.015, 0.01, 0.007, 0.006, 0.005, 0.003, 0.001, 0.0003, 0.0,
    ],
    "gated": [
        10.0, 9.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2,
        0.15, 0.1, 0.07, 0.05, 0.03, 0.025, 0.02, 0.015, 0.0125,
        0.01, 0.005, 0.0,
    ],
}

FAST_CONTROLS: dict[str, list[float | int]] = {
    "vgsae": [0.0, 1.0],
    "l1": [1.0e-4, 1.0e-3],
    "topk": [1, 2],
    "batchtopk": [1.0, 2.0],
    "jumprelu": [0.1, 1.0],
    "gated": [1.0e-4, 1.0e-3],
}

CONTROL_NAMES = {
    "vgsae": "gamma",
    "l1": "l1_coefficient",
    "topk": "k",
    "batchtopk": "k",
    "jumprelu": "l0_coefficient",
    "gated": "l1_coefficient",
}


@dataclass
class SyntheticDataConfig:
    """Serializable data boundary; ``kind`` is the future adapter switch."""

    kind: str = "synthetic_sparse_coding"
    input_dim: int = 16
    n_features: int = 32
    n_train: int = 512
    n_test: int = 512
    support_density: float = 0.1
    coherence: float = 0.3
    noise_std: float = 0.0
    frequency_skew: float = 0.01
    amplitude_scale: float = 1.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SyntheticDataConfig:
        return cls(**values)


@dataclass
class TrainingConfig:
    lr: float = 1.0e-2
    batch_size: int = 128
    train_steps: int = 1000
    history_every: int = 25
    dead_feature_window: int = 100
    gradient_clip_norm: float | None = 1.0
    beta: float = 1.0
    dead_threshold: float = 1.0e-6
    mask_threshold: float = 0.5

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainingConfig:
        return cls(**values)


@dataclass
class SweepConfig:
    experiment_name: str = "exp07_parallel"
    data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seeds: list[int] = field(default_factory=lambda: [0])
    methods: list[str] = field(default_factory=lambda: list(METHOD_ORDER))
    controls: dict[str, list[float | int]] = field(
        default_factory=lambda: {name: list(values) for name, values in FULL_CONTROLS.items()}
    )
    wandb_project: str = "vg-sae"

    def validate(self) -> None:
        if not self.methods:
            raise ValueError("At least one SAE method is required.")
        unknown = set(self.methods) - set(METHOD_ORDER)
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        if not self.seeds:
            raise ValueError("At least one experiment seed is required.")
        if self.data.kind != "synthetic_sparse_coding":
            raise ValueError(
                f"Unsupported data kind {self.data.kind!r}; add its adapter in make_train_test()."
            )
        if self.training.train_steps <= 0 or self.training.history_every <= 0:
            raise ValueError("train_steps and history_every must be positive.")
        for method in self.methods:
            if not self.controls.get(method):
                raise ValueError(f"No controls configured for method {method!r}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SweepConfig:
        payload = dict(values)
        payload["data"] = SyntheticDataConfig.from_dict(payload["data"])
        payload["training"] = TrainingConfig.from_dict(payload["training"])
        config = cls(**payload)
        config.validate()
        return config


@dataclass(frozen=True)
class RunSpec:
    method: str
    control_name: str
    control_value: float | int
    seed: int
    init_seed: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.method}_{self.control_name}={self.control_value}_seed={self.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RunSpec:
        return cls(**values)


def default_sweep_config(fast: bool = False) -> SweepConfig:
    controls = FAST_CONTROLS if fast else FULL_CONTROLS
    return SweepConfig(
        experiment_name="exp07_parallel_fast" if fast else "exp07_parallel",
        data=SyntheticDataConfig(n_train=96, n_test=96) if fast else SyntheticDataConfig(),
        training=(
            TrainingConfig(train_steps=4, history_every=1, dead_feature_window=1)
            if fast
            else TrainingConfig()
        ),
        controls={name: list(values) for name, values in controls.items()},
    )


def build_specs(config: SweepConfig) -> list[RunSpec]:
    config.validate()
    specs: list[RunSpec] = []
    for seed in config.seeds:
        for method in METHOD_ORDER:
            if method not in config.methods:
                continue
            init_seed = 100_000 + 1_000 * seed + METHOD_ORDER.index(method)
            specs.extend(
                RunSpec(method, CONTROL_NAMES[method], value, seed, init_seed)
                for value in config.controls[method]
            )
    return specs


def make_train_test(
    config: SweepConfig, seed: int, device: torch.device | str = "cpu"
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Materialize the configured data source behind one stable runner seam."""

    config.validate()
    data_cfg = config.data
    data = make_synthetic_sparse_coding(
        SyntheticSparseCodingConfig(
            input_dim=data_cfg.input_dim,
            n_features=data_cfg.n_features,
            n_samples=data_cfg.n_train + data_cfg.n_test,
            support_density=data_cfg.support_density,
            coherence=data_cfg.coherence,
            noise_std=data_cfg.noise_std,
            frequency_skew=data_cfg.frequency_skew,
            amplitude_scale=data_cfg.amplitude_scale,
            seed=seed,
        ),
        device=device,
    )
    shared = {
        "dictionary": data.dictionary,
        "feature_probabilities": data.feature_probabilities,
    }
    train = SimpleNamespace(
        x=data.x[: data_cfg.n_train],
        z=data.z[: data_cfg.n_train],
        support=data.support[: data_cfg.n_train],
        clean_x=data.clean_x[: data_cfg.n_train],
        **shared,
    )
    test = SimpleNamespace(
        x=data.x[data_cfg.n_train :],
        z=data.z[data_cfg.n_train :],
        support=data.support[data_cfg.n_train :],
        clean_x=data.clean_x[data_cfg.n_train :],
        **shared,
    )
    return train, test


def build_model(config: SweepConfig, spec: RunSpec) -> torch.nn.Module:
    data, training, value = config.data, config.training, spec.control_value
    dimensions = {"d_in": data.input_dim, "d_sae": data.n_features}
    if spec.method == "vgsae":
        return VariationalGarroteSAE(
            VGSAEConfig(
                input_dim=data.input_dim,
                n_latents=data.n_features,
                lambda_sparsity=float(value),
                beta=training.beta,
                beta_mode="profiled",
            )
        )
    if spec.method == "l1":
        return StandardSAE(StandardSAEConfig(**dimensions, l1_coefficient=float(value)))
    if spec.method == "topk":
        return TopKSAE(TopKSAEConfig(**dimensions, k=int(value)))
    if spec.method == "batchtopk":
        return BatchTopKSAE(BatchTopKSAEConfig(**dimensions, k=float(value)))
    if spec.method == "jumprelu":
        return JumpReLUSAE(JumpReLUSAEConfig(**dimensions, l0_coefficient=float(value)))
    if spec.method == "gated":
        return GatedSAE(GatedSAEConfig(**dimensions, l1_coefficient=float(value)))
    raise ValueError(f"Unknown method: {spec.method}")


def run_directory(sweep_dir: Path | str, spec: RunSpec) -> Path:
    return Path(sweep_dir) / "runs" / spec.method / spec.run_id


def _state_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for name, value in state_dict.items()
    }


def save_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    config: SweepConfig,
    spec: RunSpec,
    checkpoint_kind: str,
    state_dict: dict[str, Any] | None = None,
    step: int | None = None,
    loss: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "checkpoint_kind": checkpoint_kind,
        "step": step,
        "loss": loss,
        "sweep_config": config.to_dict(),
        "run_spec": spec.to_dict(),
        "metadata": metadata or {},
        "model_state": _state_to_cpu(state_dict or model.state_dict()),
    }
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    path: Path | str, device: torch.device | str = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    config = SweepConfig.from_dict(payload["sweep_config"])
    spec = RunSpec.from_dict(payload["run_spec"])
    model = build_model(config, spec).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
