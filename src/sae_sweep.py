"""Reusable configuration, model, data, and artifact seams for SAE sweeps."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .sae_data import (
    AMPLITUDE_MODES,
    AmplitudeMode,
    SyntheticSparseCodingConfig,
    make_synthetic_sparse_coding,
)
from .sae_model import (
    BatchTopKSAE,
    BatchTopKSAEConfig,
    BetaMode,
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
    "topk": list(range(1, 129)),
    "batchtopk": [
        0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.5, 0.75, 1.0, 1.5,
        2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0,
        20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 31.0, 32.0, 36.0,
        40.0, 48.0, 56.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0,
        112.0, 120.0, 124.0, 128.0,
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

STAGE1_DATA_KIND = "stage1_custom_baseline"
LEGACY_DATA_KIND = "synthetic_sparse_coding"


@dataclass(init=False)
class SyntheticDataConfig:
    """Serializable boundary for the simple custom sparse-coding baseline."""

    kind: str
    input_dim: int
    ground_truth_num_features: int
    sae_width: int
    n_train: int
    n_test: int
    support_density: float
    coherence: float
    noise_std: float
    frequency_skew: float
    amplitude_scale: float
    amplitude_mode: AmplitudeMode

    def __init__(
        self,
        kind: str = STAGE1_DATA_KIND,
        input_dim: int = 128,
        ground_truth_num_features: int | None = None,
        sae_width: int | None = None,
        n_train: int = 8196,
        n_test: int = 1024,
        support_density: float = 0.01,
        coherence: float = 0.0,
        noise_std: float = 0.0,
        frequency_skew: float = 0.5,
        amplitude_scale: float = 1.0,
        *,
        amplitude_mode: AmplitudeMode = "exponential",
        n_features: int | None = None,
    ) -> None:
        if ground_truth_num_features is None:
            ground_truth_num_features = 1024 if n_features is None else n_features
        elif n_features is not None and ground_truth_num_features != n_features:
            raise ValueError(
                "ground_truth_num_features and legacy n_features disagree."
            )
        if sae_width is None:
            sae_width = ground_truth_num_features if n_features is None else n_features
        elif n_features is not None and sae_width != n_features:
            raise ValueError("sae_width and legacy n_features disagree.")
        self.kind = kind
        self.input_dim = input_dim
        self.ground_truth_num_features = ground_truth_num_features
        self.sae_width = sae_width
        self.n_train = n_train
        self.n_test = n_test
        self.support_density = support_density
        self.coherence = coherence
        self.noise_std = noise_std
        self.frequency_skew = frequency_skew
        self.amplitude_scale = amplitude_scale
        self.amplitude_mode = amplitude_mode

    @property
    def n_features(self) -> int:
        """Deprecated alias for old configurations where both widths matched."""

        return self.ground_truth_num_features

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SyntheticDataConfig:
        payload = dict(values)
        if "n_features" in payload and "kind" not in payload:
            payload["kind"] = LEGACY_DATA_KIND
        if "n_features" in payload and {
            "ground_truth_num_features",
            "sae_width",
        } & payload.keys():
            raise ValueError("Do not mix legacy n_features with canonical width fields.")
        return cls(**payload)


@dataclass
class TrainingConfig:
    lr: float = 1.0e-2
    batch_size: int = 128
    train_steps: int = 1000
    history_every: int = 25
    dead_feature_window: int = 100
    gradient_clip_norm: float | None = 1.0
    beta: float = 1.0
    beta_mode: BetaMode = "profiled"
    dead_threshold: float = 1.0e-6
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.beta_mode not in {"profiled", "learned"}:
            raise ValueError("Stage-1 VG beta_mode must be profiled or learned.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainingConfig:
        return cls(**values)


@dataclass
class SweepConfig:
    experiment_name: str = "stage1_custom_baseline"
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
        if any(not isinstance(seed, Integral) or seed < 0 for seed in self.seeds):
            raise ValueError("Experiment seeds must be nonnegative integers.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Experiment seeds must be unique.")
        if self.data.kind not in {STAGE1_DATA_KIND, LEGACY_DATA_KIND}:
            raise ValueError(
                f"Unsupported data kind {self.data.kind!r}; add its adapter in make_train_test()."
            )
        if self.data.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.data.ground_truth_num_features <= 0:
            raise ValueError("ground_truth_num_features must be positive.")
        if self.data.sae_width <= 0:
            raise ValueError("sae_width must be positive.")
        if self.data.n_train <= 0 or self.data.n_test <= 0:
            raise ValueError("n_train and n_test must be positive.")
        if not 0.0 < self.data.support_density < 1.0:
            raise ValueError("support_density must be in (0, 1).")
        if self.data.frequency_skew < 0.0:
            raise ValueError("frequency_skew must be nonnegative.")
        if self.data.amplitude_scale <= 0.0:
            raise ValueError("amplitude_scale must be positive.")
        if self.data.amplitude_mode not in AMPLITUDE_MODES:
            raise ValueError(
                "amplitude_mode must be exponential, constant, or uniform."
            )
        if not 0.0 <= self.data.coherence < 1.0:
            raise ValueError("coherence must satisfy 0 <= coherence < 1.")
        if self.data.noise_std < 0.0:
            raise ValueError("noise_std must be nonnegative.")
        if self.data.kind == STAGE1_DATA_KIND:
            if self.data.ground_truth_num_features <= self.data.input_dim:
                raise ValueError(
                    "The simple baseline requires ground_truth_num_features > input_dim."
                )
            n_features = self.data.ground_truth_num_features
            mean_weight = sum(
                rank ** (-self.data.frequency_skew)
                for rank in range(1, n_features + 1)
            ) / n_features
            if self.data.support_density / mean_weight > 0.95:
                raise ValueError(
                    "support_density is too high to preserve its requested mean "
                    "without clipping feature probabilities."
                )
            if self.data.coherence != 0.0:
                raise ValueError("The simple baseline does not add dictionary coherence.")
            if self.data.noise_std != 0.0:
                raise ValueError("The simple baseline does not add observation noise.")
        if self.training.train_steps <= 0 or self.training.history_every <= 0:
            raise ValueError("train_steps and history_every must be positive.")
        if self.training.beta_mode not in {"profiled", "learned"}:
            raise ValueError(
                "Stage-1 VG beta_mode must be profiled or learned."
            )
        for method in self.methods:
            values = self.controls.get(method)
            if not values:
                raise ValueError(f"No controls configured for method {method!r}.")
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"Controls for {method!r} must be finite.")
            if len({float(value) for value in values}) != len(values):
                raise ValueError(f"Controls for {method!r} must be unique.")
            if method == "topk" and any(
                float(value) != int(value)
                or not 1 <= int(value) <= self.data.sae_width
                for value in values
            ):
                raise ValueError("TopK controls must be integers in [1, sae_width].")
            if method == "batchtopk" and any(
                not 0.0 < float(value) <= self.data.sae_width for value in values
            ):
                raise ValueError("BatchTopK controls must be in (0, sae_width].")
            if method in {"l1", "jumprelu", "gated"} and any(
                float(value) < 0.0 for value in values
            ):
                raise ValueError(f"Controls for {method!r} must be nonnegative.")

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


def _density_token(value: float) -> str:
    whole, _, fraction = f"{value:.12f}".rstrip("0").rstrip(".").partition(".")
    return f"{whole}{fraction.ljust(2, '0')}"


def sweep_experiment_id(config: SweepConfig) -> str:
    """Build the readable directory and W&B group ID for one data condition."""

    prefixes = {
        "stage1_custom_baseline": "stage1",
        "stage1_custom_baseline_fast": "stage1_fast",
    }
    prefix = prefixes.get(config.experiment_name, config.experiment_name)
    prefix = "_".join(
        "".join(
            character if character.isalnum() else " " for character in prefix
        ).split()
    )
    seeds = config.seeds
    seed_token = (
        f"seed{seeds[0]}"
        if len(seeds) == 1
        else "seeds" + "-".join(str(seed) for seed in seeds)
    )
    data = config.data
    ablation_tokens = []
    if data.amplitude_mode != "exponential":
        ablation_tokens.append(f"amp{data.amplitude_mode}")
    if data.frequency_skew != 0.5:
        frequency_token = f"{data.frequency_skew:.12g}".replace("-", "m").replace(
            ".", "p"
        )
        ablation_tokens.append(f"fs{frequency_token}")
    ablation_suffix = "".join(f"_{token}" for token in ablation_tokens)
    return (
        f"{prefix}_beta_{config.training.beta_mode}"
        f"_din{data.input_dim}_gt{data.ground_truth_num_features}"
        f"_sae{data.sae_width}_sd{_density_token(data.support_density)}"
        f"{ablation_suffix}"
        f"_{seed_token}"
    )


def default_sweep_dir(project_root: Path | str, config: SweepConfig) -> Path:
    return Path(project_root) / "outputs" / "runs" / sweep_experiment_id(config)


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
        experiment_name=(
            "stage1_custom_baseline_fast" if fast else "stage1_custom_baseline"
        ),
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
            ground_truth_num_features=data_cfg.ground_truth_num_features,
            n_samples=data_cfg.n_train + data_cfg.n_test,
            support_density=data_cfg.support_density,
            coherence=data_cfg.coherence,
            noise_std=data_cfg.noise_std,
            frequency_skew=data_cfg.frequency_skew,
            amplitude_scale=data_cfg.amplitude_scale,
            amplitude_mode=data_cfg.amplitude_mode,
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
    dimensions = {"d_in": data.input_dim, "d_sae": data.sae_width}
    if spec.method == "vgsae":
        return VariationalGarroteSAE(
            VGSAEConfig(
                input_dim=data.input_dim,
                n_latents=data.sae_width,
                lambda_sparsity=float(value),
                beta=training.beta,
                beta_mode=training.beta_mode,
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
        "format_version": 2,
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
