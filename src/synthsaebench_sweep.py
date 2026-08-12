"""Fixed-generator configuration and model seams for Stage-2 SynthSAEBench sweeps."""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from huggingface_hub import snapshot_download
from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    GatedTrainingSAE,
    GatedTrainingSAEConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
)
from sae_lens.registry import get_sae_training_class
from sae_lens.saes.sae import TrainingSAE
from sae_lens.synthetic import SyntheticModel
from sae_lens.synthetic import synthetic_model as synthetic_model_module

from .sae_sweep import CONTROL_NAMES, METHOD_LABELS, METHOD_ORDER
from .saelens_vg import VGTrainingSAE, VGTrainingSAEConfig


SYNTHSAEBENCH_DATA_KIND = "synthsaebench_pretrained"
BENCHMARK_MODEL_ID = "decoderesearch/synth-sae-bench-16k-v1"
BENCHMARK_REVISION = "b2efd8b919ae46d6d487c73d46db5ee52813621d"
BENCHMARK_CONFIG_SHA256 = (
    "ec969226283f05b69fd3b2a8c1cd14b152a998d79a491d732ccd286d096908b5"
)
SAELENS_REVISION = "8be14080485952f729ed58d674bcddf9778e0aa4"
BENCHMARK_INPUT_DIM = 768
BENCHMARK_NUM_FEATURES = 16_384
BENCHMARK_SAE_WIDTH = 4_096
BENCHMARK_SCALE_CHILDREN_BY_PARENT = False
DEFAULT_MAX_PER_DEVICE = 2

# Broad range-scout controls.  Use the 200M-calibrated FINAL_CONTROLS for the
# definitive full-budget comparison.
CALIBRATION_CONTROLS: dict[str, list[float | int]] = {
    "vgsae": [0.4, 0.5, 0.55, 0.6, 0.65, 0.75],
    "l1": [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0],
    "topk": [15, 20, 25, 30, 35, 40, 45],
    "batchtopk": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
    "jumprelu": [0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    "gated": [1.0, 1.35, 1.5, 2.0, 3.0, 4.0, 5.0],
}

# One-seed 200M calibration runs invert each method's measured calibration-stream
# hard-L0 curve toward the benchmark comparison targets [45, 40, ..., 15].  The
# final x-axis remains achieved hard L0, never coefficient order.
FINAL_CONTROLS: dict[str, list[float | int]] = {
    "vgsae": [0.77, 0.82, 0.88, 0.96, 1.07, 1.17, 1.39],
    "l1": [0.99, 1.07, 1.17, 1.36, 1.69, 2.42, 4.26],
    "topk": [15, 20, 25, 30, 35, 40, 45],
    "batchtopk": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
    "jumprelu": [0.41, 0.46, 0.52, 0.61, 0.78, 1.16, 1.80],
    "gated": [1.07, 1.10, 1.21, 1.38, 1.70, 2.17, 3.28],
}

FAST_CONTROLS: dict[str, list[float | int]] = {
    "vgsae": [0.0, 1.0],
    "l1": [1.0, 2.0],
    "topk": [20, 35],
    "batchtopk": [20.0, 35.0],
    "jumprelu": [0.3, 1.0],
    "gated": [1.0, 2.0],
}


@contextmanager
def temporary_seed_for_device(
    seed: int,
    device: torch.device | str,
    *,
    cpu_rng_state: torch.Tensor | None = None,
    device_rng_state: torch.Tensor | None = None,
):
    """Fork only the worker's CPU/current-device RNG state.

    SAELens' generic ``temporary_seed`` forks every visible CUDA device.  Sweep
    workers are assigned one explicit device, so touching all four devices adds
    contexts and can make concurrent jobs interfere with each other's RNG state.
    Optional states resume an interrupted stream exactly after entering the fork.
    """

    normalized = torch.device(device)
    cuda_devices: list[int] = []
    device_index: int | None = None
    if normalized.type == "cuda":
        device_index = (
            torch.cuda.current_device()
            if normalized.index is None
            else normalized.index
        )
        cuda_devices = [device_index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if device_index is not None:
            with torch.cuda.device(device_index):
                torch.cuda.manual_seed(seed)
        if cpu_rng_state is not None:
            torch.set_rng_state(cpu_rng_state.cpu())
        if device_rng_state is not None:
            if device_index is None:
                raise ValueError("device_rng_state requires a CUDA device.")
            torch.cuda.set_rng_state(device_rng_state.cpu(), device_index)
        yield


def capture_rng_state(device: torch.device | str) -> dict[str, torch.Tensor | None]:
    """Capture CPU and assigned-device RNG state for a rolling resume file."""

    normalized = torch.device(device)
    device_state = None
    if normalized.type == "cuda":
        device_index = (
            torch.cuda.current_device()
            if normalized.index is None
            else normalized.index
        )
        device_state = torch.cuda.get_rng_state(device_index).cpu()
    return {
        "cpu_rng_state": torch.get_rng_state().cpu(),
        "device_rng_state": device_state,
    }


@dataclass(frozen=True)
class SynthSAEBenchDataConfig:
    """The immutable official pretrained generator plus streamed sample counts."""

    kind: str = SYNTHSAEBENCH_DATA_KIND
    model_id: str = BENCHMARK_MODEL_ID
    revision: str = BENCHMARK_REVISION
    model_config_sha256: str = BENCHMARK_CONFIG_SHA256
    input_dim: int = BENCHMARK_INPUT_DIM
    ground_truth_num_features: int = BENCHMARK_NUM_FEATURES
    sae_width: int = BENCHMARK_SAE_WIDTH
    scale_children_by_parent: bool = BENCHMARK_SCALE_CHILDREN_BY_PARENT
    # Batch-aligned values: 195,312 train batches and exactly one eighth as test.
    n_train: int = 199_999_488
    n_test: int = 24_999_936

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SynthSAEBenchDataConfig:
        return cls(**values)


@dataclass(frozen=True)
class SynthSAEBenchTrainingConfig:
    """Paper-aligned streaming training settings; epochs do not apply."""

    lr: float = 3.0e-4
    batch_size: int = 1_024
    history_every: int = 1_000
    dead_feature_window: int = 1_000
    feature_sampling_window: int = 2_000
    n_batches_for_norm_estimate: int = 1_000
    autocast_sae: bool = True
    autocast_data: bool = True
    # The paper describes a final-third decay, but the released experiment configs
    # use a constant 3e-4 learning rate.  Default to the executable reference.
    lr_decay_fraction: float = 0.0
    beta: float = 1.0
    beta_mode: str = "fixed"
    mask_threshold: float = 0.5
    heatmap_samples: int = 80
    resume_every: int = 10_000

    @classmethod
    def from_dict(
        cls, values: dict[str, Any]
    ) -> SynthSAEBenchTrainingConfig:
        return cls(**values)


@dataclass
class SynthSAEBenchSweepConfig:
    experiment_name: str = "stage2_synthsaebench16k_l0calibrated"
    data: SynthSAEBenchDataConfig = field(default_factory=SynthSAEBenchDataConfig)
    training: SynthSAEBenchTrainingConfig = field(
        default_factory=SynthSAEBenchTrainingConfig
    )
    seeds: list[int] = field(default_factory=lambda: [0])
    methods: list[str] = field(default_factory=lambda: list(METHOD_ORDER))
    controls: dict[str, list[float | int]] = field(
        default_factory=lambda: {
            name: list(values) for name, values in FINAL_CONTROLS.items()
        }
    )
    wandb_project: str = "vg-sae"

    def validate(self) -> None:
        data = self.data
        immutable = {
            "kind": (data.kind, SYNTHSAEBENCH_DATA_KIND),
            "model_id": (data.model_id, BENCHMARK_MODEL_ID),
            "revision": (data.revision, BENCHMARK_REVISION),
            "model_config_sha256": (
                data.model_config_sha256,
                BENCHMARK_CONFIG_SHA256,
            ),
            "input_dim": (data.input_dim, BENCHMARK_INPUT_DIM),
            "ground_truth_num_features": (
                data.ground_truth_num_features,
                BENCHMARK_NUM_FEATURES,
            ),
            "sae_width": (data.sae_width, BENCHMARK_SAE_WIDTH),
            "scale_children_by_parent": (
                data.scale_children_by_parent,
                BENCHMARK_SCALE_CHILDREN_BY_PARENT,
            ),
        }
        changed = [name for name, (actual, expected) in immutable.items() if actual != expected]
        if changed:
            raise ValueError(
                "SynthSAEBench fixes the official pretrained generator and SAE width; "
                f"unsupported override(s): {', '.join(changed)}"
            )
        if data.n_train <= 0 or data.n_test <= 0:
            raise ValueError("n_train and n_test must be positive streamed sample counts.")
        training = self.training
        if training.lr <= 0.0 or training.batch_size <= 0:
            raise ValueError("lr and batch_size must be positive.")
        if data.n_train % training.batch_size or data.n_test % training.batch_size:
            raise ValueError(
                "n_train and n_test must be divisible by batch_size so streamed "
                "sample budgets are exact."
            )
        if training.history_every <= 0:
            raise ValueError("history_every must be positive.")
        if training.dead_feature_window <= 0 or training.feature_sampling_window <= 0:
            raise ValueError("dead-feature windows must be positive.")
        if training.n_batches_for_norm_estimate <= 0:
            raise ValueError("n_batches_for_norm_estimate must be positive.")
        if not 0.0 <= training.lr_decay_fraction < 1.0:
            raise ValueError("lr_decay_fraction must lie in [0, 1).")
        if training.beta_mode not in {"fixed", "learned"}:
            raise ValueError("SynthSAEBench VG beta_mode must be fixed or learned.")
        if not 0.0 <= training.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must lie in [0, 1].")
        if training.heatmap_samples <= 0:
            raise ValueError("heatmap_samples must be positive.")
        if training.resume_every <= 0:
            raise ValueError("resume_every must be positive.")
        if not self.methods:
            raise ValueError("At least one SAE method is required.")
        if unknown := set(self.methods) - set(METHOD_ORDER):
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        if not self.seeds:
            raise ValueError("At least one experiment seed is required.")
        if any(not isinstance(seed, Integral) or seed < 0 for seed in self.seeds):
            raise ValueError("Experiment seeds must be nonnegative integers.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Experiment seeds must be unique.")
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
                or not 1 <= int(value) <= data.sae_width
                for value in values
            ):
                raise ValueError("TopK controls must be integers in [1, sae_width].")
            if method == "batchtopk" and any(
                not 0.0 < float(value) <= data.sae_width for value in values
            ):
                raise ValueError("BatchTopK controls must be in (0, sae_width].")
            if method in {"l1", "jumprelu", "gated"} and any(
                float(value) < 0.0 for value in values
            ):
                raise ValueError(f"Controls for {method!r} must be nonnegative.")

    @property
    def total_training_steps(self) -> int:
        return math.ceil(self.data.n_train / self.training.batch_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SynthSAEBenchSweepConfig:
        payload = dict(values)
        payload["data"] = SynthSAEBenchDataConfig.from_dict(payload["data"])
        payload["training"] = SynthSAEBenchTrainingConfig.from_dict(
            payload["training"]
        )
        config = cls(**payload)
        config.validate()
        return config


@dataclass(frozen=True)
class SynthSAEBenchRunSpec:
    method: str
    control_name: str
    control_value: float | int
    seed: int
    init_seed: int
    calibration_seed: int
    train_stream_seed: int
    eval_stream_seed: int

    @property
    def run_id(self) -> str:
        return f"{self.method}_{self.control_name}={self.control_value}_seed={self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SynthSAEBenchRunSpec:
        return cls(**values)


def default_sweep_config(
    fast: bool = False, calibration: bool = False
) -> SynthSAEBenchSweepConfig:
    data = (
        SynthSAEBenchDataConfig(n_train=2_048, n_test=256)
        if fast
        else SynthSAEBenchDataConfig()
    )
    training = (
        SynthSAEBenchTrainingConfig(
            batch_size=256,
            history_every=1,
            dead_feature_window=2,
            feature_sampling_window=2,
            n_batches_for_norm_estimate=2,
            heatmap_samples=16,
            resume_every=1,
        )
        if fast
        else SynthSAEBenchTrainingConfig()
    )
    controls = (
        FAST_CONTROLS
        if fast
        else CALIBRATION_CONTROLS if calibration else FINAL_CONTROLS
    )
    return SynthSAEBenchSweepConfig(
        experiment_name=(
            "stage2_synthsaebench16k_fast"
            if fast
            else (
                "stage2_synthsaebench16k_calibration"
                if calibration
                else "stage2_synthsaebench16k_l0calibrated"
            )
        ),
        data=data,
        training=training,
        controls={name: list(values) for name, values in controls.items()},
    )


def _sample_token(value: int) -> str:
    nearest_million = round(value / 1_000_000)
    if abs(value - nearest_million * 1_000_000) <= 1_024:
        return f"{nearest_million}m"
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def sweep_experiment_id(config: SynthSAEBenchSweepConfig) -> str:
    config.validate()
    seed_token = (
        f"seed{config.seeds[0]}"
        if len(config.seeds) == 1
        else "seeds" + "-".join(str(seed) for seed in config.seeds)
    )
    return (
        f"{config.experiment_name}_sae{config.data.sae_width}"
        f"_train{_sample_token(config.data.n_train)}"
        f"_test{_sample_token(config.data.n_test)}_{seed_token}"
    )


def default_sweep_dir(
    project_root: Path | str, config: SynthSAEBenchSweepConfig
) -> Path:
    return Path(project_root) / "outputs" / "runs" / sweep_experiment_id(config)


def build_specs(config: SynthSAEBenchSweepConfig) -> list[SynthSAEBenchRunSpec]:
    config.validate()
    specs: list[SynthSAEBenchRunSpec] = []
    for seed in config.seeds:
        for method in METHOD_ORDER:
            if method not in config.methods:
                continue
            seeds = {
                "init_seed": 50_000 + seed,
                "calibration_seed": 20_000 + seed,
                "train_stream_seed": 30_000 + seed,
                "eval_stream_seed": 40_000 + seed,
            }
            specs.extend(
                SynthSAEBenchRunSpec(
                    method=method,
                    control_name=CONTROL_NAMES[method],
                    control_value=value,
                    seed=seed,
                    **seeds,
                )
                for value in config.controls[method]
            )
    return specs


def run_directory(
    sweep_dir: Path | str, spec: SynthSAEBenchRunSpec
) -> Path:
    return Path(sweep_dir) / "runs" / spec.method / spec.run_id


def load_benchmark_model(
    config: SynthSAEBenchSweepConfig, device: torch.device | str
) -> tuple[SyntheticModel, Path]:
    """Load and verify the exact pretrained snapshot instead of rebuilding a variant."""

    config.validate()
    snapshot = Path(
        snapshot_download(
            repo_id=config.data.model_id,
            revision=config.data.revision,
        )
    )
    config_path = snapshot / "synthetic_model_config.json"
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if digest != config.data.model_config_sha256:
        raise RuntimeError(
            "Pinned SynthSAEBench model config hash mismatch: "
            f"expected {config.data.model_config_sha256}, got {digest}."
        )
    # SyntheticModel internally uses SAELens' all-visible-device seed context
    # even when loading saved weights.  Scope that construction seed to this
    # scheduler worker's assigned device.
    with patch.object(
        synthetic_model_module,
        "temporary_seed",
        lambda seed: temporary_seed_for_device(seed, device),
    ):
        model = SyntheticModel.load_from_disk(snapshot, device=str(device))
    dimensions = (model.cfg.hidden_dim, model.cfg.num_features)
    expected = (config.data.input_dim, config.data.ground_truth_num_features)
    if dimensions != expected:
        raise RuntimeError(
            f"Pinned SynthSAEBench dimensions changed: expected {expected}, got {dimensions}."
        )
    actual_scale_children = model.cfg.hierarchy.scale_children_by_parent
    if actual_scale_children is not config.data.scale_children_by_parent:
        raise RuntimeError(
            "Pinned SynthSAEBench hierarchy semantics changed: expected "
            f"scale_children_by_parent={config.data.scale_children_by_parent}, "
            f"got {actual_scale_children}."
        )
    return model, snapshot


def build_model(
    config: SynthSAEBenchSweepConfig,
    spec: SynthSAEBenchRunSpec,
    device: torch.device | str = "cpu",
) -> TrainingSAE[Any]:
    """Build one native SAELens model with benchmark-specific architecture settings."""

    config.validate()
    data, training, value = config.data, config.training, spec.control_value
    common = {
        "d_in": data.input_dim,
        "d_sae": data.sae_width,
        "device": str(device),
        "normalize_activations": "none",
    }
    warmup = config.total_training_steps // 3
    if spec.method == "vgsae":
        return VGTrainingSAE(
            VGTrainingSAEConfig(
                **common,
                beta=training.beta,
                beta_mode=training.beta_mode,
                lambda_sparsity=float(value),
                inference_threshold=training.mask_threshold,
            )
        )
    if spec.method == "l1":
        return StandardTrainingSAE(
            StandardTrainingSAEConfig(
                **common,
                l1_coefficient=float(value),
                l1_warm_up_steps=warmup,
            )
        )
    if spec.method == "topk":
        return TopKTrainingSAE(
            TopKTrainingSAEConfig(
                **common,
                k=int(value),
                rescale_acts_by_decoder_norm=True,
            )
        )
    if spec.method == "batchtopk":
        return BatchTopKTrainingSAE(
            BatchTopKTrainingSAEConfig(
                **common,
                k=float(value),
                rescale_acts_by_decoder_norm=True,
            )
        )
    if spec.method == "jumprelu":
        return JumpReLUTrainingSAE(
            JumpReLUTrainingSAEConfig(
                **{**common, "normalize_activations": "expected_average_only_in"},
                l0_coefficient=float(value),
                l0_warm_up_steps=0,
                jumprelu_sparsity_loss_mode="step",
                jumprelu_bandwidth=1.0,
                jumprelu_init_threshold=1.0,
                jumprelu_tanh_scale=4.0,
                pre_act_loss_coefficient=None,
                decoder_init_norm=0.5,
            )
        )
    if spec.method == "gated":
        return GatedTrainingSAE(
            GatedTrainingSAEConfig(
                **common,
                l1_coefficient=float(value),
                l1_warm_up_steps=warmup,
            )
        )
    raise ValueError(f"Unknown method: {spec.method}")


def _state_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for name, value in state_dict.items()
    }


def save_checkpoint(
    path: Path | str,
    *,
    model: TrainingSAE[Any],
    config: SynthSAEBenchSweepConfig,
    spec: SynthSAEBenchRunSpec,
    step: int,
    n_training_samples: int,
    loss: float,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "checkpoint_kind": "last",
        "step": step,
        "n_training_samples": n_training_samples,
        "loss": loss,
        "sweep_config": config.to_dict(),
        "run_spec": spec.to_dict(),
        "model_config": model.cfg.to_dict(),
        "metadata": metadata or {},
        "model_state": _state_to_cpu(model.state_dict()),
    }
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    path: Path | str, device: torch.device | str = "cpu"
) -> tuple[TrainingSAE[Any], dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    architecture = payload["model_config"]["architecture"]
    model_class, config_class = get_sae_training_class(architecture)
    model_config = config_class.from_dict(payload["model_config"])
    model_config.device = str(device)
    model = model_class(model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


__all__ = [
    "BENCHMARK_CONFIG_SHA256",
    "BENCHMARK_INPUT_DIM",
    "BENCHMARK_MODEL_ID",
    "BENCHMARK_NUM_FEATURES",
    "BENCHMARK_REVISION",
    "BENCHMARK_SAE_WIDTH",
    "BENCHMARK_SCALE_CHILDREN_BY_PARENT",
    "CALIBRATION_CONTROLS",
    "DEFAULT_MAX_PER_DEVICE",
    "FAST_CONTROLS",
    "FINAL_CONTROLS",
    "METHOD_LABELS",
    "METHOD_ORDER",
    "SAELENS_REVISION",
    "SYNTHSAEBENCH_DATA_KIND",
    "SynthSAEBenchDataConfig",
    "SynthSAEBenchRunSpec",
    "SynthSAEBenchSweepConfig",
    "SynthSAEBenchTrainingConfig",
    "build_model",
    "build_specs",
    "capture_rng_state",
    "default_sweep_config",
    "default_sweep_dir",
    "load_benchmark_model",
    "load_checkpoint",
    "run_directory",
    "save_checkpoint",
    "sweep_experiment_id",
    "temporary_seed_for_device",
]
