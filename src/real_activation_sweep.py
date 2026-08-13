"""Stage-3 real-activation sweep configuration and native SAE factories.

This module deliberately owns its method order, target registry, and checkpoint
format.  Stage-1 and Stage-2 defaults therefore remain stable while the real
activation runner can reuse their orchestration conventions.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import torch
from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)
from sae_lens.registry import get_sae_training_class
from sae_lens.saes.sae import SAEMetadata, TrainingSAE

from .saelens_vg import VGTrainingSAE, VGTrainingSAEConfig


REAL_ACTIVATION_DATA_KIND = "real_llm_activation"
SAE_WIDTH = 32_768
MODEL_CLASS_NAME = "AutoModelForCausalLM"

# These immutable revisions were resolved from the Hugging Face APIs when the
# Stage-3 protocol was defined.  Pinning them prevents a moving ``main`` branch
# from silently changing either the activations or tokenizer.
GEMMA_MODEL_ID = "google/gemma-2-2b"
GEMMA_MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B"
LLAMA_MODEL_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
SOURCE_DATASET_ID = "monology/pile-uncopyrighted"
SOURCE_DATASET_REVISION = "3be90335b66f24456a5d6659d9c8d208c0357119"
GEMMA_DATASET_ID = "chanind/pile-uncopyrighted-gemma-1024-abbrv-2B"
GEMMA_DATASET_REVISION = "1a48ea543e13ffa22a9a29994e51f1d76a214c82"
GEMMA_LAYER12_DATASET_ID = "chanind/pile-uncopyrighted-gemma-1024-abbrv-2B"
GEMMA_LAYER12_DATASET_REVISION = "1a48ea543e13ffa22a9a29994e51f1d76a214c82"
LLAMA_DATASET_ID = "chanind/pile-uncopyrighted-llama-3_2-1024-abbrv-1B"
LLAMA_DATASET_REVISION = "31842561ac311c3103900919f969b7069e30aab1"
DATASET_SHARD_PATH_PATTERN = "data/train-{index:05d}-of-00064.parquet"
GEMMA_DATASET_SHARD_ROWS = (36_669,) * 42 + (36_668,) * 22
GEMMA_LAYER12_DATASET_SHARD_ROWS = (36_669,) * 42 + (36_668,) * 22
LLAMA_DATASET_SHARD_ROWS = (21_141,) * 12 + (21_140,) * 52
SPARSE_BUT_WRONG_REVISION = "d5886b540dc5b9cac4f76e6db2b0cce1b0b7c585"
SAELENS_REVISION = "8be14080485952f729ed58d674bcddf9778e0aa4"

STAGE3_PACKAGE_DISTRIBUTIONS: Mapping[str, str] = MappingProxyType(
    {
        "datasets": "datasets",
        "huggingface_hub": "huggingface-hub",
        "tokenizers": "tokenizers",
        "transformer_lens": "transformer-lens",
        "transformers": "transformers",
    }
)


def augment_stage3_runtime_provenance(
    base: Mapping[str, Any],
) -> dict[str, Any]:
    """Add packages that determine real activations without changing Stage 1/2."""

    provenance = dict(base)
    for field, distribution in STAGE3_PACKAGE_DISTRIBUTIONS.items():
        try:
            provenance[field] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            provenance[field] = None
    fingerprint_payload = {
        key: value
        for key, value in provenance.items()
        if key not in {"git_revision", "pipeline_fingerprint"}
    }
    provenance["pipeline_fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return provenance

PAPER_REPORTED_TRAIN_TOKENS = 500_000_000
DEFAULT_TRAIN_TOKENS = 500_002_816
GEMMA_LAYER12_PAPER_REPORTED_TRAIN_TOKENS = 1_000_000_000
GEMMA_LAYER12_TRAIN_TOKENS = 1_000_001_536
DEFAULT_EVAL_TOKENS = 1_048_576
DEFAULT_DOWNSTREAM_EVAL_TOKENS = 16_384
DEFAULT_CONTEXT_SIZE = 1_024
DEFAULT_BATCH_SIZE = 4_096
JUMPRELU_WARMUP_TOKENS = 100_000_000
TRAINING_BUDGET_RATIONALE = (
    "The default effective budget rounds the target's paper-reported 500M or "
    "1B request upward to the next complete 4096-token batch; both the paper "
    "request and resolved n_train_tokens (including explicit CLI/dev overrides) "
    "are retained in every run artifact."
)

STAGE3_METHOD_ORDER = ("vgsae", "l1", "batchtopk", "jumprelu")
STAGE3_METHOD_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "vgsae": "VG-SAE",
        "l1": "L1/ReLU SAE",
        "batchtopk": "BatchTopK SAE",
        "jumprelu": "JumpReLU SAE",
    }
)
STAGE3_CONTROL_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "vgsae": "lambda_sparsity",
        "l1": "l1_coefficient",
        "batchtopk": "k",
        "jumprelu": "l0_coefficient",
    }
)

PAPER_BATCHTOPK_K_GRID = (
    10,
    20,
    40,
    60,
    100,
    150,
    200,
    250,
    500,
    750,
    1_000,
    1_500,
    2_000,
)
GEMMA_LAYER5_BATCHTOPK_K_GRID = (
    10,
    40,
    80,
    120,
    160,
    200,
    240,
    260,
    300,
    400,
    500,
    750,
    1_000,
    1_500,
    2_000,
)
GEMMA_LAYER12_BATCHTOPK_K_GRID = (
    10,
    20,
    40,
    60,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    300,
    350,
    400,
    450,
    500,
    750,
    1_000,
    1_500,
    2_000,
    2_500,
)
PAPER_JUMPRELU_COEFFICIENT_GRID = (
    0.125,
    0.25,
    0.375,
    0.4375,
    0.5,
    0.5625,
    0.625,
    0.6875,
    0.75,
)

# The queued Stage-2 density sweep used 26 log-spaced targets from 1e-3 to
# 1e-1.  Keep the anchors here as immutable tuples rather than reading an
# ignored output directory at runtime.
STAGE2_DENSITY_ANCHORS = tuple(
    10.0 ** (-3.0 + 2.0 * index / 25.0) for index in range(26)
)
STAGE2_L1_CONTROL_ANCHORS = (
    13.7,
    12.4,
    11.0,
    9.66,
    8.32,
    6.97,
    5.62,
    4.24,
    3.0,
    2.12,
    1.61,
    1.31,
    1.11,
    0.982,
    0.907,
    0.837,
    0.774,
    0.716,
    0.662,
    0.613,
    0.567,
    0.524,
    0.485,
    0.448,
    0.415,
    0.383,
)
STAGE2_VG_CONTROL_ANCHORS = (
    21.0,
    18.8,
    16.7,
    14.5,
    12.4,
    10.3,
    7.94,
    6.25,
    3.81,
    2.58,
    2.21,
    1.97,
    1.79,
    1.64,
    1.51,
    1.45,
    1.38,
    1.32,
    1.26,
    1.2,
    1.14,
    1.08,
    1.02,
    0.956,
    0.895,
    0.834,
)

CONTROL_GRID_PROVENANCE = (
    "BatchTopK and JumpReLU grids follow sparse-but-wrong-paper@"
    f"{SPARSE_BUT_WRONG_REVISION}; L1 and learned-beta VG anchors come from "
    "the queued Stage-2 26-point rho=1e-3..1e-1 log-density protocol."
)
CONTROL_GRID_RATIONALE = (
    "For L1 and VG, linearly interpolate or extrapolate the Stage-2 control "
    "as a function of log10(target density), evaluated at each published "
    "BatchTopK K divided by the Stage-3 width 32768. Achieved hard L0, not "
    "the requested density or coefficient order, remains the evaluation axis."
)


def _linear_log_density_control(
    density: float,
    control_anchors: tuple[float, ...],
) -> float:
    """Linearly interpolate, or endpoint-extrapolate, against log10 density."""

    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be positive and finite.")
    if len(control_anchors) != len(STAGE2_DENSITY_ANCHORS):
        raise ValueError("control anchors must align with Stage-2 density anchors.")
    logged = tuple(math.log10(value) for value in STAGE2_DENSITY_ANCHORS)
    target = math.log10(density)
    left = bisect_right(logged, target) - 1
    left = max(0, min(left, len(logged) - 2))
    fraction = (target - logged[left]) / (logged[left + 1] - logged[left])
    return float(
        control_anchors[left]
        + fraction * (control_anchors[left + 1] - control_anchors[left])
    )


def interpolate_stage2_control(method: str, target_density: float) -> float:
    """Map a Stage-3 density to the queued Stage-2 L1 or learned-VG control."""

    anchors = {
        "l1": STAGE2_L1_CONTROL_ANCHORS,
        "vgsae": STAGE2_VG_CONTROL_ANCHORS,
    }.get(method)
    if anchors is None:
        raise ValueError("Stage-2 interpolation is defined only for 'l1' and 'vgsae'.")
    return _linear_log_density_control(target_density, anchors)


PAPER_TARGET_DENSITIES = tuple(value / SAE_WIDTH for value in PAPER_BATCHTOPK_K_GRID)
DEFAULT_STAGE3_CONTROLS: Mapping[str, tuple[float | int, ...]] = MappingProxyType(
    {
        "vgsae": tuple(
            interpolate_stage2_control("vgsae", density)
            for density in PAPER_TARGET_DENSITIES
        ),
        "l1": tuple(
            interpolate_stage2_control("l1", density)
            for density in PAPER_TARGET_DENSITIES
        ),
        "batchtopk": PAPER_BATCHTOPK_K_GRID,
        "jumprelu": PAPER_JUMPRELU_COEFFICIENT_GRID,
    }
)

PAPER_BATCHTOPK_K_GRIDS: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "gemma-2-2b-layer5": GEMMA_LAYER5_BATCHTOPK_K_GRID,
        "gemma-2-2b-layer12": GEMMA_LAYER12_BATCHTOPK_K_GRID,
        "llama-3.2-1b-layer7": PAPER_BATCHTOPK_K_GRID,
    }
)


def default_controls_for_target(
    target_name: str,
) -> dict[str, list[float | int]]:
    """Return the paper-main K grid and density-matched Stage-2 hypotheses."""

    try:
        k_grid = PAPER_BATCHTOPK_K_GRIDS[target_name]
    except KeyError as error:
        raise ValueError(f"Unknown Stage-3 target: {target_name!r}.") from error
    densities = tuple(value / SAE_WIDTH for value in k_grid)
    return {
        "vgsae": [
            interpolate_stage2_control("vgsae", density) for density in densities
        ],
        "l1": [interpolate_stage2_control("l1", density) for density in densities],
        "batchtopk": list(k_grid),
        "jumprelu": list(PAPER_JUMPRELU_COEFFICIENT_GRID),
    }


@dataclass(frozen=True)
class RealActivationTarget:
    """One immutable Hugging Face model/layer activation target."""

    name: str
    model_id: str
    model_revision: str
    layer: int
    hook_name: str
    paper_hook_name: str
    input_dim: int
    dataset_id: str
    dataset_revision: str
    dataset_shard_rows: tuple[int, ...]
    default_seeds: tuple[int, ...]
    paper_reported_train_tokens: int
    train_tokens: int


REAL_MODEL_TARGETS: Mapping[str, RealActivationTarget] = MappingProxyType(
    {
        "gemma-2-2b-layer5": RealActivationTarget(
            name="gemma-2-2b-layer5",
            model_id=GEMMA_MODEL_ID,
            model_revision=GEMMA_MODEL_REVISION,
            layer=5,
            hook_name="model.layers.5",
            paper_hook_name="blocks.5.hook_resid_post",
            input_dim=2_304,
            dataset_id=GEMMA_DATASET_ID,
            dataset_revision=GEMMA_DATASET_REVISION,
            dataset_shard_rows=GEMMA_DATASET_SHARD_ROWS,
            default_seeds=(0, 1, 2),
            paper_reported_train_tokens=PAPER_REPORTED_TRAIN_TOKENS,
            train_tokens=DEFAULT_TRAIN_TOKENS,
        ),
        "gemma-2-2b-layer12": RealActivationTarget(
            name="gemma-2-2b-layer12",
            model_id=GEMMA_MODEL_ID,
            model_revision=GEMMA_MODEL_REVISION,
            layer=12,
            hook_name="model.layers.12",
            paper_hook_name="blocks.12.hook_resid_post",
            input_dim=2_304,
            dataset_id=GEMMA_LAYER12_DATASET_ID,
            dataset_revision=GEMMA_LAYER12_DATASET_REVISION,
            dataset_shard_rows=GEMMA_LAYER12_DATASET_SHARD_ROWS,
            default_seeds=(0,),
            paper_reported_train_tokens=GEMMA_LAYER12_PAPER_REPORTED_TRAIN_TOKENS,
            train_tokens=GEMMA_LAYER12_TRAIN_TOKENS,
        ),
        "llama-3.2-1b-layer7": RealActivationTarget(
            name="llama-3.2-1b-layer7",
            model_id=LLAMA_MODEL_ID,
            model_revision=LLAMA_MODEL_REVISION,
            layer=7,
            hook_name="model.layers.7",
            paper_hook_name="blocks.7.hook_resid_post",
            input_dim=2_048,
            dataset_id=LLAMA_DATASET_ID,
            dataset_revision=LLAMA_DATASET_REVISION,
            dataset_shard_rows=LLAMA_DATASET_SHARD_ROWS,
            default_seeds=(0, 1, 2),
            paper_reported_train_tokens=PAPER_REPORTED_TRAIN_TOKENS,
            train_tokens=DEFAULT_TRAIN_TOKENS,
        ),
    }
)
DEFAULT_TARGET_NAME = "gemma-2-2b-layer12"
_DEFAULT_TARGET = REAL_MODEL_TARGETS[DEFAULT_TARGET_NAME]


@dataclass(frozen=True)
class RealActivationDataConfig:
    """Pinned model/dataset identity and token budgets for one target."""

    kind: str = REAL_ACTIVATION_DATA_KIND
    target_name: str = DEFAULT_TARGET_NAME
    model_id: str = _DEFAULT_TARGET.model_id
    model_revision: str = _DEFAULT_TARGET.model_revision
    model_class_name: str = MODEL_CLASS_NAME
    layer: int = _DEFAULT_TARGET.layer
    hook_name: str = _DEFAULT_TARGET.hook_name
    paper_hook_name: str = _DEFAULT_TARGET.paper_hook_name
    input_dim: int = _DEFAULT_TARGET.input_dim
    sae_width: int = SAE_WIDTH
    dataset_id: str = _DEFAULT_TARGET.dataset_id
    dataset_revision: str = _DEFAULT_TARGET.dataset_revision
    dataset_shard_path_pattern: str = DATASET_SHARD_PATH_PATTERN
    dataset_shard_rows: tuple[int, ...] = _DEFAULT_TARGET.dataset_shard_rows
    source_dataset_id: str = SOURCE_DATASET_ID
    source_dataset_revision: str = SOURCE_DATASET_REVISION
    dataset_split: str = "train"
    is_dataset_tokenized: bool = True
    prepend_bos: bool = True
    context_size: int = DEFAULT_CONTEXT_SIZE
    paper_reported_train_tokens: int = (
        GEMMA_LAYER12_PAPER_REPORTED_TRAIN_TOKENS
    )
    training_budget_rationale: str = TRAINING_BUDGET_RATIONALE
    train_token_offset: int = 0
    n_train_tokens: int = GEMMA_LAYER12_TRAIN_TOKENS
    eval_token_offset: int = GEMMA_LAYER12_TRAIN_TOKENS
    n_eval_tokens: int = DEFAULT_EVAL_TOKENS
    n_downstream_eval_tokens: int = DEFAULT_DOWNSTREAM_EVAL_TOKENS

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RealActivationDataConfig:
        payload = dict(values)
        payload["dataset_shard_rows"] = tuple(payload["dataset_shard_rows"])
        return cls(**payload)


def target_data_config(
    target_name: str,
    *,
    n_train_tokens: int | None = None,
    n_eval_tokens: int = DEFAULT_EVAL_TOKENS,
    n_downstream_eval_tokens: int = DEFAULT_DOWNSTREAM_EVAL_TOKENS,
) -> RealActivationDataConfig:
    """Construct the immutable identity fields for one supported target."""

    try:
        target = REAL_MODEL_TARGETS[target_name]
    except KeyError as error:
        raise ValueError(f"Unknown Stage-3 target: {target_name!r}.") from error
    resolved_train_tokens = target.train_tokens if n_train_tokens is None else n_train_tokens
    return RealActivationDataConfig(
        target_name=target.name,
        model_id=target.model_id,
        model_revision=target.model_revision,
        layer=target.layer,
        hook_name=target.hook_name,
        paper_hook_name=target.paper_hook_name,
        input_dim=target.input_dim,
        dataset_id=target.dataset_id,
        dataset_revision=target.dataset_revision,
        dataset_shard_rows=target.dataset_shard_rows,
        paper_reported_train_tokens=target.paper_reported_train_tokens,
        n_train_tokens=resolved_train_tokens,
        eval_token_offset=resolved_train_tokens,
        n_eval_tokens=n_eval_tokens,
        n_downstream_eval_tokens=n_downstream_eval_tokens,
    )


@dataclass(frozen=True)
class RealActivationTrainingConfig:
    """Paper-aligned optimizer and architecture settings for Stage 3."""

    batch_size: int = DEFAULT_BATCH_SIZE
    vg_learning_rate: float = 3.0e-4
    l1_learning_rate: float = 3.0e-4
    batchtopk_learning_rate: float = 3.0e-4
    jumprelu_learning_rate: float = 2.0e-4
    lr_scheduler_name: str = "constant"
    l1_warmup_fraction: float = 1.0 / 3.0
    jumprelu_coefficient_warmup_tokens: int = JUMPRELU_WARMUP_TOKENS
    batchtopk_rescale_acts_by_decoder_norm: bool = True
    batchtopk_decoder_init_norm: float = 0.1
    jumprelu_sparsity_loss_mode: str = "tanh"
    jumprelu_bandwidth: float = 2.0
    jumprelu_tanh_scale: float = 4.0
    jumprelu_pre_act_loss_coefficient: float = 3.0e-6
    jumprelu_init_threshold: float = 0.1
    jumprelu_decoder_init_norm: float = 0.1
    beta: float = 1.0
    beta_mode: Literal["learned"] = "learned"
    mask_threshold: float = 0.5
    preview_tokens: int = 80
    history_every: int = 1_000
    resume_every: int = 10_000
    dead_feature_window: int = 1_000
    feature_sampling_window: int = 2_000
    n_batches_for_norm_estimate: int = 1_000
    autocast_sae: bool = True
    autocast_data: bool = True
    store_batch_size_prompts: int = 12
    eval_store_batch_size_prompts: int = 6
    n_batches_in_buffer: int = 64
    activations_mixing_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("Stage-3 VG beta must be positive and finite.")
        if self.beta_mode != "learned":
            raise ValueError("Stage-3 fixes VG beta_mode='learned'.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RealActivationTrainingConfig:
        return cls(**values)


@dataclass
class RealActivationSweepConfig:
    """Serializable sweep protocol for one model/layer activation condition."""

    experiment_name: str = "stage3_real_activation"
    data: RealActivationDataConfig = field(default_factory=RealActivationDataConfig)
    training: RealActivationTrainingConfig = field(
        default_factory=RealActivationTrainingConfig
    )
    seeds: list[int] = field(default_factory=lambda: list(_DEFAULT_TARGET.default_seeds))
    methods: list[str] = field(default_factory=lambda: list(STAGE3_METHOD_ORDER))
    controls: dict[str, list[float | int]] = field(
        default_factory=lambda: default_controls_for_target(DEFAULT_TARGET_NAME)
    )
    control_grid_provenance: str = CONTROL_GRID_PROVENANCE
    control_grid_rationale: str = CONTROL_GRID_RATIONALE
    saelens_revision: str = SAELENS_REVISION
    sparse_but_wrong_revision: str = SPARSE_BUT_WRONG_REVISION
    wandb_project: str = "vg-sae"

    def validate(self) -> None:
        try:
            target = REAL_MODEL_TARGETS[self.data.target_name]
        except KeyError as error:
            raise ValueError(
                f"Unknown Stage-3 target: {self.data.target_name!r}."
            ) from error
        immutable = {
            "kind": (self.data.kind, REAL_ACTIVATION_DATA_KIND),
            "model_id": (self.data.model_id, target.model_id),
            "model_revision": (self.data.model_revision, target.model_revision),
            "model_class_name": (self.data.model_class_name, MODEL_CLASS_NAME),
            "layer": (self.data.layer, target.layer),
            "hook_name": (self.data.hook_name, target.hook_name),
            "paper_hook_name": (
                self.data.paper_hook_name,
                target.paper_hook_name,
            ),
            "input_dim": (self.data.input_dim, target.input_dim),
            "sae_width": (self.data.sae_width, SAE_WIDTH),
            "dataset_id": (self.data.dataset_id, target.dataset_id),
            "dataset_revision": (
                self.data.dataset_revision,
                target.dataset_revision,
            ),
            "dataset_shard_path_pattern": (
                self.data.dataset_shard_path_pattern,
                DATASET_SHARD_PATH_PATTERN,
            ),
            "dataset_shard_rows": (
                self.data.dataset_shard_rows,
                target.dataset_shard_rows,
            ),
            "source_dataset_id": (
                self.data.source_dataset_id,
                SOURCE_DATASET_ID,
            ),
            "source_dataset_revision": (
                self.data.source_dataset_revision,
                SOURCE_DATASET_REVISION,
            ),
            "dataset_split": (self.data.dataset_split, "train"),
            "is_dataset_tokenized": (self.data.is_dataset_tokenized, True),
            "prepend_bos": (self.data.prepend_bos, True),
            "context_size": (self.data.context_size, DEFAULT_CONTEXT_SIZE),
            "paper_reported_train_tokens": (
                self.data.paper_reported_train_tokens,
                target.paper_reported_train_tokens,
            ),
            "training_budget_rationale": (
                self.data.training_budget_rationale,
                TRAINING_BUDGET_RATIONALE,
            ),
        }
        changed = [
            name for name, (actual, expected) in immutable.items() if actual != expected
        ]
        if changed:
            raise ValueError(
                "Stage-3 fixes model/layer dimensions and revisions; unsupported "
                f"override(s): {', '.join(changed)}"
            )
        if self.data.context_size <= 0:
            raise ValueError("context_size must be positive.")
        if self.data.n_train_tokens <= 0 or self.data.n_eval_tokens <= 0:
            raise ValueError("Training and evaluation token budgets must be positive.")
        if not 0 < self.data.n_downstream_eval_tokens <= self.data.n_eval_tokens:
            raise ValueError(
                "n_downstream_eval_tokens must lie in (0, n_eval_tokens]."
            )
        if self.data.train_token_offset < 0 or self.data.eval_token_offset < 0:
            raise ValueError("Dataset token offsets must be nonnegative.")
        train_end = self.data.train_token_offset + self.data.n_train_tokens
        if self.data.eval_token_offset < train_end:
            raise ValueError(
                "Held-out evaluation tokens must begin after the training-token range."
            )
        dataset_tokens = sum(self.data.dataset_shard_rows) * self.data.context_size
        eval_end = self.data.eval_token_offset + self.data.n_eval_tokens
        if eval_end > dataset_tokens:
            raise ValueError(
                "Training/evaluation ranges exceed the pinned tokenized dataset."
            )
        context_aligned = {
            "train_token_offset": self.data.train_token_offset,
            "eval_token_offset": self.data.eval_token_offset,
            "n_train_tokens": self.data.n_train_tokens,
            "n_eval_tokens": self.data.n_eval_tokens,
            "n_downstream_eval_tokens": self.data.n_downstream_eval_tokens,
        }
        if misaligned := [
            name
            for name, value in context_aligned.items()
            if value % self.data.context_size
        ]:
            raise ValueError(
                "Token ranges must align to complete cached contexts: "
                f"{', '.join(misaligned)}"
            )

        training = self.training
        if training.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if training.batch_size % self.data.context_size:
            raise ValueError("batch_size must contain complete cached contexts.")
        if self.data.n_train_tokens < training.batch_size:
            raise ValueError(
                "Training budget must contain at least one complete batch."
            )
        if self.data.n_train_tokens % training.batch_size:
            raise ValueError("Training token budget must be batch-aligned.")
        learning_rates = (
            training.vg_learning_rate,
            training.l1_learning_rate,
            training.batchtopk_learning_rate,
            training.jumprelu_learning_rate,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in learning_rates):
            raise ValueError("All method learning rates must be positive and finite.")
        if training.lr_scheduler_name != "constant":
            raise ValueError(
                "Stage-3 uses the paper's constant learning-rate schedule."
            )
        if training.l1_warmup_fraction != 1.0 / 3.0:
            raise ValueError("Stage-3 L1 uses a one-third coefficient warmup.")
        if training.jumprelu_coefficient_warmup_tokens != JUMPRELU_WARMUP_TOKENS:
            raise ValueError("Stage-3 JumpReLU uses a 100M-token coefficient warmup.")
        if not training.batchtopk_rescale_acts_by_decoder_norm:
            raise ValueError("Stage-3 BatchTopK requires decoder-norm act rescaling.")
        if training.batchtopk_decoder_init_norm != 0.1:
            raise ValueError("Stage-3 fixes BatchTopK decoder_init_norm=0.1.")
        jump_settings = {
            "jumprelu_sparsity_loss_mode": (
                training.jumprelu_sparsity_loss_mode,
                "tanh",
            ),
            "jumprelu_bandwidth": (training.jumprelu_bandwidth, 2.0),
            "jumprelu_tanh_scale": (training.jumprelu_tanh_scale, 4.0),
            "jumprelu_pre_act_loss_coefficient": (
                training.jumprelu_pre_act_loss_coefficient,
                3.0e-6,
            ),
            "jumprelu_init_threshold": (training.jumprelu_init_threshold, 0.1),
            "jumprelu_decoder_init_norm": (
                training.jumprelu_decoder_init_norm,
                0.1,
            ),
        }
        changed_jump = [
            name
            for name, (actual, expected) in jump_settings.items()
            if actual != expected
        ]
        if changed_jump:
            raise ValueError(
                "Stage-3 fixes the published JumpReLU setup; unsupported "
                f"override(s): {', '.join(changed_jump)}"
            )
        if not 0.0 <= training.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must lie in [0, 1].")
        if training.preview_tokens < 0:
            raise ValueError("preview_tokens must be nonnegative.")
        if training.store_batch_size_prompts <= 0:
            raise ValueError("store_batch_size_prompts must be positive.")
        if training.eval_store_batch_size_prompts <= 0:
            raise ValueError("eval_store_batch_size_prompts must be positive.")
        if training.n_batches_in_buffer <= 0:
            raise ValueError("n_batches_in_buffer must be positive.")
        if not 0.0 <= training.activations_mixing_fraction <= 1.0:
            raise ValueError("activations_mixing_fraction must lie in [0, 1].")
        positive_integer_fields = {
            "history_every": training.history_every,
            "resume_every": training.resume_every,
            "dead_feature_window": training.dead_feature_window,
            "feature_sampling_window": training.feature_sampling_window,
            "n_batches_for_norm_estimate": training.n_batches_for_norm_estimate,
        }
        if bad := [
            name for name, value in positive_integer_fields.items() if value <= 0
        ]:
            raise ValueError(f"Training counters must be positive: {', '.join(bad)}")

        if not self.methods:
            raise ValueError("At least one Stage-3 SAE method is required.")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("Stage-3 methods must be unique.")
        if unknown := set(self.methods) - set(STAGE3_METHOD_ORDER):
            raise ValueError(f"Unknown Stage-3 methods: {sorted(unknown)}")
        if extra_controls := set(self.controls) - set(STAGE3_METHOD_ORDER):
            raise ValueError(
                f"Controls provided for unknown methods: {sorted(extra_controls)}"
            )
        if not self.seeds:
            raise ValueError("At least one Stage-3 experiment seed is required.")
        if any(not isinstance(seed, Integral) or seed < 0 for seed in self.seeds):
            raise ValueError("Stage-3 seeds must be nonnegative integers.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Stage-3 seeds must be unique.")
        for method in self.methods:
            values = self.controls.get(method)
            if not values:
                raise ValueError(f"No controls configured for method {method!r}.")
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"Controls for {method!r} must be finite.")
            if len({float(value) for value in values}) != len(values):
                raise ValueError(f"Controls for {method!r} must be unique.")
            if method == "batchtopk" and any(
                not 0.0 < float(value) <= self.data.sae_width for value in values
            ):
                raise ValueError("BatchTopK controls must lie in (0, sae_width].")
            if method != "batchtopk" and any(float(value) < 0.0 for value in values):
                raise ValueError(f"Controls for {method!r} must be nonnegative.")
        if not self.control_grid_provenance.strip():
            raise ValueError("control_grid_provenance must be recorded.")
        if not self.control_grid_rationale.strip():
            raise ValueError("control_grid_rationale must be recorded.")
        if self.saelens_revision != SAELENS_REVISION:
            raise ValueError("Stage-3 fixes the inspected SAELens source revision.")
        if self.sparse_but_wrong_revision != SPARSE_BUT_WRONG_REVISION:
            raise ValueError(
                "Stage-3 fixes the inspected sparse-but-wrong-paper revision."
            )

    @property
    def total_training_steps(self) -> int:
        """Match SAELens' complete-batch training-token convention."""

        return self.data.n_train_tokens // self.training.batch_size

    @property
    def l1_warmup_steps(self) -> int:
        return self.total_training_steps // 3

    @property
    def jumprelu_warmup_steps(self) -> int:
        return (
            self.training.jumprelu_coefficient_warmup_tokens
            // self.training.batch_size
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RealActivationSweepConfig:
        payload = dict(values)
        payload["data"] = RealActivationDataConfig.from_dict(payload["data"])
        payload["training"] = RealActivationTrainingConfig.from_dict(
            payload["training"]
        )
        config = cls(**payload)
        config.validate()
        return config


@dataclass(frozen=True)
class RealActivationRunSpec:
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
        value = (
            str(self.control_value)
            if isinstance(self.control_value, Integral)
            else repr(float(self.control_value))
        )
        return f"{self.method}_{self.control_name}={value}_seed={self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RealActivationRunSpec:
        return cls(**values)


def default_sweep_config(
    target_name: str = DEFAULT_TARGET_NAME,
    *,
    n_train_tokens: int | None = None,
    n_eval_tokens: int = DEFAULT_EVAL_TOKENS,
    n_downstream_eval_tokens: int = DEFAULT_DOWNSTREAM_EVAL_TOKENS,
) -> RealActivationSweepConfig:
    config = RealActivationSweepConfig(
        data=target_data_config(
            target_name,
            n_train_tokens=n_train_tokens,
            n_eval_tokens=n_eval_tokens,
            n_downstream_eval_tokens=n_downstream_eval_tokens,
        ),
        seeds=list(REAL_MODEL_TARGETS[target_name].default_seeds),
        controls=default_controls_for_target(target_name),
    )
    config.validate()
    return config


def default_sweep_configs(
    *,
    n_eval_tokens: int = DEFAULT_EVAL_TOKENS,
    n_downstream_eval_tokens: int = DEFAULT_DOWNSTREAM_EVAL_TOKENS,
) -> list[RealActivationSweepConfig]:
    """Return one independent sweep config for each paper-main target."""

    return [
        default_sweep_config(
            target_name,
            n_eval_tokens=n_eval_tokens,
            n_downstream_eval_tokens=n_downstream_eval_tokens,
        )
        for target_name in REAL_MODEL_TARGETS
    ]


def _sample_token(value: int) -> str:
    nearest_million = round(value / 1_000_000)
    if abs(value - nearest_million * 1_000_000) <= DEFAULT_BATCH_SIZE:
        return f"{nearest_million}m"
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def sweep_experiment_id(config: RealActivationSweepConfig) -> str:
    config.validate()
    seed_token = (
        f"seed{config.seeds[0]}"
        if len(config.seeds) == 1
        else "seeds" + "-".join(str(seed) for seed in config.seeds)
    )
    return (
        f"{config.experiment_name}_{config.data.target_name}"
        f"_sae{config.data.sae_width}"
        f"_train{_sample_token(config.data.n_train_tokens)}"
        f"_eval{_sample_token(config.data.n_eval_tokens)}"
        f"_ctx{config.data.context_size}"
        f"_beta_{config.training.beta_mode}_{seed_token}"
    )


def default_sweep_dir(
    project_root: Path | str, config: RealActivationSweepConfig
) -> Path:
    return Path(project_root) / "outputs" / "runs" / sweep_experiment_id(config)


def build_specs(config: RealActivationSweepConfig) -> list[RealActivationRunSpec]:
    config.validate()
    specs: list[RealActivationRunSpec] = []
    for seed in config.seeds:
        shared_seeds = {
            "init_seed": 50_000 + seed,
            "calibration_seed": 20_000 + seed,
            "train_stream_seed": 30_000 + seed,
            "eval_stream_seed": 40_000 + seed,
        }
        for method in STAGE3_METHOD_ORDER:
            if method not in config.methods:
                continue
            specs.extend(
                RealActivationRunSpec(
                    method=method,
                    control_name=STAGE3_CONTROL_NAMES[method],
                    control_value=value,
                    seed=seed,
                    **shared_seeds,
                )
                for value in config.controls[method]
            )
    return specs


def run_directory(
    sweep_dir: Path | str, spec: RealActivationRunSpec
) -> Path:
    return Path(sweep_dir) / "runs" / spec.method / spec.run_id


def method_learning_rate(
    config: RealActivationSweepConfig, method: str
) -> float:
    """Return the runner-level optimizer LR without embedding it in SAE config."""

    config.validate()
    values = {
        "vgsae": config.training.vg_learning_rate,
        "l1": config.training.l1_learning_rate,
        "batchtopk": config.training.batchtopk_learning_rate,
        "jumprelu": config.training.jumprelu_learning_rate,
    }
    try:
        return values[method]
    except KeyError as error:
        raise ValueError(f"Unknown Stage-3 method: {method!r}.") from error


def _sae_metadata(config: RealActivationSweepConfig) -> SAEMetadata:
    data = config.data
    return SAEMetadata(
        model_name=data.model_id,
        model_revision=data.model_revision,
        model_class_name=data.model_class_name,
        hook_name=data.hook_name,
        paper_hook_name=data.paper_hook_name,
        hook_layer=data.layer,
        dataset_path=data.dataset_id,
        dataset_revision=data.dataset_revision,
        source_dataset_path=data.source_dataset_id,
        source_dataset_revision=data.source_dataset_revision,
        dataset_split=data.dataset_split,
        train_token_offset=data.train_token_offset,
        eval_token_offset=data.eval_token_offset,
        context_size=data.context_size,
        prepend_bos=data.prepend_bos,
        training_tokens=data.n_train_tokens,
    )


def build_model(
    config: RealActivationSweepConfig,
    spec: RealActivationRunSpec,
    device: torch.device | str = "cpu",
) -> TrainingSAE[Any]:
    """Build one official SAELens training SAE for the pinned real target."""

    config.validate()
    if spec.method not in config.methods:
        raise ValueError(f"Run method {spec.method!r} is not enabled in this sweep.")
    if spec.control_name != STAGE3_CONTROL_NAMES[spec.method]:
        raise ValueError(f"Control name does not match method {spec.method!r}.")
    common = {
        "d_in": config.data.input_dim,
        "d_sae": config.data.sae_width,
        "device": str(device),
        "normalize_activations": "none",
        "metadata": _sae_metadata(config),
    }
    value = spec.control_value
    if spec.method == "vgsae":
        return VGTrainingSAE(
            VGTrainingSAEConfig(
                **common,
                beta=config.training.beta,
                beta_mode="learned",
                lambda_sparsity=float(value),
                inference_threshold=config.training.mask_threshold,
            )
        )
    if spec.method == "l1":
        return StandardTrainingSAE(
            StandardTrainingSAEConfig(
                **common,
                l1_coefficient=float(value),
                l1_warm_up_steps=config.l1_warmup_steps,
            )
        )
    if spec.method == "batchtopk":
        return BatchTopKTrainingSAE(
            BatchTopKTrainingSAEConfig(
                **common,
                k=float(value),
                rescale_acts_by_decoder_norm=(
                    config.training.batchtopk_rescale_acts_by_decoder_norm
                ),
                decoder_init_norm=config.training.batchtopk_decoder_init_norm,
            )
        )
    if spec.method == "jumprelu":
        return JumpReLUTrainingSAE(
            JumpReLUTrainingSAEConfig(
                **common,
                l0_coefficient=float(value),
                l0_warm_up_steps=config.jumprelu_warmup_steps,
                jumprelu_sparsity_loss_mode="tanh",
                jumprelu_bandwidth=config.training.jumprelu_bandwidth,
                jumprelu_init_threshold=config.training.jumprelu_init_threshold,
                jumprelu_tanh_scale=config.training.jumprelu_tanh_scale,
                pre_act_loss_coefficient=(
                    config.training.jumprelu_pre_act_loss_coefficient
                ),
                decoder_init_norm=config.training.jumprelu_decoder_init_norm,
            )
        )
    raise ValueError(f"Unknown Stage-3 method: {spec.method!r}.")


def _state_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for name, value in state_dict.items()
    }


def save_checkpoint(
    path: Path | str,
    *,
    model: TrainingSAE[Any],
    config: RealActivationSweepConfig,
    spec: RealActivationRunSpec,
    step: int,
    n_training_tokens: int,
    loss: float,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save one final Stage-3 native-training checkpoint."""

    config.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "data_kind": REAL_ACTIVATION_DATA_KIND,
        "checkpoint_kind": "last",
        "step": step,
        "n_training_tokens": n_training_tokens,
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
    """Restore a native Stage-3 training checkpoint through the SAE registry."""

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if (
        payload.get("format_version") != 1
        or payload.get("data_kind") != REAL_ACTIVATION_DATA_KIND
        or payload.get("checkpoint_kind") != "last"
    ):
        raise ValueError("Not a supported Stage-3 final checkpoint.")
    architecture = payload["model_config"]["architecture"]
    model_class, config_class = get_sae_training_class(architecture)
    model_config = config_class.from_dict(payload["model_config"])
    model_config.device = str(device)
    model = model_class(model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


__all__ = [
    "CONTROL_GRID_PROVENANCE",
    "CONTROL_GRID_RATIONALE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CONTEXT_SIZE",
    "DEFAULT_DOWNSTREAM_EVAL_TOKENS",
    "DEFAULT_EVAL_TOKENS",
    "DEFAULT_STAGE3_CONTROLS",
    "DEFAULT_TARGET_NAME",
    "DEFAULT_TRAIN_TOKENS",
    "GEMMA_LAYER12_BATCHTOPK_K_GRID",
    "GEMMA_LAYER12_DATASET_ID",
    "GEMMA_LAYER12_DATASET_REVISION",
    "GEMMA_LAYER12_PAPER_REPORTED_TRAIN_TOKENS",
    "GEMMA_LAYER12_TRAIN_TOKENS",
    "GEMMA_LAYER5_BATCHTOPK_K_GRID",
    "GEMMA_MODEL_ID",
    "GEMMA_MODEL_REVISION",
    "GEMMA_DATASET_ID",
    "GEMMA_DATASET_REVISION",
    "JUMPRELU_WARMUP_TOKENS",
    "LLAMA_MODEL_ID",
    "LLAMA_MODEL_REVISION",
    "LLAMA_DATASET_ID",
    "LLAMA_DATASET_REVISION",
    "MODEL_CLASS_NAME",
    "PAPER_BATCHTOPK_K_GRID",
    "PAPER_BATCHTOPK_K_GRIDS",
    "PAPER_JUMPRELU_COEFFICIENT_GRID",
    "PAPER_REPORTED_TRAIN_TOKENS",
    "PAPER_TARGET_DENSITIES",
    "REAL_ACTIVATION_DATA_KIND",
    "REAL_MODEL_TARGETS",
    "RealActivationDataConfig",
    "RealActivationRunSpec",
    "RealActivationSweepConfig",
    "RealActivationTarget",
    "RealActivationTrainingConfig",
    "SAE_WIDTH",
    "SAELENS_REVISION",
    "SPARSE_BUT_WRONG_REVISION",
    "SOURCE_DATASET_ID",
    "SOURCE_DATASET_REVISION",
    "STAGE2_DENSITY_ANCHORS",
    "STAGE2_L1_CONTROL_ANCHORS",
    "STAGE2_VG_CONTROL_ANCHORS",
    "STAGE3_CONTROL_NAMES",
    "STAGE3_METHOD_LABELS",
    "STAGE3_METHOD_ORDER",
    "TRAINING_BUDGET_RATIONALE",
    "augment_stage3_runtime_provenance",
    "build_model",
    "build_specs",
    "default_sweep_config",
    "default_sweep_configs",
    "default_sweep_dir",
    "default_controls_for_target",
    "interpolate_stage2_control",
    "load_checkpoint",
    "method_learning_rate",
    "run_directory",
    "save_checkpoint",
    "sweep_experiment_id",
    "target_data_config",
]
