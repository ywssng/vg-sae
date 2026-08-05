"""Exact aliases to the pinned official SAELens training SAEs."""

from __future__ import annotations

from typing import Any

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.gated_sae import GatedTrainingSAE, GatedTrainingSAEConfig
from sae_lens.saes.jumprelu_sae import (
    JumpReLU,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    Step,
)
from sae_lens.saes.sae import SAE, TrainingSAE
from sae_lens.saes.standard_sae import (
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

# Public baseline names are identities, not wrappers or reimplementations.
StandardSAE = StandardTrainingSAE
StandardSAEConfig = StandardTrainingSAEConfig
TopKSAE = TopKTrainingSAE
TopKSAEConfig = TopKTrainingSAEConfig
BatchTopKSAE = BatchTopKTrainingSAE
BatchTopKSAEConfig = BatchTopKTrainingSAEConfig
JumpReLUSAE = JumpReLUTrainingSAE
JumpReLUSAEConfig = JumpReLUTrainingSAEConfig
GatedSAE = GatedTrainingSAE
GatedSAEConfig = GatedTrainingSAEConfig


def to_inference_sae(
    model: TrainingSAE[Any], *, fold_decoder_norm: bool = False
) -> SAE[Any]:
    """Convert with official config/state hooks, then optionally fold decoder norms."""

    state_dict = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.process_state_dict_for_saving_inference(state_dict)
    inference = SAE.from_dict(model.cfg.get_inference_sae_cfg_dict())
    inference.load_state_dict(state_dict)
    inference.eval()
    if fold_decoder_norm:
        inference.fold_W_dec_norm()
    return inference

__all__ = [
    "BatchTopKSAE",
    "BatchTopKSAEConfig",
    "BatchTopKTrainingSAE",
    "BatchTopKTrainingSAEConfig",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedTrainingSAE",
    "GatedTrainingSAEConfig",
    "JumpReLU",
    "JumpReLUSAE",
    "JumpReLUSAEConfig",
    "JumpReLUTrainingSAE",
    "JumpReLUTrainingSAEConfig",
    "Step",
    "StandardSAE",
    "StandardSAEConfig",
    "StandardTrainingSAE",
    "StandardTrainingSAEConfig",
    "TopKSAE",
    "TopKSAEConfig",
    "TopKTrainingSAE",
    "TopKTrainingSAEConfig",
    "TrainingSAE",
    "to_inference_sae",
]
