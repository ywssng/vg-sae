"""SAELens v6.47 adapter for the local Variational Garrote SAE."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
import torch.nn as nn
from typing_extensions import override

from sae_lens import (
    SAE,
    SAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
    __version__ as saelens_version,
    register_sae_class,
    register_sae_training_class,
)
from sae_lens.registry import get_sae_class, get_sae_training_class
from sae_lens.saes.sae import (
    TrainCoefficientConfig,
    TrainStepInput,
    TrainStepOutput,
)
from sae_lens.training.sae_trainer import SAETrainer

from .sae_model import VGSAEConfig as CoreVGSAEConfig
from .sae_model import VariationalGarroteSAE


SAELENS_VERSION = "6.47.0"
ARCHITECTURE = "vg"
BetaMode = Literal["profiled", "fixed", "learned"]

if saelens_version != SAELENS_VERSION:
    raise RuntimeError(
        f"src.saelens_vg requires sae-lens=={SAELENS_VERSION}; found {saelens_version}."
    )


@dataclass
class VGSAEConfig(SAEConfig):
    """Inference configuration for a thresholded VG posterior."""

    beta: float = 1.0
    lambda_sparsity: float = 1.0
    use_variance_term: bool = True
    use_entropy_term: bool = True
    entropy_weight: float = 1.0
    beta_mode: BetaMode = "profiled"
    decoder_bias: bool = True
    tie_encoder_init: bool = True
    gate_bias_init: float = -2.0
    amplitude_bias_init: float = 0.0
    nonnegative_amplitudes: bool = True
    normalize_decoder: bool = True
    loss_eps: float = 1.0e-8
    inference_threshold: float = 0.5

    @override
    @classmethod
    def architecture(cls) -> str:
        return ARCHITECTURE

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_config(self)


@dataclass(kw_only=True)
class VGTrainingSAEConfig(TrainingSAEConfig):
    """Training configuration; VG coefficients are recorded in inference metadata."""

    beta: float = 1.0
    lambda_sparsity: float = 1.0
    lambda_warm_up_steps: int = 0
    use_variance_term: bool = True
    use_entropy_term: bool = True
    entropy_weight: float = 1.0
    beta_mode: BetaMode = "profiled"
    decoder_bias: bool = True
    tie_encoder_init: bool = True
    gate_bias_init: float = -2.0
    amplitude_bias_init: float = 0.0
    nonnegative_amplitudes: bool = True
    normalize_decoder: bool = True
    loss_eps: float = 1.0e-8
    inference_threshold: float = 0.5
    decoder_init_norm: float | None = None

    @override
    @classmethod
    def architecture(cls) -> str:
        return ARCHITECTURE

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return VGSAEConfig

    @override
    def __post_init__(self) -> None:
        if (
            isinstance(self.lambda_warm_up_steps, bool)
            or not isinstance(self.lambda_warm_up_steps, int)
            or self.lambda_warm_up_steps < 0
        ):
            raise ValueError("lambda_warm_up_steps must be a non-negative integer.")
        if self.decoder_init_norm is not None:
            raise ValueError("VG-SAE initialization is controlled by normalize_decoder.")
        super().__post_init__()
        _validate_config(self)


VGConfig = VGSAEConfig | VGTrainingSAEConfig


def _core_config(cfg: VGConfig) -> CoreVGSAEConfig:
    return CoreVGSAEConfig(
        input_dim=cfg.d_in,
        n_latents=cfg.d_sae,
        beta=cfg.beta,
        lambda_sparsity=cfg.lambda_sparsity,
        use_variance_term=cfg.use_variance_term,
        use_entropy_term=cfg.use_entropy_term,
        entropy_weight=cfg.entropy_weight,
        beta_mode=cfg.beta_mode,
        trace_beta=None,
        decoder_bias=cfg.decoder_bias,
        tie_encoder_init=cfg.tie_encoder_init,
        gate_bias_init=cfg.gate_bias_init,
        amplitude_bias_init=cfg.amplitude_bias_init,
        nonnegative_amplitudes=cfg.nonnegative_amplitudes,
        normalize_decoder=cfg.normalize_decoder,
        loss_eps=cfg.loss_eps,
        inference_threshold=cfg.inference_threshold,
        dtype=cfg.dtype,
    )


def _validate_config(cfg: VGConfig) -> None:
    _core_config(cfg).validate()
    if not cfg.nonnegative_amplitudes:
        raise ValueError("SAELens firing metrics require nonnegative VG amplitudes.")
    if cfg.decoder_bias and not cfg.apply_b_dec_to_input:
        raise ValueError("VG-SAE requires centered inputs when decoder_bias=True.")
    if cfg.normalize_activations not in {"none", "expected_average_only_in"}:
        raise ValueError("VG-SAE supports 'none' or 'expected_average_only_in' normalization.")


class _VGCoreMixin:
    """Expose the local model through the parameter and hook conventions of SAELens."""

    cfg: VGConfig
    core: VariationalGarroteSAE
    dtype: torch.dtype
    device: torch.device

    @override
    def get_activation_fn(self) -> nn.Module:
        return nn.Identity()

    @override
    def initialize_weights(self) -> None:
        with torch.device(self.device):
            self.core = VariationalGarroteSAE(_core_config(self.cfg))
            if self.core.pre_bias is None:
                self.register_buffer(
                    "_zero_b_dec",
                    torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device),
                )

    @property
    def W_enc(self) -> torch.Tensor:
        """Amplitude encoder in SAELens' ``(d_in, d_sae)`` orientation."""
        return self.core.amplitude_encoder.weight.T

    @property
    def W_dec(self) -> torch.Tensor:
        """Decoder atoms in SAELens' ``(d_sae, d_in)`` orientation."""
        return self.core.decoder.weight.T

    @property
    def b_dec(self) -> torch.Tensor:
        return self.core.pre_bias if self.core.pre_bias is not None else self._zero_b_dec

    def _prepare_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        self.normalize_decoder_columns()
        x = self.run_time_activation_norm_fn_in(
            self.hook_sae_input(self.reshape_fn_in(x.to(self.dtype)))
        )
        return x.reshape(-1, self.cfg.d_in), x.shape[:-1]

    def posterior(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return posterior probabilities, amplitudes, and their expected code."""
        flat, leading_shape = self._prepare_input(x)
        gate_logits, m, a, expected = self.core._encode_centered(  # noqa: SLF001
            self.core._center(flat)  # noqa: SLF001
        )
        shape = (*leading_shape, self.cfg.d_sae)
        return {
            "m": m.reshape(shape),
            "a": a.reshape(shape),
            "expected_code": expected.reshape(shape),
            "gate_logits": gate_logits.reshape(shape),
        }

    def support_mask(
        self, x: torch.Tensor, threshold: float | None = None
    ) -> torch.Tensor:
        tau = self.cfg.inference_threshold if threshold is None else float(threshold)
        if not 0.0 <= tau <= 1.0:
            raise ValueError("threshold must lie in [0, 1].")
        return self.posterior(x)["m"] > tau

    def encode_hard(
        self, x: torch.Tensor, threshold: float | None = None
    ) -> torch.Tensor:
        posterior = self.posterior(x)
        tau = self.cfg.inference_threshold if threshold is None else float(threshold)
        if not 0.0 <= tau <= 1.0:
            raise ValueError("threshold must lie in [0, 1].")
        return posterior["a"] * (posterior["m"] > tau).to(posterior["a"])

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        sae_out = self.hook_sae_recons(self.core.decode(feature_acts))
        sae_out = self.run_time_activation_norm_fn_out(sae_out)
        return self.reshape_fn_out(sae_out, self.d_head)

    @override
    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float) -> None:
        if not math.isfinite(scaling_factor) or scaling_factor <= 0.0:
            raise ValueError("scaling_factor must be positive and finite.")
        self.core.gate_encoder.weight.mul_(scaling_factor)
        self.core.amplitude_encoder.weight.mul_(scaling_factor)
        self.core.decoder.weight.div_(scaling_factor)
        if self.core.pre_bias is not None:
            self.core.pre_bias.div_(scaling_factor)
        if self.cfg.beta_mode == "learned":
            assert self.core.log_beta is not None
            self.core.log_beta.add_(2.0 * math.log(scaling_factor))
        elif self.cfg.beta_mode == "fixed":
            self.cfg.beta *= scaling_factor**2
            self.core.config.beta = self.cfg.beta
        self.cfg.metadata.decoder_normalized_during_training = self.cfg.normalize_decoder
        self.cfg.normalize_decoder = False
        self.core.config.normalize_decoder = False
        decoder_hook = getattr(self, "_decoder_grad_hook", None)
        if decoder_hook is not None:
            decoder_hook.remove()
            self._decoder_grad_hook = None
        self.cfg.normalize_activations = "none"

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        norms = self.W_dec.norm(dim=-1)
        unit = torch.ones_like(norms)
        if self.cfg.normalize_decoder and not torch.allclose(
            norms, unit, rtol=1.0e-6, atol=1.0e-6
        ):
            self.core.normalize_decoder_columns()

    @torch.no_grad()
    def remove_decoder_parallel_grad(self) -> None:
        if self.cfg.normalize_decoder:
            self.core.remove_decoder_parallel_grad()

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        if not torch.allclose(
            self.W_dec.norm(dim=-1),
            torch.ones(self.cfg.d_sae, device=self.device, dtype=self.dtype),
        ):
            raise NotImplementedError(
                "VG decoder norms cannot be folded through the nonlinear amplitude encoder."
            )


class VGSAE(_VGCoreMixin, SAE[VGSAEConfig]):
    """Inference VG-SAE whose public code has explicit hard support."""

    def __init__(self, cfg: VGSAEConfig, use_error_term: bool = False) -> None:
        if use_error_term:
            raise ValueError("VG-SAE does not support use_error_term=True.")
        super().__init__(cfg, use_error_term=False)

    @override
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.hook_sae_acts_post(self.encode_hard(x))


class VGTrainingSAE(_VGCoreMixin, TrainingSAE[VGTrainingSAEConfig]):
    """Train with the expected code while exposing a hard code to SAELens.

    ``TrainStepOutput.sae_out`` is the variational expected reconstruction, while
    ``feature_acts`` is the hard posterior code used by SAELens firing metrics.
    The free-energy path is intentionally supported only without training-time
    activation or reconstruction intervention hooks; public inference encode/decode
    hooks remain available.
    """

    def __init__(
        self, cfg: VGTrainingSAEConfig, use_error_term: bool = False
    ) -> None:
        if use_error_term:
            raise ValueError("VG-SAE does not support use_error_term=True.")
        super().__init__(cfg, use_error_term=False)
        self._install_decoder_gradient_hook()

    def _install_decoder_gradient_hook(self) -> None:
        old_hook = getattr(self, "_decoder_grad_hook", None)
        if old_hook is not None:
            old_hook.remove()
        self._decoder_grad_hook = None
        if not self.cfg.normalize_decoder:
            return
        decoder = self.core.decoder.weight

        def tangent_gradient(gradient: torch.Tensor) -> torch.Tensor:
            weight = decoder.detach()
            norm_sq = weight.pow(2).sum(dim=0, keepdim=True).clamp_min(self.cfg.loss_eps)
            projection = (gradient * weight).sum(dim=0, keepdim=True) / norm_sq
            return gradient - projection * weight

        self._decoder_grad_hook = decoder.register_hook(tangent_gradient)

    @override
    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._install_decoder_gradient_hook()
        return result

    @override
    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        self.normalize_decoder_columns()
        state_dict["core.decoder.weight"] = (
            self.core.decoder.weight.detach().clone().contiguous()
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        posterior = self.posterior(x)
        hard = posterior["a"] * (posterior["m"] > self.cfg.inference_threshold).to(
            posterior["a"]
        )
        return self.hook_sae_acts_post(hard), posterior["gate_logits"]

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "lambda_sparsity": TrainCoefficientConfig(
                self.cfg.lambda_sparsity, self.cfg.lambda_warm_up_steps
            )
        }

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del step_input, feature_acts, hidden_pre, sae_out
        return {}

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        flat, _ = self._prepare_input(step_input.sae_in)
        output = self.core.free_energy(
            flat,
            lambda_sparsity=step_input.coefficients["lambda_sparsity"],
        )
        entropy_weight = self.cfg.entropy_weight if self.cfg.use_entropy_term else 0.0
        m = output["m"]
        hard = output["a"] * (m > self.cfg.inference_threshold).to(output["a"])
        hard_l0 = hard.gt(0).sum(dim=-1).float().mean()
        decoder_error = (self.W_dec.norm(dim=-1) - 1.0).abs().max()
        return TrainStepOutput(
            sae_in=flat,
            sae_out=output["x_hat"],
            feature_acts=hard,
            hidden_pre=output["gate_logits"],
            loss=output["loss"],
            losses={
                "free_energy": output["loss"],
                "expected_reconstruction_energy": output["recon"],
                "posterior_variance_energy": output["variance"],
                "posterior_prior_energy": output["prior"],
                "posterior_negative_entropy_energy": (
                    -entropy_weight * output["entropy"]
                ),
            },
            metrics={
                "posterior_expected_l0": output["sparsity"],
                "hard_support_l0": hard_l0,
                "posterior_mean_probability": m.mean(),
                "posterior_bernoulli_variance": (m * (1.0 - m)).mean(),
                "beta_precision": output["beta_eff"],
                "decoder_norm_error": decoder_error,
            },
        )

    def firing_mask(self, output: TrainStepOutput) -> torch.Tensor:
        """Hard posterior support used for SAELens firing and dead-feature stats."""
        return torch.sigmoid(output.hidden_pre) > self.cfg.inference_threshold


class VGSAETrainer(SAETrainer[VGTrainingSAE, VGTrainingSAEConfig]):
    """Add the VG decoder constraint and explicit hard/expected log semantics."""

    @override
    def step(self, batch: torch.Tensor) -> TrainStepOutput:
        output = super().step(batch)
        self.sae.normalize_decoder_columns()
        return output

    @override
    @torch.no_grad()
    def build_train_step_log_dict(
        self, output: TrainStepOutput, n_training_samples: int
    ) -> dict[str, Any]:
        log = super().build_train_step_log_dict(output, n_training_samples)
        renames = {
            "metrics/l0": "metrics/hard_support_l0",
            "metrics/explained_variance": (
                "metrics/expected_reconstruction_explained_variance"
            ),
            "metrics/explained_variance_legacy": (
                "metrics/expected_reconstruction_explained_variance_legacy"
            ),
            "metrics/explained_variance_legacy_std": (
                "metrics/expected_reconstruction_explained_variance_legacy_std"
            ),
        }
        for old, new in renames.items():
            log[new] = log.pop(old)
        return log


def register_vg_saes() -> None:
    """Register the custom inference and training classes, safely and idempotently."""

    def register_one(getter: Any, register: Any, cls: type[Any], cfg: type[Any]) -> None:
        try:
            registered = getter(ARCHITECTURE)
        except KeyError:
            register(ARCHITECTURE, cls, cfg)
        else:
            if registered != (cls, cfg):
                raise ValueError(
                    f"SAELens architecture {ARCHITECTURE!r} is already registered."
                )

    register_one(get_sae_class, register_sae_class, VGSAE, VGSAEConfig)
    register_one(
        get_sae_training_class,
        register_sae_training_class,
        VGTrainingSAE,
        VGTrainingSAEConfig,
    )


register_vg_saes()

__all__ = [
    "ARCHITECTURE",
    "SAELENS_VERSION",
    "VGSAE",
    "VGSAEConfig",
    "VGSAETrainer",
    "VGTrainingSAE",
    "VGTrainingSAEConfig",
    "register_vg_saes",
]
