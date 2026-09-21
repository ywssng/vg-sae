"""Paper-inspired data and feature-identity metrics for bounded VG-SAE pilots.

The copula correlation is a latent Gaussian correlation, not the Pearson
correlation of the resulting Bernoulli support. No experiment runs on import.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class ToyBatch:
    x: torch.Tensor
    z: torch.Tensor
    support: torch.Tensor
    dictionary: torch.Tensor


def orthogonal_dictionary(seed: int, input_dim: int = 20, width: int = 5) -> torch.Tensor:
    if not 1 <= width <= input_dim:
        raise ValueError("Require 1 <= width <= input_dim for the orthogonal toy.")
    gen = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(input_dim, width, generator=gen, dtype=torch.float64))
    signs = r.diag().sign()
    signs[signs == 0] = 1
    return (q * signs).float()


def hub_correlation(width: int, correlation: float) -> torch.Tensor:
    if width < 2:
        raise ValueError("The correlated toy requires at least two features.")
    result = torch.eye(width, dtype=torch.float64)
    result[0, 1:] = correlation
    result[1:, 0] = correlation
    if torch.linalg.eigvalsh(result).min() <= 0:
        raise ValueError("Hub Gaussian correlation must be positive definite.")
    return result


def sample_toy(dictionary: torch.Tensor, count: int, seed: int,
               correlation: float = 0.0, amplitude: str = "narrow_normal",
               firing_probability: float = 0.4, device: str = "cpu") -> ToyBatch:
    if count < 1 or not 0 < firing_probability < 1:
        raise ValueError("Positive sample count and probability strictly in (0,1) required.")
    width = dictionary.shape[1]
    chol = torch.linalg.cholesky(hub_correlation(width, correlation))
    support_gen = torch.Generator().manual_seed(seed)
    amp_gen = torch.Generator().manual_seed(seed + 10_000_019)
    normal = torch.randn(count, width, generator=support_gen, dtype=torch.float64) @ chol.T
    quantile = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(firing_probability, dtype=torch.float64))
    support = normal < quantile
    if amplitude == "narrow_normal":
        active = (1 + 0.15 * torch.randn(count, width, generator=amp_gen, dtype=torch.float64)).clamp_min(0)
    elif amplitude == "exponential":
        # Match the intended active second moment 1 + .15**2; negative normal
        # clipping has only a negligible, explicitly non-exact effect on it.
        scale = ((1 + 0.15**2) / 2)**0.5
        uniform = torch.rand(count, width, generator=amp_gen, dtype=torch.float64)
        active = -torch.log1p(-uniform) * scale
    else:
        raise ValueError("Unknown amplitude distribution.")
    z = (support * active).float()
    support = z > 0
    d = dictionary.detach().cpu().float()
    return ToyBatch((z @ d.T).to(device), z.to(device), support.to(device), d.to(device))


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def geometry_metrics(decoder: torch.Tensor, truth: torch.Tensor) -> tuple[dict, tuple]:
    """Bijective matching for equal-width toys; retain signs and mixing energy."""
    if decoder.shape != truth.shape:
        raise ValueError("This pilot metric requires equal learned/true dictionary shapes.")
    learned = torch.nn.functional.normalize(decoder.double(), dim=0)
    true = torch.nn.functional.normalize(truth.double(), dim=0)
    gram = true.T @ true
    if not torch.allclose(gram, torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype), atol=1e-6):
        raise ValueError("Off-diagonal energy decomposition requires orthonormal truth.")
    cosine = learned.T @ true
    # Positive coefficients do not make atom sign flips an equivalent truth.
    # Primary alignment maximizes signed cosine; absolute alignment is secondary.
    li, ti = linear_sum_assignment(-cosine.detach().cpu().numpy())
    abs_li, abs_ti = linear_sum_assignment(-cosine.abs().detach().cpu().numpy())
    signs = torch.ones(len(li), device=decoder.device, dtype=decoder.dtype)
    matched = cosine[li, ti]
    projected_energy = cosine.square().sum(1)
    mixing = (projected_energy[li] - matched.square()).clamp_min(0)
    width = decoder.shape[1]
    pairwise = learned.T @ learned
    mask = ~torch.eye(width, device=decoder.device, dtype=torch.bool)
    signed_off = cosine[li].clone()
    signed_off[torch.arange(width, device=decoder.device), torch.as_tensor(ti, device=decoder.device)] = 0
    result = {
        "matched_abs_cosine": float(cosine[abs_li, abs_ti].abs().mean()),
        "matched_positive_cosine": float(matched.mean()),
        "recovered_fraction_cosine_095": float((matched >= .95).double().mean()),
        "mixing_energy": float(mixing.mean()),
        "off_manifold_energy": float((1 - projected_energy).clamp_min(0).mean()),
        "negative_mixing_energy": float(signed_off.clamp_max(0).square().sum(1).mean()),
        "c_dec": float(pairwise[mask].abs().mean()),
        "decoder_norm_min": float(decoder.norm(dim=0).min()),
        "signed_hub_leakage": float(cosine[li[ti != 0], 0].mean()),
    }
    return result, (li, ti, signs)


@torch.no_grad()
def evaluate_model(model, data: ToyBatch) -> dict:
    from src.sae_model import VariationalGarroteSAE
    from src.sae_baselines import TrainingSAE, to_inference_sae

    extra = {}
    if isinstance(model, VariationalGarroteSAE):
        m, amplitude, mean_code = model.encode(data.x)
        code = amplitude * (m > .5)
        reconstruction = model.decode(code)
        decoder = model.decoder.weight
        terms = model.free_energy(data.x)
        mean_mse = (model.decode(mean_code)-data.x).square().mean()
        extra = {"expected_l0": float(m.sum(1).mean()),
                 "mean_mse": float(mean_mse),
                 "sampled_expected_mse": float(mean_mse + 2*terms["variance"]/data.x.shape[1]),
                 "mean_gate_entropy": float(terms["entropy"] / m.shape[1]),
                 "beta_eff_on_evaluation": float(terms["beta_eff"])}
    elif isinstance(model, TrainingSAE):
        inference = to_inference_sae(model, fold_decoder_norm=True).to(data.x.device)
        code = inference.encode(data.x)
        reconstruction = inference.decode(code)
        decoder = inference.W_dec.T
    else:
        raise TypeError(type(model))
    geometry, (li, ti, signs) = geometry_metrics(decoder, data.dictionary)
    aligned = torch.zeros_like(data.z)
    aligned[:, ti] = code[:, li] * signs.to(code.dtype)
    support = code > 0
    tp = (support[:, li] & data.support[:, ti]).sum()
    residual = (reconstruction-data.x).square().sum(1).mean()
    variance = (data.x-data.x.mean(0)).square().sum(1).mean().clamp_min(1e-12)
    l0 = support.sum(1).float()
    true_l0 = data.support.sum(1).float()
    result = {**geometry, **extra,
              "hard_latent_relative_error": float(((aligned-data.z).square().sum()/data.z.square().sum().clamp_min(1e-12)).sqrt()),
              "support_f1": float(2*tp/(support.sum()+data.support.sum()).clamp_min(1)),
              "hard_l0": float(l0.mean()), "true_l0": float(true_l0.mean()),
              "l0_absolute_sample_error": float((l0-true_l0).abs().mean()),
              "hard_mse": float(residual/data.x.shape[1]), "hard_ev": float(1-residual/variance),
              "active_latent_fraction": float(support.any(0).float().mean()),
              "actual_l0_threshold": "strictly positive native code; VG uses m > 0.5"}
    return result


def train_shared_batches(model, x: torch.Tensor, *, steps: int, seed: int,
                         lr: float = .003, batch_size: int = 256,
                         checkpoints: tuple[int, ...] = (), snapshot=None,
                         before_step=None, after_step=None) -> dict:
    """Reuse native VG/official baseline optimizers with the same batch stream.

    No activation-statistics pass is allowed to consume one method's stream.
    Checkpoint indices count completed updates, not zero-based loop indices.
    """
    import time
    from src.sae_train import _CyclingTensorBatches
    from src.sae_baselines import TrainingSAE
    from src.sae_model import VariationalGarroteSAE
    from sae_lens.config import LoggingConfig, SAETrainerConfig
    from sae_lens.training.sae_trainer import SAETrainer

    if steps < 1 or x.shape[0] < batch_size or x.shape[0] % batch_size:
        raise ValueError("Use a positive budget and train size divisible by batch size.")
    provider = _CyclingTensorBatches(x, batch_size, seed)
    official = isinstance(model, TrainingSAE)
    if official:
        if model.cfg.normalize_activations != "none":
            raise ValueError("Shared-stream pilot requires explicit no input rescaling.")
        trainer = SAETrainer(
            cfg=SAETrainerConfig(total_training_samples=steps*batch_size,
                train_batch_size_samples=batch_size, lr=lr, lr_end=lr,
                lr_scheduler_name="constant", device=str(x.device),
                dead_feature_window=1000, logger=LoggingConfig(log_to_wandb=False)),
            sae=model, data_provider=provider)
    else:
        if not isinstance(model, VariationalGarroteSAE):
            raise TypeError(type(model))
        gamma = getattr(model, "gamma", None)
        if isinstance(gamma, torch.nn.Parameter):
            groups = [{"params": [p for p in model.parameters() if p is not gamma]},
                      {"params": [gamma], "lr": lr}]
            optimizer = torch.optim.Adam(groups, lr=lr, weight_decay=0)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    started = time.perf_counter()
    model.train()
    first_batch_hash = None
    for index in range(steps):
        if before_step is not None:
            before_step(model, index)
        batch = next(provider)
        if index == 0:
            first_batch_hash = tensor_hash(batch)
        if official:
            trainer.maybe_reset_sparsity()
            trainer.step(batch)
            trainer.n_training_steps += 1
        else:
            optimizer.zero_grad(set_to_none=True)
            terms = model.free_energy(batch)
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError(f"Nonfinite training loss at update {index+1}")
            terms["loss"].backward()
            if model.config.normalize_decoder:
                model.remove_decoder_parallel_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if model.config.normalize_decoder:
                model.normalize_decoder_columns()
        if after_step is not None:
            after_step(model, index+1)
        if index+1 in checkpoints and snapshot is not None:
            model.eval()
            snapshot(model, index+1)
            model.train()
    if official:
        trainer.set_final_sae_metadata()
    model.eval()
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    return {"completed_updates": steps, "sample_presentations": steps*batch_size,
            "first_batch_sha256": first_batch_hash,
            "training_and_snapshot_seconds": time.perf_counter()-started}
