"""Train the fixed-generator Stage-2 SynthSAEBench method/control sweep."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from sae_lens import LoggingConfig
from sae_lens.config import SAETrainerConfig
from sae_lens.synthetic import SyntheticActivationIterator
from sae_lens.training.sae_trainer import SAETrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs._sweep_io import (  # noqa: E402
    aggregate_csv,
    fingerprint,
    read_json,
    resolve_devices,
    runtime_provenance,
    utc_now,
    write_json,
    write_rows,
)
from runs.gpu_scheduler import ParallelExecutor, ScriptTask  # noqa: E402
from src.sae_sweep import CONTROL_NAMES, METHOD_LABELS, METHOD_ORDER  # noqa: E402
from src.saelens_vg import VGSAETrainer  # noqa: E402
from src.synthsaebench_sweep import (  # noqa: E402
    SAELENS_REVISION,
    SynthSAEBenchRunSpec,
    SynthSAEBenchSweepConfig,
    build_model,
    build_specs,
    capture_rng_state,
    default_sweep_config,
    default_sweep_dir,
    load_benchmark_model,
    run_directory,
    save_checkpoint,
    sweep_experiment_id,
    temporary_seed_for_device,
)


WANDB_PROJECT = "vg-sae"
TRAIN_SOURCE_FILES = (
    "runs/_sweep_io.py",
    "runs/gpu_scheduler.py",
    "runs/run_SynthSAEBench_sweep.py",
    "src/model.py",
    "src/sae_baselines.py",
    "src/sae_model.py",
    "src/sae_sweep.py",
    "src/saelens_vg.py",
    "src/synthsaebench_sweep.py",
)


def _load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Serialized SynthSAEBench config JSON.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument(
        "--calibration-grid",
        action="store_true",
        help="Use broad range-scout controls instead of 200M-calibrated final controls.",
    )
    parser.add_argument("--methods", default="all", help="Comma-separated methods, or all.")
    seed = parser.add_mutually_exclusive_group()
    seed.add_argument("--seed", type=int, help="Run one initialization/stream seed.")
    seed.add_argument("--seeds", help="Comma-separated seed sweep.")
    parser.add_argument(
        "--model-sparsity-control",
        "--sparsity-control",
        dest="sparsity_controls",
        action="append",
        metavar="METHOD=V1,V2,...",
        help="Replace one method's direct sparsity-control grid; repeat per method.",
    )
    parser.add_argument("--training-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--history-every", type=int)
    parser.add_argument(
        "--resume-every",
        type=int,
        help="Write one rolling trainer-state checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--lr-decay-fraction",
        type=float,
        help=(
            "Final fraction with linear LR decay. Default 0 follows released runner "
            "configs; use about 0.333333 for the paper-described schedule."
        ),
    )
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:2,cuda:3",
        help="auto, cpu, or cuda:0,cuda:1,...",
    )
    parser.add_argument("--max-per-device", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs.")
    logging = parser.add_mutually_exclusive_group()
    logging.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    logging.add_argument(
        "--no-wandb", action="store_const", const="disabled", dest="wandb_mode"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    return parser.parse_args()


def _control_overrides(entries: list[str]) -> dict[str, list[float | int]]:
    overrides: dict[str, list[float | int]] = {}
    for entry in entries:
        method, separator, raw_values = entry.partition("=")
        method = method.strip().lower()
        if not separator or method not in CONTROL_NAMES:
            raise ValueError(
                f"Invalid sparsity control {entry!r}; expected METHOD=v1,v2,..."
            )
        if method in overrides:
            raise ValueError(f"Duplicate sparsity-control override for {method!r}.")
        tokens = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not tokens:
            raise ValueError(f"Sparsity-control grid for {method!r} is empty.")
        if method == "topk":
            floats = [float(value) for value in tokens]
            if any(not value.is_integer() for value in floats):
                raise ValueError("TopK controls must be integers.")
            overrides[method] = [int(value) for value in floats]
        else:
            overrides[method] = [float(value) for value in tokens]
    return overrides


def _one_eighth_train(train_samples: int, batch_size: int) -> int:
    target = max(1, train_samples // 8)
    if target >= batch_size:
        target = max(batch_size, target // batch_size * batch_size)
    return target


def configured_sweep(args: argparse.Namespace) -> SynthSAEBenchSweepConfig:
    base = (
        read_json(args.config)
        if args.config is not None
        else default_sweep_config(
            fast=args.fast_dev_run,
            calibration=getattr(args, "calibration_grid", False),
        ).to_dict()
    )
    raw = SynthSAEBenchSweepConfig.from_dict(base).to_dict()
    seed = getattr(args, "seed", None)
    seeds = getattr(args, "seeds", None)
    if seed is not None:
        raw["seeds"] = [seed]
    elif seeds:
        raw["seeds"] = [int(value) for value in seeds.split(",")]

    overrides = _control_overrides(getattr(args, "sparsity_controls", None) or [])
    raw["controls"].update(overrides)
    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None:
        raw["training"]["batch_size"] = batch_size
    training_samples = getattr(args, "training_samples", None)
    test_samples = getattr(args, "test_samples", None)
    if training_samples is not None:
        raw["data"]["n_train"] = training_samples
        if test_samples is None:
            raw["data"]["n_test"] = _one_eighth_train(
                training_samples, raw["training"]["batch_size"]
            )
    if test_samples is not None:
        raw["data"]["n_test"] = test_samples
    history_every = getattr(args, "history_every", None)
    if history_every is not None:
        raw["training"]["history_every"] = history_every
    resume_every = getattr(args, "resume_every", None)
    if resume_every is not None:
        raw["training"]["resume_every"] = resume_every
    lr_decay_fraction = getattr(args, "lr_decay_fraction", None)
    if lr_decay_fraction is not None:
        raw["training"]["lr_decay_fraction"] = lr_decay_fraction
    return SynthSAEBenchSweepConfig.from_dict(raw)


def selected_specs(
    config: SynthSAEBenchSweepConfig, methods: str
) -> list[SynthSAEBenchRunSpec]:
    specs = build_specs(config)
    if methods == "all":
        return specs
    requested = {value.strip().lower() for value in methods.split(",") if value.strip()}
    if not requested:
        raise ValueError("--methods must name at least one method or use 'all'.")
    if unknown := requested - set(METHOD_ORDER):
        raise ValueError(f"Unknown method(s): {', '.join(sorted(unknown))}")
    selected = [spec for spec in specs if spec.method in requested]
    present = {spec.method for spec in selected}
    if missing := requested - present:
        raise ValueError(
            "Methods are absent from the configured sweep: "
            f"{', '.join(sorted(missing))}"
        )
    return selected


def _bundle(
    config: SynthSAEBenchSweepConfig,
    spec: SynthSAEBenchRunSpec,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "format_version": 1,
        "sweep_config": config.to_dict(),
        "spec": spec.to_dict(),
        "checkpoint_selection": "final streaming-training state",
        "train_provenance": provenance,
        "benchmark_identity": {
            "model_id": config.data.model_id,
            "revision": config.data.revision,
            "model_config_sha256": config.data.model_config_sha256,
            "saelens_revision": SAELENS_REVISION,
        },
    }
    fingerprint_payload = {
        **payload,
        "train_provenance": provenance["pipeline_fingerprint"],
    }
    payload["fingerprint"] = fingerprint(fingerprint_payload)
    return payload


def _is_complete(run_dir: Path, run_fingerprint: str) -> bool:
    status_path = run_dir / "train_status.json"
    if not status_path.exists():
        return False
    status = read_json(status_path)
    return (
        status.get("state") == "complete"
        and status.get("fingerprint") == run_fingerprint
        and (run_dir / "checkpoints" / "last.pt").exists()
        and (run_dir / "training_history.csv").exists()
        and (run_dir / "training_summary.json").exists()
    )


def _preflight_wandb(mode: str) -> None:
    if mode in {"disabled", "offline"}:
        return
    import wandb

    try:
        authenticated = wandb.login(verify=True, force=True)
    except Exception:
        raise RuntimeError(
            "W&B authentication preflight failed; check WANDB_API_KEY in .env, "
            "the shell environment, or your existing W&B login."
        ) from None
    if not authenticated:
        raise RuntimeError(
            "W&B is not authenticated. Set WANDB_API_KEY in .env or the shell, "
            "run `wandb login`, or pass --no-wandb."
        )


def _wandb_run(
    bundle: dict[str, Any],
    spec: SynthSAEBenchRunSpec,
    run_dir: Path,
    mode: str,
):
    if mode == "disabled":
        return None
    import wandb

    config = SynthSAEBenchSweepConfig.from_dict(bundle["sweep_config"])
    sweep_dir = next(
        (parent for parent in run_dir.parents if (parent / "manifest.json").exists()),
        run_dir.parent,
    )
    return wandb.init(
        project=config.wandb_project or WANDB_PROJECT,
        name=spec.run_id,
        group=sweep_dir.name,
        config={**bundle, "exp_id": sweep_experiment_id(config)},
        mode=mode,
        dir=str(run_dir),
    )


def _trainer(
    config: SynthSAEBenchSweepConfig,
    spec: SynthSAEBenchRunSpec,
    model,
    synthetic,
    device: str,
) -> SAETrainer:
    training = config.training
    data_provider = SyntheticActivationIterator(
        feature_dict=synthetic.feature_dict,
        activations_generator=synthetic.activation_generator,
        batch_size=training.batch_size,
        autocast=training.autocast_data,
    )
    decay_steps = round(config.total_training_steps * training.lr_decay_fraction)
    trainer_type = VGSAETrainer if spec.method == "vgsae" else SAETrainer
    return trainer_type(
        cfg=SAETrainerConfig(
            total_training_samples=config.data.n_train,
            train_batch_size_samples=training.batch_size,
            lr=training.lr,
            lr_end=training.lr / 10,
            lr_scheduler_name="constant",
            lr_warm_up_steps=0,
            lr_decay_steps=decay_steps,
            adam_beta1=0.9,
            adam_beta2=0.999,
            device=device,
            autocast=training.autocast_sae,
            dead_feature_window=training.dead_feature_window,
            feature_sampling_window=training.feature_sampling_window,
            n_batches_for_norm_estimate=training.n_batches_for_norm_estimate,
            logger=LoggingConfig(log_to_wandb=False),
            n_checkpoints=0,
            save_final_checkpoint=False,
        ),
        sae=model,
        data_provider=data_provider,
    )


def _state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    return value


def _save_resume_checkpoint(
    path: Path,
    *,
    trainer: SAETrainer,
    run_fingerprint: str,
    history: list[dict[str, Any]],
    last_loss: float,
    elapsed_seconds: float,
    device: str,
) -> None:
    """Atomically save the rolling state needed to continue an exact stream."""

    rng_state = capture_rng_state(device)
    payload = {
        "format_version": 1,
        "run_fingerprint": run_fingerprint,
        "model_state": _state_to_cpu(trainer.sae.state_dict()),
        "optimizer_state": _state_to_cpu(trainer.optimizer.state_dict()),
        "grad_scaler_state": _state_to_cpu(trainer.grad_scaler.state_dict()),
        "lr_scheduler_state": _state_to_cpu(trainer.lr_scheduler.state_dict()),
        "coefficient_scheduler_states": {
            name: {
                "state_dict": _state_to_cpu(scheduler.state_dict()),
                # SAELens 6.47 omits current_value from state_dict even though
                # get_coefficients() consumes it on the next optimizer step.
                "current_value": scheduler.current_value,
            }
            for name, scheduler in trainer.coefficient_schedulers.items()
        },
        "n_training_samples": trainer.n_training_samples,
        "n_training_steps": trainer.n_training_steps,
        "act_freq_scores": _state_to_cpu(trainer.act_freq_scores),
        "n_forward_passes_since_fired": _state_to_cpu(
            trainer.n_forward_passes_since_fired
        ),
        "n_frac_active_samples": trainer.n_frac_active_samples,
        "started_fine_tuning": trainer.started_fine_tuning,
        "activation_scaling_factor": _state_to_cpu(
            trainer.activation_scaler.scaling_factor
        ),
        "history": history,
        "last_loss": last_loss,
        "elapsed_seconds": elapsed_seconds,
        "rng_device_type": torch.device(device).type,
        **rng_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_resume_checkpoint(
    path: Path,
    *,
    trainer: SAETrainer,
    run_fingerprint: str,
    device: str,
) -> tuple[
    list[dict[str, Any]],
    float,
    float,
    torch.Tensor,
    torch.Tensor | None,
]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format_version") != 1
        or payload.get("run_fingerprint") != run_fingerprint
    ):
        raise ValueError(f"Rolling checkpoint does not match {path.parent.parent}.")
    current_device_type = torch.device(device).type
    if payload.get("rng_device_type") != current_device_type:
        raise ValueError(
            "Rolling checkpoint cannot change RNG device type: "
            f"saved {payload.get('rng_device_type')!r}, requested "
            f"{current_device_type!r}."
        )
    trainer.sae.load_state_dict(payload["model_state"])
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.grad_scaler.load_state_dict(payload["grad_scaler_state"])
    trainer.lr_scheduler.load_state_dict(payload["lr_scheduler_state"])
    for name, state in payload["coefficient_scheduler_states"].items():
        scheduler = trainer.coefficient_schedulers[name]
        scheduler.load_state_dict(state["state_dict"])
        scheduler.current_value = state["current_value"]
    trainer.n_training_samples = int(payload["n_training_samples"])
    trainer.n_training_steps = int(payload["n_training_steps"])
    trainer.act_freq_scores = payload["act_freq_scores"].to(device)
    trainer.n_forward_passes_since_fired = payload[
        "n_forward_passes_since_fired"
    ].to(device)
    trainer.n_frac_active_samples = int(payload["n_frac_active_samples"])
    trainer.started_fine_tuning = bool(payload["started_fine_tuning"])
    trainer.activation_scaler.scaling_factor = payload["activation_scaling_factor"]
    history = [dict(row) for row in payload["history"]]
    return (
        history,
        float(payload["last_loss"]),
        float(payload.get("elapsed_seconds", 0.0)),
        payload["cpu_rng_state"].cpu(),
        (
            payload["device_rng_state"].cpu()
            if payload["device_rng_state"] is not None
            else None
        ),
    )


@torch.no_grad()
def _history_row(trainer: SAETrainer, output, spec: SynthSAEBenchRunSpec) -> dict[str, Any]:
    sae_out = trainer.activation_scaler.unscale(output.sae_out)
    sae_in = trainer.activation_scaler.unscale(output.sae_in)
    reconstruction_mse = (sae_out - sae_in).float().pow(2).mean()
    hard_l0 = output.feature_acts.bool().float().sum(-1).mean()
    row: dict[str, Any] = {
        "step": trainer.n_training_steps,
        "n_training_samples": trainer.n_training_samples,
        "loss": float(output.loss.detach().float().cpu()),
        "reconstruction_mse": float(reconstruction_mse.cpu()),
        "rho": float(hard_l0.cpu() / trainer.sae.cfg.d_sae),
        "average_l0": float(hard_l0.cpu()),
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
    }
    for name, value in output.losses.items():
        row[name] = float(value.detach().float().cpu())
    for name, value in output.metrics.items():
        row[name] = float(value.detach().float().cpu())
    row.update(
        {
            f"{name}_coefficient": value
            for name, value in trainer.get_coefficients().items()
        }
    )
    return row


def _run_metadata(
    config: SynthSAEBenchSweepConfig,
    spec: SynthSAEBenchRunSpec,
    device: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "exp_id": sweep_experiment_id(config),
        "run_id": spec.run_id,
        "seed": spec.seed,
        "init_seed": spec.init_seed,
        "calibration_seed": spec.calibration_seed,
        "train_stream_seed": spec.train_stream_seed,
        "eval_stream_seed": spec.eval_stream_seed,
        "method": spec.method,
        "method_label": METHOD_LABELS[spec.method],
        "control_name": spec.control_name,
        "control_value": spec.control_value,
        "input_dim": config.data.input_dim,
        "ground_truth_num_features": config.data.ground_truth_num_features,
        "sae_width": config.data.sae_width,
        "n_training_samples_requested": config.data.n_train,
        "n_test_samples": config.data.n_test,
        "batch_size": config.training.batch_size,
        "benchmark_model_id": config.data.model_id,
        "benchmark_revision": config.data.revision,
        "benchmark_model_config_sha256": config.data.model_config_sha256,
        "benchmark_scale_children_by_parent": (
            config.data.scale_children_by_parent
        ),
        "saelens_revision": SAELENS_REVISION,
        "train_device": device,
        "train_source_fingerprint": provenance["source_fingerprint"],
        "train_pipeline_fingerprint": provenance["pipeline_fingerprint"],
    }


def train_one(run_dir: Path, device: str, wandb_mode: str, force: bool) -> None:
    bundle = read_json(run_dir / "config.json")
    config = SynthSAEBenchSweepConfig.from_dict(bundle["sweep_config"])
    spec = SynthSAEBenchRunSpec.from_dict(bundle["spec"])
    current_provenance = runtime_provenance(PROJECT_ROOT, TRAIN_SOURCE_FILES)
    if (
        current_provenance["pipeline_fingerprint"]
        != bundle["train_provenance"]["pipeline_fingerprint"]
    ):
        raise RuntimeError("Training source changed after this run was queued.")
    if not force and _is_complete(run_dir, bundle["fingerprint"]):
        print(f"Skipping complete run: {spec.run_id}")
        return

    write_json(
        run_dir / "train_status.json",
        {
            "state": "running",
            "fingerprint": bundle["fingerprint"],
            "device": device,
            "started_at": utc_now(),
        },
    )
    checkpoint = run_dir / "checkpoints" / "last.pt"
    resume_checkpoint = run_dir / "checkpoints" / "resume.pt"
    checkpoint.unlink(missing_ok=True)
    if force:
        resume_checkpoint.unlink(missing_ok=True)
    wandb_run = _wandb_run(bundle, spec, run_dir, wandb_mode)
    started = time.perf_counter()
    try:
        synthetic, _ = load_benchmark_model(config, device)
        with temporary_seed_for_device(spec.init_seed, device):
            model = build_model(config, spec, device)
        trainer = _trainer(config, spec, model, synthetic, device)
        history: list[dict[str, Any]] = []
        last_loss: float | None = None
        elapsed_before = 0.0
        resume_cpu_rng: torch.Tensor | None = None
        resume_device_rng: torch.Tensor | None = None
        resumed = resume_checkpoint.exists()
        if resumed:
            (
                history,
                last_loss,
                elapsed_before,
                resume_cpu_rng,
                resume_device_rng,
            ) = _load_resume_checkpoint(
                resume_checkpoint,
                trainer=trainer,
                run_fingerprint=bundle["fingerprint"],
                device=device,
            )
            print(
                f"Resuming {spec.run_id} at step={trainer.n_training_steps} "
                f"samples={trainer.n_training_samples}"
            )
        elif model.cfg.normalize_activations == "expected_average_only_in":
            calibration_provider = SyntheticActivationIterator(
                feature_dict=synthetic.feature_dict,
                activations_generator=synthetic.activation_generator,
                batch_size=config.training.batch_size,
                autocast=config.training.autocast_data,
            )
            with temporary_seed_for_device(spec.calibration_seed, device):
                trainer.activation_scaler.estimate_scaling_factor(
                    d_in=model.cfg.d_in,
                    data_provider=calibration_provider,
                    n_batches_for_norm_estimate=(
                        config.training.n_batches_for_norm_estimate
                    ),
                )

        with temporary_seed_for_device(
            spec.train_stream_seed,
            device,
            cpu_rng_state=resume_cpu_rng,
            device_rng_state=resume_device_rng,
        ):
            while trainer.n_training_samples < config.data.n_train:
                trainer.maybe_reset_sparsity()
                output = trainer.step(next(trainer.data_provider))
                step = trainer.n_training_steps
                is_last = trainer.n_training_samples >= config.data.n_train
                last_loss = float(output.loss.detach().float().cpu())
                if step % config.training.history_every == 0 or is_last:
                    row = _history_row(trainer, output, spec)
                    history.append(row)
                    if wandb_run is not None:
                        wandb_run.log(row, step=step)
                    print(
                        f"{spec.run_id} step={step} "
                        f"samples={trainer.n_training_samples} "
                        f"loss={row['loss']:.6g} l0={row['average_l0']:.3f}"
                    )
                trainer.n_training_steps += 1
                if (
                    trainer.n_training_steps % config.training.resume_every == 0
                    or is_last
                ):
                    _save_resume_checkpoint(
                        resume_checkpoint,
                        trainer=trainer,
                        run_fingerprint=bundle["fingerprint"],
                        history=history,
                        last_loss=last_loss,
                        elapsed_seconds=(
                            elapsed_before + time.perf_counter() - started
                        ),
                        device=device,
                    )
        if last_loss is None:
            raise RuntimeError("Training produced no optimizer steps.")

        activation_scale = trainer.activation_scaler.scaling_factor
        if activation_scale is not None:
            model.fold_activation_norm_scaling_factor(activation_scale)
            trainer.activation_scaler.scaling_factor = None
        trainer.set_final_sae_metadata()
        elapsed = elapsed_before + time.perf_counter() - started
        metadata = _run_metadata(config, spec, device, bundle["train_provenance"])
        metadata.update(
            activation_scale=(
                float(activation_scale) if activation_scale is not None else 1.0
            ),
            n_training_steps=trainer.n_training_steps,
            n_training_samples_actual=trainer.n_training_samples,
            train_seconds=elapsed,
        )
        save_checkpoint(
            checkpoint,
            model=model,
            config=config,
            spec=spec,
            step=trainer.n_training_steps - 1,
            n_training_samples=trainer.n_training_samples,
            loss=last_loss,
            metadata={
                "train_fingerprint": bundle["fingerprint"],
                "train_provenance": bundle["train_provenance"],
                **metadata,
            },
        )
        write_rows(
            run_dir / "training_history.csv",
            ({**metadata, **row} for row in history),
        )
        summary = {
            **metadata,
            "last_step": trainer.n_training_steps - 1,
            "last_loss": last_loss,
            "resumed": resumed,
            "train_provenance": bundle["train_provenance"],
        }
        write_json(run_dir / "training_summary.json", summary)
        if wandb_run is not None:
            wandb_run.summary.update(summary)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    write_json(
        run_dir / "train_status.json",
        {
            "state": "complete",
            "fingerprint": bundle["fingerprint"],
            "device": device,
            "finished_at": utc_now(),
        },
    )
    resume_checkpoint.unlink(missing_ok=True)


def prepare_runs(
    sweep_dir: Path,
    config: SynthSAEBenchSweepConfig,
    specs: list[SynthSAEBenchRunSpec],
    force: bool,
    wandb_mode: str,
) -> tuple[list[ScriptTask], list[Path]]:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config.to_dict()
    manifest_path = sweep_dir / "manifest.json"
    entries: dict[str, dict[str, Any]] = {}
    provenance = runtime_provenance(PROJECT_ROOT, TRAIN_SOURCE_FILES)
    if manifest_path.exists():
        previous = read_json(manifest_path)
        previous_config = SynthSAEBenchSweepConfig.from_dict(
            previous["sweep_config"]
        ).to_dict()
        if fingerprint(previous_config) != fingerprint(config_dict):
            raise ValueError("Output directory contains a different sweep configuration.")
        entries = {entry["run_id"]: entry for entry in previous["runs"]}

    tasks: list[ScriptTask] = []
    for spec in specs:
        run_dir = run_directory(sweep_dir, spec)
        bundle = _bundle(config, spec, provenance)
        config_path = run_dir / "config.json"
        if (
            config_path.exists()
            and read_json(config_path).get("fingerprint") != bundle["fingerprint"]
            and not force
        ):
            raise ValueError(
                f"Configuration collision in {run_dir}; use --force or a new output."
            )
        write_json(config_path, bundle)
        relative_dir = str(run_dir.relative_to(sweep_dir))
        entries[spec.run_id] = {
            "run_id": spec.run_id,
            "relative_dir": relative_dir,
            "spec": spec.to_dict(),
        }
        if force or not _is_complete(run_dir, bundle["fingerprint"]):
            logging_arg = (
                "--no-wandb"
                if wandb_mode == "disabled"
                else f"--wandb-mode={wandb_mode}"
            )
            worker_args = (
                "--worker",
                f"--run-dir={run_dir}",
                logging_arg,
                *(("--force",) if force else ()),
            )
            tasks.append(ScriptTask(Path(__file__).resolve(), worker_args, spec.run_id))
            write_json(
                run_dir / "train_status.json",
                {
                    "state": "queued",
                    "fingerprint": bundle["fingerprint"],
                    "queued_at": utc_now(),
                },
            )

    ordered_entries = sorted(entries.values(), key=lambda entry: entry["run_id"])
    write_json(
        manifest_path,
        {
            "format_version": 1,
            "benchmark_identity": {
                "model_id": config.data.model_id,
                "revision": config.data.revision,
                "model_config_sha256": config.data.model_config_sha256,
                "pretrained_artifact_scale_children_by_parent": (
                    config.data.scale_children_by_parent
                ),
                "saelens_revision": SAELENS_REVISION,
            },
            "sweep_config": config_dict,
            "runs": ordered_entries,
        },
    )
    write_json(sweep_dir / "sweep_config.json", config_dict)
    all_run_dirs = [sweep_dir / entry["relative_dir"] for entry in ordered_entries]
    return tasks, all_run_dirs


def main(args: argparse.Namespace) -> int:
    _load_project_env()
    if args.worker:
        if args.run_dir is None:
            raise ValueError("--worker requires --run-dir.")
        try:
            train_one(args.run_dir.resolve(), args.device, args.wandb_mode, args.force)
        except BaseException as error:
            config_path = args.run_dir / "config.json"
            run_fingerprint = (
                read_json(config_path).get("fingerprint") if config_path.exists() else None
            )
            write_json(
                args.run_dir / "train_status.json",
                {
                    "state": "failed",
                    "fingerprint": run_fingerprint,
                    "failed_at": utc_now(),
                    "error": repr(error),
                },
            )
            raise
        return 0

    config = configured_sweep(args)
    specs = selected_specs(config, args.methods)
    sweep_dir = (args.output_dir or default_sweep_dir(PROJECT_ROOT, config)).resolve()
    tasks, run_dirs = prepare_runs(
        sweep_dir, config, specs, args.force, args.wandb_mode
    )
    if tasks:
        _preflight_wandb(args.wandb_mode)
    return_code = ParallelExecutor(
        tasks,
        resolve_devices(args.devices),
        max_per_device=args.max_per_device,
    ).run_all()
    incomplete = [
        run_dir
        for run_dir in run_dirs
        if not _is_complete(
            run_dir, read_json(run_dir / "config.json")["fingerprint"]
        )
    ]
    if return_code or incomplete:
        print(f"Training incomplete: {len(incomplete)} run(s) lack valid checkpoints.")
        return 1
    aggregate_csv(
        run_dirs,
        Path("training_history.csv"),
        sweep_dir / "summary" / "training_curves.csv",
    )
    print(f"Sweep artifacts: {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
