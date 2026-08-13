"""Train the Stage-3 real-model activation sweep, then launch evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from dotenv import load_dotenv
from sae_lens import LoggingConfig
from sae_lens.config import SAETrainerConfig
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
from runs.gpu_scheduler import (  # noqa: E402
    ParallelExecutor,
    ScriptTask,
    activate_worker_device,
)
from src.real_activations import (  # noqa: E402
    ResumableActivationProvider,
    load_real_language_model,
    make_live_activation_provider,
)
from src.real_activation_sweep import (  # noqa: E402
    REAL_MODEL_TARGETS,
    STAGE3_CONTROL_NAMES,
    STAGE3_METHOD_LABELS,
    STAGE3_METHOD_ORDER,
    RealActivationRunSpec,
    RealActivationSweepConfig,
    augment_stage3_runtime_provenance,
    build_model,
    build_specs,
    default_sweep_config,
    default_sweep_dir,
    method_learning_rate,
    run_directory,
    save_checkpoint,
    sweep_experiment_id,
)
from src.saelens_vg import VGSAETrainer  # noqa: E402


DEFAULT_MAX_PER_DEVICE = 1
WANDB_PROJECT = "vg-sae"
TRAIN_SOURCE_FILES = (
    "runs/_sweep_io.py",
    "runs/gpu_scheduler.py",
    "runs/run_RealActivation_sweep.py",
    "src/real_activations.py",
    "src/real_activation_sweep.py",
    "src/sae_baselines.py",
    "src/sae_model.py",
    "src/model.py",
    "src/saelens_vg.py",
)


def _stage3_runtime_provenance() -> dict[str, Any]:
    return augment_stage3_runtime_provenance(
        runtime_provenance(PROJECT_ROOT, TRAIN_SOURCE_FILES)
    )


def _load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="One serialized Stage-3 config JSON.")
    parser.add_argument(
        "--targets",
        default="all",
        help="Comma-separated target names, or all (ignored with --config).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Parent for target-specific sweep directories (default: outputs/runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Exact sweep directory; valid only when one target is selected.",
    )
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--methods", default="all", help="Comma-separated methods, or all.")
    seed = parser.add_mutually_exclusive_group()
    seed.add_argument("--seed", type=int)
    seed.add_argument("--seeds", help="Comma-separated experiment seeds.")
    parser.add_argument(
        "--model-sparsity-control",
        "--sparsity-control",
        dest="sparsity_controls",
        action="append",
        metavar="METHOD=V1,V2,...",
        help="Replace one method's target-specific default grid; repeat per method.",
    )
    parser.add_argument("--training-tokens", type=int)
    parser.add_argument("--eval-tokens", type=int)
    parser.add_argument("--downstream-eval-tokens", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--history-every", type=int)
    parser.add_argument("--resume-every", type=int)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--max-per-device", type=int, default=DEFAULT_MAX_PER_DEVICE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Stop after training; the default automatically evaluates every checkpoint.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    return parser.parse_args()


def _selected_target_names(raw: str) -> list[str]:
    if raw == "all":
        return list(REAL_MODEL_TARGETS)
    names = [value.strip() for value in raw.split(",") if value.strip()]
    if not names:
        raise ValueError("--targets must name at least one target or use 'all'.")
    unknown = set(names) - set(REAL_MODEL_TARGETS)
    if unknown:
        raise ValueError(f"Unknown Stage-3 target(s): {', '.join(sorted(unknown))}")
    if len(set(names)) != len(names):
        raise ValueError("--targets contains duplicates.")
    return names


def _control_overrides(entries: list[str]) -> dict[str, list[float | int]]:
    overrides: dict[str, list[float | int]] = {}
    for entry in entries:
        method, separator, raw_values = entry.partition("=")
        method = method.strip().lower()
        if not separator or method not in STAGE3_CONTROL_NAMES:
            raise ValueError(f"Invalid sparsity control {entry!r}.")
        if method in overrides:
            raise ValueError(f"Duplicate sparsity-control override for {method!r}.")
        tokens = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not tokens:
            raise ValueError(f"Sparsity-control grid for {method!r} is empty.")
        if method == "batchtopk":
            floats = [float(value) for value in tokens]
            if any(not value.is_integer() for value in floats):
                raise ValueError("BatchTopK controls must be integers.")
            overrides[method] = [int(value) for value in floats]
        else:
            overrides[method] = [float(value) for value in tokens]
    return overrides


def _selected_methods(raw: str) -> list[str]:
    if raw == "all":
        return list(STAGE3_METHOD_ORDER)
    methods = [value.strip().lower() for value in raw.split(",") if value.strip()]
    if not methods:
        raise ValueError("--methods must name at least one method or use 'all'.")
    unknown = set(methods) - set(STAGE3_METHOD_ORDER)
    if unknown:
        raise ValueError(f"Unknown Stage-3 method(s): {', '.join(sorted(unknown))}")
    if len(set(methods)) != len(methods):
        raise ValueError("--methods contains duplicates.")
    return [method for method in STAGE3_METHOD_ORDER if method in methods]


def _config_from_args(
    base: RealActivationSweepConfig, args: argparse.Namespace
) -> RealActivationSweepConfig:
    raw = base.to_dict()
    if args.seed is not None:
        raw["seeds"] = [args.seed]
    elif args.seeds:
        raw["seeds"] = [int(value) for value in args.seeds.split(",")]
    # CLI --methods selects queued specs without mutating the sweep identity;
    # this permits adding another method to the same target root later.
    _selected_methods(args.methods)
    raw["controls"].update(_control_overrides(args.sparsity_controls or []))
    if args.training_tokens is not None:
        raw["data"]["n_train_tokens"] = args.training_tokens
        raw["data"]["eval_token_offset"] = (
            raw["data"]["train_token_offset"] + args.training_tokens
        )
    if args.eval_tokens is not None:
        raw["data"]["n_eval_tokens"] = args.eval_tokens
        if args.downstream_eval_tokens is None:
            raw["data"]["n_downstream_eval_tokens"] = min(
                raw["data"]["n_downstream_eval_tokens"], args.eval_tokens
            )
    if args.downstream_eval_tokens is not None:
        raw["data"]["n_downstream_eval_tokens"] = args.downstream_eval_tokens
    if args.batch_size is not None:
        raw["training"]["batch_size"] = args.batch_size
    if args.history_every is not None:
        raw["training"]["history_every"] = args.history_every
    if args.resume_every is not None:
        raw["training"]["resume_every"] = args.resume_every
    if args.fast_dev_run:
        raw["seeds"] = raw["seeds"][:1]
        raw["controls"] = {
            method: [raw["controls"][method][0]] for method in raw["methods"]
        }
        raw["data"].update(
            n_train_tokens=4_096,
            eval_token_offset=4_096,
            n_eval_tokens=4_096,
            n_downstream_eval_tokens=4_096,
        )
        raw["training"].update(
            batch_size=4_096,
            history_every=1,
            resume_every=1,
            preview_tokens=16,
            store_batch_size_prompts=1,
            eval_store_batch_size_prompts=1,
            n_batches_in_buffer=1,
            activations_mixing_fraction=0.0,
        )
    return RealActivationSweepConfig.from_dict(raw)


def configured_sweeps(args: argparse.Namespace) -> list[RealActivationSweepConfig]:
    if args.config is not None:
        configs = [RealActivationSweepConfig.from_dict(read_json(args.config))]
    else:
        configs = [default_sweep_config(name) for name in _selected_target_names(args.targets)]
    return [_config_from_args(config, args) for config in configs]


def selected_specs(
    config: RealActivationSweepConfig, methods: str
) -> list[RealActivationRunSpec]:
    requested = set(_selected_methods(methods))
    available = set(config.methods)
    if missing := requested - available:
        raise ValueError(
            "Methods are absent from the serialized config: "
            f"{', '.join(sorted(missing))}"
        )
    return [spec for spec in build_specs(config) if spec.method in requested]


def _bundle(
    config: RealActivationSweepConfig,
    spec: RealActivationRunSpec,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "format_version": 1,
        "sweep_config": config.to_dict(),
        "spec": spec.to_dict(),
        "checkpoint_selection": "final streaming-training state",
        "train_provenance": provenance,
        "activation_identity": {
            "model_id": config.data.model_id,
            "model_revision": config.data.model_revision,
            "hook_name": config.data.hook_name,
            "dataset_id": config.data.dataset_id,
            "dataset_revision": config.data.dataset_revision,
            "train_token_offset": config.data.train_token_offset,
            "n_train_tokens": config.data.n_train_tokens,
        },
    }
    payload["fingerprint"] = fingerprint(
        {**payload, "train_provenance": provenance["pipeline_fingerprint"]}
    )
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


def _completed_manifest_training_runs(sweep_dir: Path) -> list[Path]:
    """Return every current manifest run with valid complete training artifacts."""

    manifest = read_json(sweep_dir / "manifest.json")
    completed: list[Path] = []
    for entry in manifest["runs"]:
        run_dir = sweep_dir / entry["relative_dir"]
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        run_fingerprint = read_json(config_path).get("fingerprint")
        if isinstance(run_fingerprint, str) and _is_complete(
            run_dir, run_fingerprint
        ):
            completed.append(run_dir)
    return completed


def _preflight_wandb() -> None:
    import wandb

    try:
        authenticated = wandb.login(verify=True, force=True)
    except Exception:
        raise RuntimeError(
            "W&B authentication preflight failed; check WANDB_API_KEY in .env "
            "or the existing W&B login."
        ) from None
    if not authenticated:
        raise RuntimeError("Stage-3 training requires authenticated online W&B logging.")


def _wandb_run(bundle: dict[str, Any], spec: RealActivationRunSpec, run_dir: Path):
    import wandb

    config = RealActivationSweepConfig.from_dict(bundle["sweep_config"])
    sweep_dir = next(
        (parent for parent in run_dir.parents if (parent / "manifest.json").exists()),
        None,
    )
    if sweep_dir is None:
        raise ValueError(f"Cannot identify sweep root for {run_dir}.")
    return wandb.init(
        project=config.wandb_project or WANDB_PROJECT,
        id=fingerprint(
            {
                "sweep": sweep_dir.name,
                "run_id": spec.run_id,
                "train_fingerprint": bundle["fingerprint"],
            }
        )[:24],
        resume="allow",
        name=spec.run_id,
        group=sweep_dir.name,
        job_type=spec.method,
        tags=(
            f"stage:{config.experiment_name}",
            f"target:{config.data.target_name}",
            f"method:{spec.method}",
            "beta_mode:learned",
        ),
        config={
            **bundle,
            "exp_id": sweep_experiment_id(config),
            "stage": config.experiment_name,
            "sweep_root": sweep_dir.name,
        },
        mode="online",
        force=True,
        dir=str(run_dir),
    )


@contextmanager
def _temporary_seed(
    seed: int,
    device: torch.device | str,
    *,
    cpu_rng_state: torch.Tensor | None = None,
    device_rng_state: torch.Tensor | None = None,
) -> Iterator[None]:
    normalized = torch.device(device)
    cuda_devices: list[int] = []
    device_index: int | None = None
    if normalized.type == "cuda":
        device_index = normalized.index
        if device_index is None:
            device_index = torch.cuda.current_device()
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
                raise ValueError("A saved CUDA RNG state requires a CUDA worker.")
            torch.cuda.set_rng_state(device_rng_state.cpu(), device_index)
        yield


def _capture_rng_state(device: torch.device | str) -> dict[str, torch.Tensor | None]:
    normalized = torch.device(device)
    device_state = None
    if normalized.type == "cuda":
        index = normalized.index
        if index is None:
            index = torch.cuda.current_device()
        device_state = torch.cuda.get_rng_state(index).cpu()
    return {"cpu_rng_state": torch.get_rng_state().cpu(), "device_rng_state": device_state}


def _trainer(
    config: RealActivationSweepConfig,
    spec: RealActivationRunSpec,
    model: Any,
    provider: ResumableActivationProvider,
    device: str,
) -> SAETrainer:
    trainer_type = VGSAETrainer if spec.method == "vgsae" else SAETrainer
    learning_rate = method_learning_rate(config, spec.method)
    return trainer_type(
        cfg=SAETrainerConfig(
            total_training_samples=config.data.n_train_tokens,
            train_batch_size_samples=config.training.batch_size,
            lr=learning_rate,
            lr_end=learning_rate,
            lr_scheduler_name="constant",
            lr_warm_up_steps=0,
            lr_decay_steps=0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            device=device,
            autocast=config.training.autocast_sae,
            dead_feature_window=config.training.dead_feature_window,
            feature_sampling_window=config.training.feature_sampling_window,
            n_batches_for_norm_estimate=config.training.n_batches_for_norm_estimate,
            logger=LoggingConfig(log_to_wandb=False),
            n_checkpoints=0,
            save_final_checkpoint=False,
        ),
        sae=model,
        data_provider=provider,
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
    provider: ResumableActivationProvider,
    run_fingerprint: str,
    history: list[dict[str, Any]],
    last_loss: float,
    elapsed_seconds: float,
    device: str,
) -> None:
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
        "activation_provider_state": provider.state_dict(),
        "history": history,
        "last_loss": last_loss,
        "elapsed_seconds": elapsed_seconds,
        "rng_device_type": torch.device(device).type,
        **_capture_rng_state(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_resume_checkpoint(
    path: Path,
    *,
    trainer: SAETrainer,
    provider: ResumableActivationProvider,
    run_fingerprint: str,
    device: str,
) -> tuple[list[dict[str, Any]], float, float, torch.Tensor, torch.Tensor | None]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("run_fingerprint") != run_fingerprint:
        raise ValueError(f"Rolling checkpoint does not match {path.parent.parent}.")
    if payload.get("rng_device_type") != torch.device(device).type:
        raise ValueError("A rolling checkpoint cannot change RNG device type.")
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
    provider.load_state_dict(payload["activation_provider_state"])
    return (
        [dict(row) for row in payload["history"]],
        float(payload["last_loss"]),
        float(payload.get("elapsed_seconds", 0.0)),
        payload["cpu_rng_state"].cpu(),
        None
        if payload["device_rng_state"] is None
        else payload["device_rng_state"].cpu(),
    )


@torch.no_grad()
def _history_row(trainer: SAETrainer, output: Any) -> dict[str, Any]:
    sae_out = trainer.activation_scaler.unscale(output.sae_out).float()
    sae_in = trainer.activation_scaler.unscale(output.sae_in).float()
    hard_l0 = output.feature_acts.bool().float().sum(-1).mean()
    row: dict[str, Any] = {
        "step": trainer.n_training_steps,
        "n_training_tokens": trainer.n_training_samples,
        "loss": float(output.loss.detach().float().cpu()),
        "reconstruction_mse": float((sae_out - sae_in).pow(2).mean().cpu()),
        "average_l0": float(hard_l0.cpu()),
        "rho": float((hard_l0 / trainer.sae.cfg.d_sae).cpu()),
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
    }
    for name, value in output.losses.items():
        row[name] = float(value.detach().float().cpu())
    for name, value in output.metrics.items():
        row[name] = float(value.detach().float().cpu())
    row.update(
        {f"{name}_coefficient": value for name, value in trainer.get_coefficients().items()}
    )
    return row


def _run_metadata(
    config: RealActivationSweepConfig,
    spec: RealActivationRunSpec,
    device: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "exp_id": sweep_experiment_id(config),
        "run_id": spec.run_id,
        "target_name": config.data.target_name,
        "model_id": config.data.model_id,
        "model_revision": config.data.model_revision,
        "layer": config.data.layer,
        "hook_name": config.data.hook_name,
        "paper_hook_name": config.data.paper_hook_name,
        "dataset_id": config.data.dataset_id,
        "dataset_revision": config.data.dataset_revision,
        "seed": spec.seed,
        "init_seed": spec.init_seed,
        "train_stream_seed": spec.train_stream_seed,
        "eval_stream_seed": spec.eval_stream_seed,
        "method": spec.method,
        "method_label": STAGE3_METHOD_LABELS[spec.method],
        "control_name": spec.control_name,
        "control_value": spec.control_value,
        "beta_mode": "learned",
        "beta_initial": config.training.beta,
        "input_dim": config.data.input_dim,
        "sae_width": config.data.sae_width,
        "paper_reported_train_tokens": config.data.paper_reported_train_tokens,
        "n_training_tokens_requested": config.data.n_train_tokens,
        "n_eval_tokens": config.data.n_eval_tokens,
        "batch_size": config.training.batch_size,
        "train_device": device,
        "train_source_fingerprint": provenance["source_fingerprint"],
        "train_pipeline_fingerprint": provenance["pipeline_fingerprint"],
    }


def _final_beta(spec: RealActivationRunSpec, model: Any) -> float | None:
    if spec.method != "vgsae":
        return None
    if model.core.log_beta is None:
        raise RuntimeError("Learned-beta VG run has no trainable log_beta.")
    return float(model.core.log_beta.exp().detach().cpu())


def train_one(run_dir: Path, device: str, force: bool) -> None:
    activate_worker_device(device)
    bundle = read_json(run_dir / "config.json")
    config = RealActivationSweepConfig.from_dict(bundle["sweep_config"])
    spec = RealActivationRunSpec.from_dict(bundle["spec"])
    current_provenance = _stage3_runtime_provenance()
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
    final_checkpoint = run_dir / "checkpoints" / "last.pt"
    resume_checkpoint = run_dir / "checkpoints" / "resume.pt"
    final_checkpoint.unlink(missing_ok=True)
    if force:
        resume_checkpoint.unlink(missing_ok=True)
    wandb_run = _wandb_run(bundle, spec, run_dir)
    started = time.perf_counter()
    try:
        language_model = load_real_language_model(config.data, device)
        provider = make_live_activation_provider(
            config.data,
            language_model,
            start_token=config.data.train_token_offset,
            total_tokens=config.data.n_train_tokens,
            batch_size=config.training.batch_size,
            prompt_batch_size=config.training.store_batch_size_prompts,
            n_batches_in_buffer=config.training.n_batches_in_buffer,
            activation_device=device,
            mix_fraction=config.training.activations_mixing_fraction,
            seed=spec.train_stream_seed,
            autocast_lm=config.training.autocast_data,
        )
        with _temporary_seed(spec.init_seed, device):
            model = build_model(config, spec, device)
        trainer = _trainer(config, spec, model, provider, device)
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
                provider=provider,
                run_fingerprint=bundle["fingerprint"],
                device=device,
            )
            print(
                f"Resuming {spec.run_id} at step={trainer.n_training_steps} "
                f"tokens={trainer.n_training_samples}"
            )

        with _temporary_seed(
            spec.train_stream_seed,
            device,
            cpu_rng_state=resume_cpu_rng,
            device_rng_state=resume_device_rng,
        ):
            while trainer.n_training_samples < config.data.n_train_tokens:
                trainer.maybe_reset_sparsity()
                output = trainer.step(next(provider))
                step = trainer.n_training_steps
                is_last = trainer.n_training_samples >= config.data.n_train_tokens
                last_loss = float(output.loss.detach().float().cpu())
                if not math.isfinite(last_loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at step={step}, "
                        f"tokens={trainer.n_training_samples}: {last_loss}"
                    )
                if step % config.training.history_every == 0 or is_last:
                    row = _history_row(trainer, output)
                    history.append(row)
                    wandb_run.log(row, step=step)
                    print(
                        f"{spec.run_id} step={step} tokens={trainer.n_training_samples} "
                        f"loss={row['loss']:.6g} l0={row['average_l0']:.3f}"
                    )
                trainer.n_training_steps += 1
                if trainer.n_training_steps % config.training.resume_every == 0 or is_last:
                    _save_resume_checkpoint(
                        resume_checkpoint,
                        trainer=trainer,
                        provider=provider,
                        run_fingerprint=bundle["fingerprint"],
                        history=history,
                        last_loss=last_loss,
                        elapsed_seconds=elapsed_before + time.perf_counter() - started,
                        device=device,
                    )
        if last_loss is None:
            raise RuntimeError("Training produced no optimizer steps.")
        if provider.tokens_yielded != config.data.n_train_tokens:
            raise RuntimeError("Training activation stream did not finish at its exact budget.")

        activation_scale = trainer.activation_scaler.scaling_factor
        if activation_scale is not None:
            model.fold_activation_norm_scaling_factor(activation_scale)
            trainer.activation_scaler.scaling_factor = None
        trainer.set_final_sae_metadata()
        elapsed = elapsed_before + time.perf_counter() - started
        metadata = _run_metadata(config, spec, device, bundle["train_provenance"])
        metadata.update(
            activation_scale=float(activation_scale) if activation_scale is not None else 1.0,
            n_training_steps=trainer.n_training_steps,
            n_training_tokens_actual=trainer.n_training_samples,
            train_seconds=elapsed,
        )
        final_beta = _final_beta(spec, model)
        save_checkpoint(
            final_checkpoint,
            model=model,
            config=config,
            spec=spec,
            step=trainer.n_training_steps - 1,
            n_training_tokens=trainer.n_training_samples,
            loss=last_loss,
            metadata={
                "train_fingerprint": bundle["fingerprint"],
                "train_provenance": bundle["train_provenance"],
                "final_beta_precision": final_beta,
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
            "final_beta_precision": final_beta,
            "resumed": resumed,
            "train_provenance": bundle["train_provenance"],
        }
        write_json(run_dir / "training_summary.json", summary)
        wandb_run.summary.update(summary)
    finally:
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
    config: RealActivationSweepConfig,
    specs: list[RealActivationRunSpec],
    force: bool,
) -> tuple[list[ScriptTask], list[Path]]:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config.to_dict()
    manifest_path = sweep_dir / "manifest.json"
    entries: dict[str, dict[str, Any]] = {}
    provenance = _stage3_runtime_provenance()
    if manifest_path.exists():
        previous = read_json(manifest_path)
        previous_config = RealActivationSweepConfig.from_dict(previous["sweep_config"])
        if fingerprint(previous_config.to_dict()) != fingerprint(config_dict):
            raise ValueError("Output directory contains a different Stage-3 config.")
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
            raise ValueError(f"Configuration collision in {run_dir}.")
        write_json(config_path, bundle)
        entries[spec.run_id] = {
            "run_id": spec.run_id,
            "relative_dir": str(run_dir.relative_to(sweep_dir)),
            "spec": spec.to_dict(),
        }
        if force or not _is_complete(run_dir, bundle["fingerprint"]):
            worker_args = (
                "--worker",
                f"--run-dir={run_dir}",
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
            "activation_identity": {
                "target_name": config.data.target_name,
                "model_id": config.data.model_id,
                "model_revision": config.data.model_revision,
                "hook_name": config.data.hook_name,
                "dataset_id": config.data.dataset_id,
                "dataset_revision": config.data.dataset_revision,
            },
            "sweep_config": config_dict,
            "runs": ordered_entries,
        },
    )
    write_json(sweep_dir / "sweep_config.json", config_dict)
    return tasks, [sweep_dir / entry["relative_dir"] for entry in ordered_entries]


def _sweep_dir(
    config: RealActivationSweepConfig,
    *,
    output_root: Path | None,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    if output_root is None:
        return default_sweep_dir(PROJECT_ROOT, config).resolve()
    return (output_root / sweep_experiment_id(config)).resolve()


def _launch_evaluation(
    sweep_dir: Path,
    *,
    methods: str,
    devices: str,
    max_per_device: int,
    force: bool,
) -> int:
    script = PROJECT_ROOT / "runs" / "run_RealActivation_sweep_eval.py"
    command = [
        sys.executable,
        str(script),
        f"--sweep-dir={sweep_dir}",
        f"--methods={methods}",
        f"--devices={devices}",
        f"--max-per-device={max_per_device}",
    ]
    if force:
        command.append("--force")
    print(f"Training complete; launching automatic evaluation for {sweep_dir.name}.")
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def main(args: argparse.Namespace) -> int:
    _load_project_env()
    if args.worker:
        if args.run_dir is None:
            raise ValueError("--worker requires --run-dir.")
        try:
            train_one(args.run_dir.resolve(), args.device, args.force)
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

    configs = configured_sweeps(args)
    if args.output_dir is not None and len(configs) != 1:
        raise ValueError("--output-dir requires exactly one target; use --output-root.")
    prepared: list[tuple[Path, list[Path]]] = []
    tasks: list[ScriptTask] = []
    for config in configs:
        sweep_dir = _sweep_dir(
            config, output_root=args.output_root, output_dir=args.output_dir
        )
        specs = selected_specs(config, args.methods)
        target_tasks, manifest_run_dirs = prepare_runs(
            sweep_dir, config, specs, args.force
        )
        tasks.extend(target_tasks)
        selected_ids = {spec.run_id for spec in specs}
        selected_run_dirs = [
            run_dir for run_dir in manifest_run_dirs if run_dir.name in selected_ids
        ]
        prepared.append((sweep_dir, selected_run_dirs))
    if tasks:
        _preflight_wandb()
    return_code = ParallelExecutor(
        tasks,
        resolve_devices(args.devices),
        max_per_device=args.max_per_device,
    ).run_all()
    incomplete: list[Path] = []
    for _, run_dirs in prepared:
        incomplete.extend(
            run_dir
            for run_dir in run_dirs
            if not _is_complete(
                run_dir, read_json(run_dir / "config.json")["fingerprint"]
            )
        )
    if return_code or incomplete:
        print(f"Training incomplete: {len(incomplete)} run(s) lack valid checkpoints.")
        return 1
    for sweep_dir, _ in prepared:
        run_dirs = _completed_manifest_training_runs(sweep_dir)
        aggregate_csv(
            run_dirs,
            Path("training_history.csv"),
            sweep_dir / "summary" / "training_curves.csv",
        )
        print(f"Training artifacts: {sweep_dir}")
    if not args.skip_eval:
        for sweep_dir, _ in prepared:
            eval_code = _launch_evaluation(
                sweep_dir,
                # Evaluate only this invocation's selected methods; the eval
                # runner merges every already-complete manifest artifact into
                # the shared summary afterward.
                methods=args.methods,
                devices=args.devices,
                max_per_device=args.max_per_device,
                force=args.force,
            )
            if eval_code:
                return eval_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
