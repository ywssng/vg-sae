"""Train the Stage-1 custom-baseline SAE sweep as parallel jobs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
from src.sae_sweep import (  # noqa: E402
    CONTROL_NAMES,
    METHOD_LABELS,
    METHOD_ORDER,
    RunSpec,
    SweepConfig,
    build_model,
    build_specs,
    default_sweep_config,
    make_train_test,
    run_directory,
    save_checkpoint,
)
from src.sae_train import fit_sae  # noqa: E402
from src.utils import set_seed  # noqa: E402


WANDB_PROJECT = "vg-sae"
WANDB_API_KEY = "paste-your-wandb-api-key-here"
TRAIN_SOURCE_FILES = (
    "runs/_sweep_io.py",
    "runs/run_saes_sweep.py",
    "src/sae_baselines.py",
    "src/sae_data.py",
    "src/sae_loss.py",
    "src/sae_model.py",
    "src/sae_sweep.py",
    "src/sae_train.py",
    "src/saelens_vg.py",
    "src/utils.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Serialized SweepConfig JSON.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--methods", default="all", help="Comma-separated methods, or all.")
    parser.add_argument("--input-dim", type=int)
    parser.add_argument("--ground-truth-num-features", type=int)
    parser.add_argument("--sae-width", type=int)
    parser.add_argument("--support-density", type=float)
    seed = parser.add_mutually_exclusive_group()
    seed.add_argument("--seed", type=int, help="Run one data/initialization seed.")
    seed.add_argument("--seeds", help="Comma-separated seed sweep.")
    parser.add_argument(
        "--model-sparsity-control",
        "--sparsity-control",
        dest="sparsity_controls",
        action="append",
        metavar="METHOD=V1,V2,...",
        help="Replace one method's sparsity-control grid; repeat per method.",
    )
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--history-every", type=int)
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:2,cuda:3",
        help="auto, cpu, or cuda:0,cuda:1,...",
    )
    parser.add_argument("--max-per-device", type=int, default=16)
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


def configured_sweep(args: argparse.Namespace) -> SweepConfig:
    base = (
        read_json(args.config)
        if args.config is not None
        else default_sweep_config(fast=args.fast_dev_run).to_dict()
    )
    raw = SweepConfig.from_dict(base).to_dict()
    for name in (
        "input_dim",
        "ground_truth_num_features",
        "sae_width",
        "support_density",
    ):
        value = getattr(args, name, None)
        if value is not None:
            raw["data"][name] = value

    seed = getattr(args, "seed", None)
    seeds = getattr(args, "seeds", None)
    if seed is not None:
        raw["seeds"] = [seed]
    elif seeds:
        raw["seeds"] = [int(value) for value in seeds.split(",")]

    overrides = _control_overrides(getattr(args, "sparsity_controls", None) or [])
    raw["controls"].update(overrides)
    width_changed = getattr(args, "sae_width", None) is not None
    if width_changed and args.config is None:
        width = raw["data"]["sae_width"]
        if "topk" not in overrides and "topk" in raw["controls"]:
            raw["controls"]["topk"] = (
                sorted({1, min(2, width)})
                if args.fast_dev_run
                else list(range(1, width + 1))
            )
        if "batchtopk" not in overrides and "batchtopk" in raw["controls"]:
            valid = [value for value in raw["controls"]["batchtopk"] if value <= width]
            if not args.fast_dev_run:
                valid.append(float(width))
            raw["controls"]["batchtopk"] = sorted(set(valid or [float(width)]))

    if args.train_steps is not None:
        raw["training"]["train_steps"] = args.train_steps
    if args.history_every is not None:
        raw["training"]["history_every"] = args.history_every
    return SweepConfig.from_dict(raw)


def selected_specs(config: SweepConfig, methods: str) -> list[RunSpec]:
    specs = build_specs(config)
    if methods == "all":
        return specs
    requested = {value.strip().lower() for value in methods.split(",") if value.strip()}
    if not requested:
        raise ValueError("--methods must name at least one method or use 'all'.")
    available = {spec.method for spec in specs}
    if unknown := requested - available:
        raise ValueError(f"Unknown method(s): {', '.join(sorted(unknown))}")
    return [spec for spec in specs if spec.method in requested]


def _bundle(config: SweepConfig, spec: RunSpec, provenance: dict) -> dict:
    payload = {
        "format_version": 2,
        "sweep_config": config.to_dict(),
        "spec": spec.to_dict(),
        "checkpoint_selection": "lowest full-training objective at a history evaluation step",
        "train_provenance": provenance,
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
    checkpoints = run_dir / "checkpoints"
    return (
        status.get("state") == "complete"
        and status.get("fingerprint") == run_fingerprint
        and (checkpoints / "best.pt").exists()
        and (checkpoints / "last.pt").exists()
        and (run_dir / "training_history.csv").exists()
        and (run_dir / "training_summary.json").exists()
    )


def _wandb_run(bundle: dict, spec: RunSpec, run_dir: Path, mode: str):
    if mode == "disabled":
        return None
    import wandb

    if WANDB_API_KEY != "paste-your-wandb-api-key-here":
        os.environ.setdefault("WANDB_API_KEY", WANDB_API_KEY)
    elif mode == "online" and not os.getenv("WANDB_API_KEY"):
        try:
            existing_key = wandb.Api().api_key
        except Exception:
            existing_key = None
        if not existing_key:
            raise RuntimeError(
                "W&B is not authenticated. Set WANDB_API_KEY, edit WANDB_API_KEY in "
                "this script, or pass --no-wandb."
            )
    sweep_dir = next(
        (parent for parent in run_dir.parents if (parent / "manifest.json").exists()),
        run_dir.parent,
    )
    return wandb.init(
        project=bundle["sweep_config"].get("wandb_project", WANDB_PROJECT),
        name=spec.run_id,
        group=sweep_dir.name,
        config=bundle,
        mode=mode,
        dir=str(run_dir),
    )


def _preflight_wandb(mode: str) -> None:
    if mode in {"disabled", "offline"}:
        return
    import wandb

    if WANDB_API_KEY != "paste-your-wandb-api-key-here":
        os.environ.setdefault("WANDB_API_KEY", WANDB_API_KEY)
    if os.getenv("WANDB_API_KEY"):
        return
    try:
        existing_key = wandb.Api().api_key
    except Exception:
        existing_key = None
    if not existing_key:
        raise RuntimeError(
            "W&B is not authenticated. Set WANDB_API_KEY, edit WANDB_API_KEY in "
            "this script, or pass --no-wandb."
        )


def _run_metadata(
    config: SweepConfig, spec: RunSpec, device: str, provenance: dict
) -> dict:
    data = config.data
    return {
        "run_id": spec.run_id,
        "seed": spec.seed,
        "init_seed": spec.init_seed,
        "input_dim": data.input_dim,
        "ground_truth_num_features": data.ground_truth_num_features,
        "sae_width": data.sae_width,
        "support_density": data.support_density,
        "method": spec.method,
        "method_label": METHOD_LABELS[spec.method],
        "control_name": spec.control_name,
        "control_value": spec.control_value,
        "train_device": device,
        "train_source_fingerprint": provenance["source_fingerprint"],
        "train_pipeline_fingerprint": provenance["pipeline_fingerprint"],
    }


def train_one(run_dir: Path, device: str, wandb_mode: str, force: bool) -> None:
    bundle = read_json(run_dir / "config.json")
    config = SweepConfig.from_dict(bundle["sweep_config"])
    spec = RunSpec.from_dict(bundle["spec"])
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
    checkpoints = run_dir / "checkpoints"
    for checkpoint_kind in ("best", "last"):
        (checkpoints / f"{checkpoint_kind}.pt").unlink(missing_ok=True)
    wandb_run = _wandb_run(bundle, spec, run_dir, wandb_mode)
    try:
        train_data, _ = make_train_test(config, spec.seed, device)
        set_seed(spec.init_seed)
        model = build_model(config, spec)
        training = config.training
        callback = None
        if wandb_run is not None:
            callback = lambda row: wandb_run.log(row, step=int(row["step"]))
        result = fit_sae(
            model,
            train_data.x,
            lr=training.lr,
            batch_size=training.batch_size,
            max_steps=training.train_steps,
            gradient_clip_norm=training.gradient_clip_norm,
            history_every=training.history_every,
            dead_feature_window=training.dead_feature_window,
            seed=spec.init_seed,
            history_callback=callback,
        )

        checkpoint_metadata = {
            "train_fingerprint": bundle["fingerprint"],
            "train_device": device,
            "train_provenance": bundle["train_provenance"],
        }
        save_checkpoint(
            checkpoints / "best.pt",
            model=result.model,
            config=config,
            spec=spec,
            checkpoint_kind="best",
            state_dict=result.best_state_dict,
            step=result.best_step,
            loss=result.best_loss,
            metadata=checkpoint_metadata,
        )
        save_checkpoint(
            checkpoints / "last.pt",
            model=result.model,
            config=config,
            spec=spec,
            checkpoint_kind="last",
            step=training.train_steps - 1,
            loss=result.history[-1]["loss"],
            metadata=checkpoint_metadata,
        )
        metadata = _run_metadata(
            config,
            spec,
            device,
            bundle["train_provenance"],
        )
        write_rows(
            run_dir / "training_history.csv",
            ({**metadata, **row} for row in result.history),
        )
        summary = {
            **metadata,
            "best_step": result.best_step,
            "best_loss": result.best_loss,
            "last_step": training.train_steps - 1,
            "last_loss": result.history[-1]["loss"],
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


def prepare_runs(
    sweep_dir: Path,
    config: SweepConfig,
    specs: list[RunSpec],
    force: bool,
    wandb_mode: str,
) -> tuple[list[ScriptTask], list[Path]]:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config.to_dict()
    manifest_path = sweep_dir / "manifest.json"
    entries: dict[str, dict] = {}
    provenance = runtime_provenance(PROJECT_ROOT, TRAIN_SOURCE_FILES)
    if manifest_path.exists():
        previous = read_json(manifest_path)
        previous_config = SweepConfig.from_dict(previous["sweep_config"]).to_dict()
        if fingerprint(previous_config) != fingerprint(config_dict):
            raise ValueError("Output directory contains a different sweep configuration.")
        entries = {entry["run_id"]: entry for entry in previous["runs"]}

    tasks = []
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
            "format_version": 2,
            "sweep_config": config_dict,
            "runs": ordered_entries,
        },
    )
    write_json(sweep_dir / "sweep_config.json", config_dict)
    all_run_dirs = [sweep_dir / entry["relative_dir"] for entry in ordered_entries]
    return tasks, all_run_dirs


def main(args: argparse.Namespace) -> int:
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
    default_dir = PROJECT_ROOT / "outputs" / "runs" / config.experiment_name
    sweep_dir = (args.output_dir or default_dir).resolve()
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
