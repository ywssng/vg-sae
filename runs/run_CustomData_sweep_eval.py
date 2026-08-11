"""Evaluate saved Stage-1 custom-baseline checkpoints in parallel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs._sweep_io import (  # noqa: E402
    manifest_run_dirs,
    read_json,
    read_rows,
    resolve_devices,
    runtime_provenance,
    utc_now,
    write_json,
    write_rows,
)
from runs.gpu_scheduler import ParallelExecutor, ScriptTask  # noqa: E402
from src.sae_sweep import (  # noqa: E402
    METHOD_ORDER,
    RunSpec,
    SweepConfig,
    default_sweep_dir,
    default_sweep_config,
    load_checkpoint,
    make_train_test,
)
from src.sae_sweep_eval import evaluate_model  # noqa: E402


EVAL_SOURCE_FILES = (
    "runs/_sweep_io.py",
    "runs/run_CustomData_sweep_eval.py",
    "src/evaluate.py",
    "src/sae_baselines.py",
    "src/sae_data.py",
    "src/sae_model.py",
    "src/sae_sweep.py",
    "src/sae_sweep_eval.py",
)
CHECKPOINT_KINDS = ("last", "best")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument(
        "--checkpoint",
        choices=CHECKPOINT_KINDS,
        help="Evaluate one checkpoint only; the default evaluates both last and best.",
    )
    parser.add_argument("--methods", default="all", help="Comma-separated methods, or all.")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3", help="auto, cpu, or cuda:0,cuda:1,...")
    parser.add_argument("--max-per-device", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    return parser.parse_args()


def _checkpoint_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _eval_dir(run_dir: Path, checkpoint_kind: str) -> Path:
    return run_dir / "eval" / checkpoint_kind


def _checkpoint_kinds(requested: str | None) -> tuple[str, ...]:
    return CHECKPOINT_KINDS if requested is None else (requested,)


def _training_ready(run_dir: Path, checkpoint_kind: str) -> bool:
    config_path = run_dir / "config.json"
    status_path = run_dir / "train_status.json"
    checkpoint = run_dir / "checkpoints" / f"{checkpoint_kind}.pt"
    if not config_path.exists() or not status_path.exists() or not checkpoint.exists():
        return False
    bundle = read_json(config_path)
    status = read_json(status_path)
    return (
        status.get("state") == "complete"
        and status.get("fingerprint") == bundle.get("fingerprint")
        and (run_dir / "training_history.csv").exists()
        and (run_dir / "training_summary.json").exists()
    )


def _is_complete(run_dir: Path, checkpoint_kind: str, eval_fingerprint: str) -> bool:
    checkpoint = run_dir / "checkpoints" / f"{checkpoint_kind}.pt"
    status_path = _eval_dir(run_dir, checkpoint_kind) / "status.json"
    if not _training_ready(run_dir, checkpoint_kind) or not status_path.exists():
        return False
    status = read_json(status_path)
    return (
        status.get("state") == "complete"
        and status.get("checkpoint") == _checkpoint_identity(checkpoint)
        and status.get("eval_fingerprint") == eval_fingerprint
        and (_eval_dir(run_dir, checkpoint_kind) / "metrics.json").exists()
        and (_eval_dir(run_dir, checkpoint_kind) / "cache.npz").exists()
    )


def evaluate_one(run_dir: Path, checkpoint_kind: str, device: str, force: bool) -> None:
    provenance = runtime_provenance(PROJECT_ROOT, EVAL_SOURCE_FILES)
    eval_fingerprint = provenance["pipeline_fingerprint"]
    if not force and _is_complete(run_dir, checkpoint_kind, eval_fingerprint):
        print(f"Skipping complete evaluation: {run_dir.name} [{checkpoint_kind}]")
        return
    if not _training_ready(run_dir, checkpoint_kind):
        raise RuntimeError(f"Training is not complete or current for {run_dir}.")
    bundle = read_json(run_dir / "config.json")
    config = SweepConfig.from_dict(bundle["sweep_config"])
    spec = RunSpec.from_dict(bundle["spec"])
    checkpoint_path = run_dir / "checkpoints" / f"{checkpoint_kind}.pt"
    identity = _checkpoint_identity(checkpoint_path)
    destination = _eval_dir(run_dir, checkpoint_kind)
    write_json(
        destination / "status.json",
        {
            "state": "running",
            "checkpoint": identity,
            "eval_fingerprint": eval_fingerprint,
            "eval_device": device,
            "started_at": utc_now(),
        },
    )

    model, payload = load_checkpoint(checkpoint_path, device)
    checkpoint_metadata = payload.get("metadata", {})
    checkpoint_config = SweepConfig.from_dict(payload["sweep_config"])
    if (
        RunSpec.from_dict(payload["run_spec"]) != spec
        or checkpoint_config.to_dict() != config.to_dict()
        or payload.get("checkpoint_kind") != checkpoint_kind
        or checkpoint_metadata.get("train_fingerprint") != bundle["fingerprint"]
    ):
        raise ValueError(f"Checkpoint metadata does not match {run_dir / 'config.json'}.")
    train_data, test_data = make_train_test(config, spec.seed, device)
    row, cache = evaluate_model(model, train_data, test_data, config, spec, spec.run_id)
    row.update(
        checkpoint_kind=checkpoint_kind,
        checkpoint_step=payload.get("step"),
        checkpoint_training_loss=payload.get("loss"),
        train_device=checkpoint_metadata.get("train_device"),
        eval_device=device,
        train_source_fingerprint=checkpoint_metadata["train_provenance"][
            "source_fingerprint"
        ],
        train_pipeline_fingerprint=checkpoint_metadata["train_provenance"][
            "pipeline_fingerprint"
        ],
        eval_source_fingerprint=provenance["source_fingerprint"],
    )
    write_json(destination / "metrics.json", row)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "cache.npz", **cache)
    write_json(
        destination / "status.json",
        {
            "state": "complete",
            "checkpoint": identity,
            "eval_fingerprint": eval_fingerprint,
            "eval_device": device,
            "eval_provenance": provenance,
            "finished_at": utc_now(),
        },
    )


def _selected_run_dirs(sweep_dir: Path, methods: str) -> list[Path]:
    run_dirs = manifest_run_dirs(sweep_dir)
    if methods == "all":
        return run_dirs
    requested = {value.strip().lower() for value in methods.split(",") if value.strip()}
    if not requested:
        raise ValueError("--methods must name at least one method or use 'all'.")
    selected = []
    for run_dir in run_dirs:
        spec = RunSpec.from_dict(read_json(run_dir / "config.json")["spec"])
        if spec.method in requested:
            selected.append(run_dir)
    present = {
        RunSpec.from_dict(read_json(path / "config.json")["spec"]).method
        for path in selected
    }
    if missing := requested - present:
        raise ValueError(f"Methods are absent from the manifest: {', '.join(sorted(missing))}")
    return selected


def _metric_sort_key(row: dict) -> tuple:
    rank = METHOD_ORDER.index(row["method"])
    rho = float(row["rho_model"]) if row.get("rho_model") is not None else float("inf")
    return rank, rho, int(row["seed"]), float(row["control_value"])


def _aggregate(
    sweep_dir: Path,
    run_dirs: list[Path],
    history_run_dirs: list[Path],
    checkpoint_kind: str,
) -> None:
    import pandas as pd

    summary_dir = sweep_dir / "summary"
    metric_rows = [
        read_json(_eval_dir(run_dir, checkpoint_kind) / "metrics.json")
        for run_dir in run_dirs
    ]
    metric_rows.sort(key=_metric_sort_key)
    train_fingerprints = {row["train_source_fingerprint"] for row in metric_rows}
    train_pipeline_fingerprints = {
        row["train_pipeline_fingerprint"] for row in metric_rows
    }
    if len(train_fingerprints) != 1 or len(train_pipeline_fingerprints) != 1:
        raise ValueError(
            "Refusing to aggregate runs trained by different source or package versions. "
            "Use a new sweep directory or retrain every method with --force."
        )
    checkpoint_summary = summary_dir / checkpoint_kind
    write_rows(checkpoint_summary / "final_metrics.csv", metric_rows)
    eval_status = read_json(_eval_dir(run_dirs[0], checkpoint_kind) / "status.json")
    write_json(
        checkpoint_summary / "summary.json",
        {
            "checkpoint_kind": checkpoint_kind,
            "n_evaluated_runs": len(run_dirs),
            "methods": sorted({row["method"] for row in metric_rows}),
            "train_source_fingerprint": next(iter(train_fingerprints)),
            "train_pipeline_fingerprint": next(iter(train_pipeline_fingerprints)),
            "eval_fingerprint": eval_status["eval_fingerprint"],
            "eval_provenance": eval_status["eval_provenance"],
            "generated_at": utc_now(),
        },
    )

    metrics = pd.DataFrame(metric_rows)
    data_axes = [
        "input_dim",
        "ground_truth_num_features",
        "sae_width",
        "support_density",
    ]
    groups = [
        *[column for column in data_axes if column in metrics],
        "method",
        "method_label",
        "control_name",
        "control_value",
    ]
    numeric = [
        column
        for column in metrics.select_dtypes(include=np.number).columns
        if column not in {"seed", "control_value"}
    ]
    means = (
        metrics.groupby(groups, as_index=False)
        .agg({**{column: "mean" for column in numeric}, "seed": "nunique"})
        .rename(columns={"seed": "n_seeds"})
    )
    means["_method_order"] = means["method"].map({name: i for i, name in enumerate(METHOD_ORDER)})
    means = means.sort_values(["_method_order", "rho_model", "control_value"], kind="stable")
    means = means.drop(columns="_method_order")
    write_rows(checkpoint_summary / "final_metrics_seed_mean.csv", means.to_dict("records"))

    histories = [
        row
        for run_dir in history_run_dirs
        for row in read_rows(run_dir / "training_history.csv")
    ]
    histories.sort(
        key=lambda row: (
            METHOD_ORDER.index(row["method"]),
            row["run_id"],
            float(row["step"]),
        )
    )
    write_rows(summary_dir / "training_curves.csv", histories)

    first_bundle = min(
        (read_json(run_dir / "config.json") for run_dir in run_dirs),
        key=lambda item: (item["spec"]["seed"], item["spec"]["method"]),
    )
    config = SweepConfig.from_dict(first_bundle["sweep_config"])
    seed = int(first_bundle["spec"]["seed"])
    train_data, _ = make_train_test(config, seed, "cpu")
    summary_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        summary_dir / "data_preview.npz",
        feature_probabilities=train_data.feature_probabilities.cpu().numpy(),
        dictionary=train_data.dictionary.cpu().numpy(),
        z0=train_data.z[0].cpu().numpy(),
        input_dim=config.data.input_dim,
        ground_truth_num_features=config.data.ground_truth_num_features,
        sae_width=config.data.sae_width,
        support_density=config.data.support_density,
        frequency_skew=config.data.frequency_skew,
        amplitude_scale=config.data.amplitude_scale,
        target_model_density=float(
            train_data.feature_probabilities.sum() / config.data.sae_width
        ),
    )


def main(args: argparse.Namespace) -> int:
    if args.worker:
        if args.run_dir is None or args.checkpoint is None:
            raise ValueError("--worker requires --run-dir and --checkpoint.")
        destination = _eval_dir(args.run_dir, args.checkpoint)
        try:
            evaluate_one(args.run_dir.resolve(), args.checkpoint, args.device, args.force)
        except BaseException as error:
            write_json(
                destination / "status.json",
                {"state": "failed", "failed_at": utc_now(), "error": repr(error)},
            )
            raise
        return 0

    default_config = default_sweep_config(args.fast_dev_run)
    default_dir = default_sweep_dir(PROJECT_ROOT, default_config)
    sweep_dir = (args.sweep_dir or default_dir).resolve()
    run_dirs = _selected_run_dirs(sweep_dir, args.methods)
    checkpoint_kinds = _checkpoint_kinds(args.checkpoint)
    run_checkpoints = [
        (run_dir, checkpoint_kind)
        for run_dir in run_dirs
        for checkpoint_kind in checkpoint_kinds
    ]
    invalid = [
        (run_dir, checkpoint_kind)
        for run_dir, checkpoint_kind in run_checkpoints
        if not _training_ready(run_dir, checkpoint_kind)
    ]
    if invalid:
        examples = ", ".join(
            f"{path} [{kind}]" for path, kind in invalid[:3]
        )
        raise RuntimeError(
            "Training is incomplete or stale for "
            f"{len(invalid)} run/checkpoint pair(s): {examples}"
        )

    provenance = runtime_provenance(PROJECT_ROOT, EVAL_SOURCE_FILES)
    eval_fingerprint = provenance["pipeline_fingerprint"]

    tasks = [
        ScriptTask(
            Path(__file__).resolve(),
            (
                "--worker",
                f"--run-dir={run_dir}",
                f"--checkpoint={checkpoint_kind}",
                *(('--force',) if args.force else ()),
            ),
            f"{run_dir.name}[{checkpoint_kind}]",
        )
        for run_dir, checkpoint_kind in run_checkpoints
        if args.force
        or not _is_complete(run_dir, checkpoint_kind, eval_fingerprint)
    ]
    return_code = ParallelExecutor(
        tasks,
        resolve_devices(args.devices),
        max_per_device=args.max_per_device,
    ).run_all()
    incomplete = [
        (path, checkpoint_kind)
        for path, checkpoint_kind in run_checkpoints
        if not _is_complete(path, checkpoint_kind, eval_fingerprint)
    ]
    if return_code or incomplete:
        print(
            "Evaluation incomplete: "
            f"{len(incomplete)} run/checkpoint pair(s) are missing valid artifacts."
        )
        return 1
    all_run_dirs = manifest_run_dirs(sweep_dir)
    for checkpoint_kind in checkpoint_kinds:
        evaluated_run_dirs = [
            path
            for path in all_run_dirs
            if _is_complete(path, checkpoint_kind, eval_fingerprint)
        ]
        history_run_dirs = [
            path
            for path in all_run_dirs
            if _training_ready(path, checkpoint_kind)
        ]
        _aggregate(
            sweep_dir,
            evaluated_run_dirs,
            history_run_dirs,
            checkpoint_kind,
        )
        print(f"Evaluation summary: {sweep_dir / 'summary' / checkpoint_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
