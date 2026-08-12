"""Stream-evaluate saved Stage-2 SynthSAEBench checkpoints in parallel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
from runs.gpu_scheduler import (  # noqa: E402
    ParallelExecutor,
    ScriptTask,
    activate_worker_device,
)
from src.sae_sweep import METHOD_ORDER  # noqa: E402
from src.synthsaebench_eval import evaluate_model  # noqa: E402
from src.synthsaebench_sweep import (  # noqa: E402
    DEFAULT_MAX_PER_DEVICE,
    SynthSAEBenchRunSpec,
    SynthSAEBenchSweepConfig,
    default_sweep_config,
    default_sweep_dir,
    load_benchmark_model,
    load_checkpoint,
    temporary_seed_for_device,
)


EVAL_SOURCE_FILES = (
    "runs/_sweep_io.py",
    "runs/gpu_scheduler.py",
    "runs/run_SynthSAEBench_sweep_eval.py",
    "src/model.py",
    "src/sae_baselines.py",
    "src/sae_model.py",
    "src/sae_sweep.py",
    "src/saelens_vg.py",
    "src/synthsaebench_eval.py",
    "src/synthsaebench_sweep.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--calibration-grid", action="store_true")
    parser.add_argument("--methods", default="all", help="Comma-separated methods, or all.")
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:2,cuda:3",
        help="auto, cpu, or cuda:0,cuda:1,...",
    )
    parser.add_argument(
        "--max-per-device",
        type=int,
        default=DEFAULT_MAX_PER_DEVICE,
        help="Concurrent workers per device (default: benchmarked value 2).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    return parser.parse_args()


def _checkpoint_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _eval_dir(run_dir: Path) -> Path:
    return run_dir / "eval" / "last"


def _training_ready(run_dir: Path) -> bool:
    config_path = run_dir / "config.json"
    status_path = run_dir / "train_status.json"
    checkpoint = run_dir / "checkpoints" / "last.pt"
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


def _is_complete(run_dir: Path, eval_fingerprint: str) -> bool:
    checkpoint = run_dir / "checkpoints" / "last.pt"
    status_path = _eval_dir(run_dir) / "status.json"
    if not _training_ready(run_dir) or not status_path.exists():
        return False
    status = read_json(status_path)
    return (
        status.get("state") == "complete"
        and status.get("checkpoint") == _checkpoint_identity(checkpoint)
        and status.get("eval_fingerprint") == eval_fingerprint
        and (_eval_dir(run_dir) / "metrics.json").exists()
        and (_eval_dir(run_dir) / "cache.npz").exists()
    )


def evaluate_one(run_dir: Path, device: str, force: bool) -> None:
    activate_worker_device(device)
    provenance = runtime_provenance(PROJECT_ROOT, EVAL_SOURCE_FILES)
    eval_fingerprint = provenance["pipeline_fingerprint"]
    if not force and _is_complete(run_dir, eval_fingerprint):
        print(f"Skipping complete evaluation: {run_dir.name}")
        return
    if not _training_ready(run_dir):
        raise RuntimeError(f"Training is not complete or current for {run_dir}.")
    bundle = read_json(run_dir / "config.json")
    config = SynthSAEBenchSweepConfig.from_dict(bundle["sweep_config"])
    spec = SynthSAEBenchRunSpec.from_dict(bundle["spec"])
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    identity = _checkpoint_identity(checkpoint_path)
    destination = _eval_dir(run_dir)
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
    if (
        SynthSAEBenchRunSpec.from_dict(payload["run_spec"]) != spec
        or SynthSAEBenchSweepConfig.from_dict(payload["sweep_config"]).to_dict()
        != config.to_dict()
        or payload.get("checkpoint_kind") != "last"
        or checkpoint_metadata.get("train_fingerprint") != bundle["fingerprint"]
    ):
        raise ValueError(f"Checkpoint metadata does not match {run_dir / 'config.json'}.")
    synthetic, _ = load_benchmark_model(config, device)
    row, cache = evaluate_model(model, synthetic, config, spec)
    row.update(
        checkpoint_kind="last",
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
        spec = SynthSAEBenchRunSpec.from_dict(
            read_json(run_dir / "config.json")["spec"]
        )
        if spec.method in requested:
            selected.append(run_dir)
    present = {
        SynthSAEBenchRunSpec.from_dict(
            read_json(path / "config.json")["spec"]
        ).method
        for path in selected
    }
    if missing := requested - present:
        raise ValueError(f"Methods are absent from the manifest: {', '.join(sorted(missing))}")
    return selected


def _metric_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    rank = METHOD_ORDER.index(row["method"])
    rho = float(row["rho_model"]) if row.get("rho_model") is not None else float("inf")
    return rank, rho, int(row["seed"]), float(row["control_value"])


def _write_data_preview(
    sweep_dir: Path,
    config: SynthSAEBenchSweepConfig,
    spec: SynthSAEBenchRunSpec,
    metric_rows: list[dict[str, Any]],
) -> None:
    summary_dir = sweep_dir / "summary"
    synthetic, _ = load_benchmark_model(config, "cpu")
    preview_width = min(512, config.data.ground_truth_num_features)
    indices = np.arange(preview_width, dtype=np.int64)
    with temporary_seed_for_device(spec.eval_stream_seed, "cpu"):
        _, features = synthetic.sample_with_features(1)
    empirical_true_l0 = float(
        np.mean([float(row["true_l0"]) for row in metric_rows])
    )
    np.savez_compressed(
        summary_dir / "data_preview.npz",
        feature_probabilities=(
            synthetic.activation_generator.firing_probabilities.detach().cpu().numpy()
        ),
        dictionary=(
            synthetic.feature_dict.feature_vectors[:preview_width]
            .detach()
            .cpu()
            .numpy()
            .T
        ),
        z0=features[0, :preview_width].detach().cpu().numpy(),
        preview_feature_indices=indices,
        input_dim=config.data.input_dim,
        ground_truth_num_features=config.data.ground_truth_num_features,
        sae_width=config.data.sae_width,
        empirical_true_l0=empirical_true_l0,
        target_model_density=empirical_true_l0 / config.data.sae_width,
        data_kind=config.data.kind,
        probability_semantics="pre_hierarchy_base_probability",
    )


def _aggregate(
    sweep_dir: Path,
    run_dirs: list[Path],
    history_run_dirs: list[Path],
) -> None:
    import pandas as pd

    summary_dir = sweep_dir / "summary"
    metric_rows = [
        read_json(_eval_dir(run_dir) / "metrics.json") for run_dir in run_dirs
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
    checkpoint_summary = summary_dir / "last"
    write_rows(checkpoint_summary / "final_metrics.csv", metric_rows)
    eval_status = read_json(_eval_dir(run_dirs[0]) / "status.json")
    write_json(
        checkpoint_summary / "summary.json",
        {
            "checkpoint_kind": "last",
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
    groups = ["method", "method_label", "control_name", "control_value"]
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
    means["_method_order"] = means["method"].map(
        {name: index for index, name in enumerate(METHOD_ORDER)}
    )
    means = means.sort_values(
        ["_method_order", "rho_model", "control_value"], kind="stable"
    ).drop(columns="_method_order")
    write_rows(
        checkpoint_summary / "final_metrics_seed_mean.csv",
        means.to_dict("records"),
    )

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
    config = SynthSAEBenchSweepConfig.from_dict(first_bundle["sweep_config"])
    spec = SynthSAEBenchRunSpec.from_dict(first_bundle["spec"])
    _write_data_preview(sweep_dir, config, spec, metric_rows)


def main(args: argparse.Namespace) -> int:
    if args.worker:
        if args.run_dir is None:
            raise ValueError("--worker requires --run-dir.")
        destination = _eval_dir(args.run_dir)
        try:
            evaluate_one(args.run_dir.resolve(), args.device, args.force)
        except BaseException as error:
            write_json(
                destination / "status.json",
                {"state": "failed", "failed_at": utc_now(), "error": repr(error)},
            )
            raise
        return 0

    default_config = default_sweep_config(
        args.fast_dev_run, calibration=args.calibration_grid
    )
    sweep_dir = (
        args.sweep_dir or default_sweep_dir(PROJECT_ROOT, default_config)
    ).resolve()
    run_dirs = _selected_run_dirs(sweep_dir, args.methods)
    invalid = [run_dir for run_dir in run_dirs if not _training_ready(run_dir)]
    if invalid:
        examples = ", ".join(str(path) for path in invalid[:3])
        raise RuntimeError(
            f"Training is incomplete or stale for {len(invalid)} run(s): {examples}"
        )

    provenance = runtime_provenance(PROJECT_ROOT, EVAL_SOURCE_FILES)
    eval_fingerprint = provenance["pipeline_fingerprint"]
    tasks = [
        ScriptTask(
            Path(__file__).resolve(),
            (
                "--worker",
                f"--run-dir={run_dir}",
                *(("--force",) if args.force else ()),
            ),
            run_dir.name,
        )
        for run_dir in run_dirs
        if args.force or not _is_complete(run_dir, eval_fingerprint)
    ]
    return_code = ParallelExecutor(
        tasks,
        resolve_devices(args.devices),
        max_per_device=args.max_per_device,
    ).run_all()
    incomplete = [
        path for path in run_dirs if not _is_complete(path, eval_fingerprint)
    ]
    if return_code or incomplete:
        print(f"Evaluation incomplete: {len(incomplete)} run(s) lack valid artifacts.")
        return 1
    all_run_dirs = manifest_run_dirs(sweep_dir)
    evaluated_run_dirs = [
        path for path in all_run_dirs if _is_complete(path, eval_fingerprint)
    ]
    history_run_dirs = [path for path in all_run_dirs if _training_ready(path)]
    _aggregate(sweep_dir, evaluated_run_dirs, history_run_dirs)
    print(f"Evaluation summary: {sweep_dir / 'summary' / 'last'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
