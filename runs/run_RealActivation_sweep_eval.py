"""Evaluate completed Stage-3 real-activation SAE runs in parallel.

Each worker reloads both the final native SAELens checkpoint and the pinned
language model.  Activation-space metrics consume the complete held-out token
range, while the more expensive language-model intervention metrics consume a
smaller, explicitly recorded prefix of that same held-out range.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from sae_lens.evals import get_recons_loss
from sae_lens.training.activation_scaler import ActivationScaler

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
    validate_beta_modes,
    write_json,
    write_rows,
)
from runs.gpu_scheduler import (  # noqa: E402
    ParallelExecutor,
    ScriptTask,
    activate_worker_device,
)
from src.real_activation_eval import evaluate_model  # noqa: E402
from src.real_activation_plot import plot_all  # noqa: E402
from src.real_activation_sweep import (  # noqa: E402
    STAGE3_METHOD_ORDER,
    RealActivationRunSpec,
    RealActivationSweepConfig,
    augment_stage3_runtime_provenance,
    default_sweep_config,
    default_sweep_dir,
    load_checkpoint,
)
from src.real_activations import (  # noqa: E402
    iter_token_batches,
    load_real_language_model,
    make_live_activation_provider,
    model_input_device,
)
from src.sae_baselines import to_inference_sae  # noqa: E402


DEFAULT_MAX_PER_DEVICE = 1
EVAL_SOURCE_FILES = (
    "runs/_sweep_io.py",
    "runs/gpu_scheduler.py",
    "runs/run_RealActivation_sweep_eval.py",
    "src/real_activation_eval.py",
    "src/real_activation_plot.py",
    "src/real_activation_sweep.py",
    "src/real_activations.py",
    "src/model.py",
    "src/sae_baselines.py",
    "src/sae_evaluate.py",
    "src/sae_model.py",
    "src/saelens_vg.py",
)
_DOWNSTREAM_NAMES = (
    "ce_loss_with_sae",
    "ce_loss_without_sae",
    "ce_loss_with_ablation",
    "kl_div_with_sae",
    "kl_div_with_ablation",
)


def _stage3_runtime_provenance() -> dict[str, Any]:
    return augment_stage3_runtime_provenance(
        runtime_provenance(PROJECT_ROOT, EVAL_SOURCE_FILES)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument(
        "--target",
        default=None,
        help="Target used only to resolve the default sweep directory.",
    )
    parser.add_argument("--methods", default="all", help="Comma-separated methods, or all.")
    parser.add_argument(
        "--devices",
        default="auto",
        help="auto, cpu, or cuda:0,cuda:1,...",
    )
    parser.add_argument(
        "--max-per-device",
        type=int,
        default=DEFAULT_MAX_PER_DEVICE,
        help="Concurrent workers per device (default: one 2B-model worker).",
    )
    parser.add_argument(
        "--decoder-pairwise-block-size",
        type=int,
        default=256,
        help="Memory-bounded exact decoder-cosine block size.",
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


def _mean_finite(value: Any) -> tuple[float, int]:
    tensor = torch.as_tensor(value).detach().float().cpu().reshape(-1)
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() != tensor.numel():
        raise ValueError("Downstream evaluation produced a non-finite loss.")
    return float(finite.double().sum()), int(finite.numel())


def _safe_normalized_score(numerator: float, denominator: float) -> float | None:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or abs(denominator) <= 1.0e-12
    ):
        return None
    return numerator / denominator


@torch.no_grad()
def _evaluate_downstream(
    training_model: Any,
    language_model: Any,
    config: RealActivationSweepConfig,
) -> dict[str, Any]:
    """Compute CE/KL intervention metrics on the held-out dataset prefix."""

    inference_sae = to_inference_sae(training_model, fold_decoder_norm=True)
    inference_sae.eval()
    input_device = model_input_device(language_model)
    sums = {name: 0.0 for name in _DOWNSTREAM_NAMES}
    counts = {name: 0 for name in _DOWNSTREAM_NAMES}
    n_input_tokens = 0
    n_contexts = 0

    batches = iter_token_batches(
        config.data,
        start_token=config.data.eval_token_offset,
        n_tokens=config.data.n_downstream_eval_tokens,
        prompt_batch_size=config.training.eval_store_batch_size_prompts,
        device=input_device,
    )
    for tokens in batches:
        n_input_tokens += int(tokens.numel())
        n_contexts += int(tokens.shape[0])
        metrics = get_recons_loss(
            inference_sae,
            language_model,
            ActivationScaler(),
            tokens,
            compute_kl=True,
            compute_ce_loss=True,
            model_kwargs={"prepend_bos": False},
            hook_name=config.data.hook_name,
        )
        for name in _DOWNSTREAM_NAMES:
            if name not in metrics:
                raise ValueError(f"SAELens get_recons_loss omitted {name!r}.")
            batch_sum, batch_count = _mean_finite(metrics[name])
            sums[name] += batch_sum
            counts[name] += batch_count

    if n_input_tokens != config.data.n_downstream_eval_tokens:
        raise RuntimeError(
            "Downstream stream yielded "
            f"{n_input_tokens} tokens, expected {config.data.n_downstream_eval_tokens}."
        )
    means = {
        name: sums[name] / counts[name] if counts[name] else None
        for name in _DOWNSTREAM_NAMES
    }
    if any(means[name] is None for name in _DOWNSTREAM_NAMES):
        raise RuntimeError("Downstream evaluation produced no loss elements.")

    ce_sae = float(means["ce_loss_with_sae"])
    ce_original = float(means["ce_loss_without_sae"])
    ce_ablation = float(means["ce_loss_with_ablation"])
    kl_sae = float(means["kl_div_with_sae"])
    kl_ablation = float(means["kl_div_with_ablation"])
    ce_score = _safe_normalized_score(
        ce_ablation - ce_sae,
        ce_ablation - ce_original,
    )
    kl_ratio = _safe_normalized_score(kl_sae, kl_ablation)
    return {
        **means,
        "ce_loss_score": ce_score,
        "kl_div_score": None if kl_ratio is None else 1.0 - kl_ratio,
        "n_downstream_evaluation_tokens": n_input_tokens,
        "n_downstream_evaluation_contexts": n_contexts,
        "downstream_eval_start_token": config.data.eval_token_offset,
        "downstream_eval_end_token": (
            config.data.eval_token_offset + n_input_tokens
        ),
        "downstream_prepend_bos": config.data.prepend_bos,
        "downstream_added_bos": False,
    }


def evaluate_one(
    run_dir: Path,
    device: str,
    force: bool,
    *,
    decoder_pairwise_block_size: int = 256,
) -> None:
    activate_worker_device(device)
    provenance = _stage3_runtime_provenance()
    eval_fingerprint = provenance["pipeline_fingerprint"]
    if not force and _is_complete(run_dir, eval_fingerprint):
        print(f"Skipping complete evaluation: {run_dir.name}")
        return
    if not _training_ready(run_dir):
        raise RuntimeError(f"Training is not complete or current for {run_dir}.")

    bundle = read_json(run_dir / "config.json")
    config = RealActivationSweepConfig.from_dict(bundle["sweep_config"])
    spec = RealActivationRunSpec.from_dict(bundle["spec"])
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    checkpoint_identity = _checkpoint_identity(checkpoint_path)
    destination = _eval_dir(run_dir)
    write_json(
        destination / "status.json",
        {
            "state": "running",
            "checkpoint": checkpoint_identity,
            "eval_fingerprint": eval_fingerprint,
            "eval_device": device,
            "started_at": utc_now(),
        },
    )

    training_model, payload = load_checkpoint(checkpoint_path, device)
    checkpoint_metadata = payload.get("metadata", {})
    if (
        RealActivationRunSpec.from_dict(payload["run_spec"]) != spec
        or RealActivationSweepConfig.from_dict(payload["sweep_config"]).to_dict()
        != config.to_dict()
        or payload.get("checkpoint_kind") != "last"
        or checkpoint_metadata.get("train_fingerprint") != bundle["fingerprint"]
    ):
        raise ValueError(f"Checkpoint metadata does not match {run_dir / 'config.json'}.")

    language_model = load_real_language_model(config.data, device)
    eval_batch_size = math.gcd(
        config.training.batch_size,
        config.data.n_eval_tokens,
    )
    data_provider = make_live_activation_provider(
        config.data,
        language_model,
        start_token=config.data.eval_token_offset,
        total_tokens=config.data.n_eval_tokens,
        batch_size=eval_batch_size,
        prompt_batch_size=config.training.eval_store_batch_size_prompts,
        n_batches_in_buffer=1,
        activation_device=device,
        mix_fraction=0.0,
        seed=spec.eval_stream_seed,
        autocast_lm=config.training.autocast_data,
    )
    row, cache = evaluate_model(
        training_model,
        data_provider,
        config,
        spec,
        n_eval_tokens=config.data.n_eval_tokens,
        preview_tokens=config.training.preview_tokens,
        decoder_pairwise_block_size=decoder_pairwise_block_size,
    )
    row.update(_evaluate_downstream(training_model, language_model, config))

    train_provenance = checkpoint_metadata.get("train_provenance")
    if not isinstance(train_provenance, Mapping):
        raise ValueError("Checkpoint is missing metadata.train_provenance.")
    final_beta_precision = checkpoint_metadata.get("final_beta_precision")
    core = getattr(training_model, "core", None)
    log_beta = getattr(core, "log_beta", None)
    if spec.method == "vgsae" and log_beta is not None:
        final_beta_precision = float(log_beta.exp().detach().cpu())
    row.update(
        final_beta_precision=final_beta_precision,
        checkpoint_kind="last",
        checkpoint_step=payload.get("step"),
        checkpoint_training_tokens=payload.get("n_training_tokens"),
        checkpoint_training_loss=payload.get("loss"),
        train_device=checkpoint_metadata.get("train_device"),
        eval_device=device,
        activation_eval_start_token=config.data.eval_token_offset,
        activation_eval_end_token=(
            config.data.eval_token_offset + config.data.n_eval_tokens
        ),
        train_source_fingerprint=train_provenance["source_fingerprint"],
        train_pipeline_fingerprint=train_provenance["pipeline_fingerprint"],
        eval_source_fingerprint=provenance["source_fingerprint"],
        eval_pipeline_fingerprint=eval_fingerprint,
    )
    write_json(destination / "metrics.json", row)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "cache.npz", **cache)
    write_json(
        destination / "status.json",
        {
            "state": "complete",
            "checkpoint": checkpoint_identity,
            "eval_fingerprint": eval_fingerprint,
            "eval_device": device,
            "eval_provenance": provenance,
            "n_evaluation_tokens": config.data.n_eval_tokens,
            "n_downstream_evaluation_tokens": (
                config.data.n_downstream_eval_tokens
            ),
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
    unknown = requested - set(STAGE3_METHOD_ORDER)
    if unknown:
        raise ValueError(f"Unknown Stage-3 methods: {', '.join(sorted(unknown))}")
    selected = []
    for run_dir in run_dirs:
        spec = RealActivationRunSpec.from_dict(
            read_json(run_dir / "config.json")["spec"]
        )
        if spec.method in requested:
            selected.append(run_dir)
    present = {
        RealActivationRunSpec.from_dict(
            read_json(path / "config.json")["spec"]
        ).method
        for path in selected
    }
    if missing := requested - present:
        raise ValueError(f"Methods are absent from the manifest: {', '.join(sorted(missing))}")
    return selected


def _completed_run_dirs(
    sweep_dir: Path, eval_fingerprint: str
) -> list[Path]:
    """Return every current manifest run with a valid evaluation artifact."""

    return [
        run_dir
        for run_dir in manifest_run_dirs(sweep_dir)
        if _is_complete(run_dir, eval_fingerprint)
    ]


def _metric_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    rank = STAGE3_METHOD_ORDER.index(row["method"])
    rho = float(row["rho_model"]) if row.get("rho_model") is not None else float("inf")
    return rank, rho, int(row["seed"]), float(row["control_value"])


def _aggregate(
    sweep_dir: Path,
    run_dirs: Iterable[Path],
    history_run_dirs: Iterable[Path],
) -> None:
    import pandas as pd

    run_dirs = list(run_dirs)
    history_run_dirs = list(history_run_dirs)
    if not run_dirs:
        raise ValueError("No completed Stage-3 evaluations to aggregate.")
    metric_rows = [read_json(_eval_dir(run_dir) / "metrics.json") for run_dir in run_dirs]
    beta_mode = validate_beta_modes(
        (row["beta_mode"] for row in metric_rows),
        context="Stage-3 real-activation evaluation metrics",
    )
    metric_rows.sort(key=_metric_sort_key)
    train_sources = {row["train_source_fingerprint"] for row in metric_rows}
    train_pipelines = {row["train_pipeline_fingerprint"] for row in metric_rows}
    eval_pipelines = {row["eval_pipeline_fingerprint"] for row in metric_rows}
    if len(train_sources) != 1 or len(train_pipelines) != 1 or len(eval_pipelines) != 1:
        raise ValueError(
            "Refusing to aggregate runs produced by different source or package "
            "versions. Retrain/re-evaluate every selected run with --force."
        )

    checkpoint_summary = sweep_dir / "summary" / "last"
    write_rows(checkpoint_summary / "final_metrics.csv", metric_rows)
    metrics = pd.DataFrame(metric_rows)
    groups = [
        "target_name",
        "model_id",
        "model_revision",
        "layer",
        "hook_name",
        "method",
        "method_label",
        "beta_mode",
        "control_name",
        "control_value",
    ]
    numeric = [
        column
        for column in metrics.select_dtypes(include=np.number).columns
        if column not in {"seed", "control_value", "layer"}
    ]
    means = (
        metrics.groupby(groups, as_index=False, dropna=False)
        .agg({**{column: "mean" for column in numeric}, "seed": "nunique"})
        .rename(columns={"seed": "n_seeds"})
    )
    means["_method_order"] = means["method"].map(
        {name: index for index, name in enumerate(STAGE3_METHOD_ORDER)}
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
            STAGE3_METHOD_ORDER.index(row["method"]),
            row["run_id"],
            float(row["step"]),
        )
    )
    write_rows(sweep_dir / "summary" / "training_curves.csv", histories)

    figure_paths = plot_all(sweep_dir)
    first_status = read_json(_eval_dir(run_dirs[0]) / "status.json")
    write_json(
        checkpoint_summary / "summary.json",
        {
            "checkpoint_kind": "last",
            "n_evaluated_runs": len(run_dirs),
            "methods": sorted({row["method"] for row in metric_rows}),
            "targets": sorted({row["target_name"] for row in metric_rows}),
            "beta_mode": beta_mode,
            "train_source_fingerprint": next(iter(train_sources)),
            "train_pipeline_fingerprint": next(iter(train_pipelines)),
            "eval_fingerprint": next(iter(eval_pipelines)),
            "eval_provenance": first_status["eval_provenance"],
            "figures": [str(path.relative_to(sweep_dir)) for path in figure_paths],
            "generated_at": utc_now(),
        },
    )


def main(args: argparse.Namespace) -> int:
    if args.worker:
        if args.run_dir is None:
            raise ValueError("--worker requires --run-dir.")
        destination = _eval_dir(args.run_dir)
        try:
            evaluate_one(
                args.run_dir.resolve(),
                args.device,
                args.force,
                decoder_pairwise_block_size=args.decoder_pairwise_block_size,
            )
        except BaseException as error:
            write_json(
                destination / "status.json",
                {"state": "failed", "failed_at": utc_now(), "error": repr(error)},
            )
            raise
        return 0

    default_config = default_sweep_config(
        args.target or default_sweep_config().data.target_name
    )
    sweep_dir = (args.sweep_dir or default_sweep_dir(PROJECT_ROOT, default_config)).resolve()
    run_dirs = _selected_run_dirs(sweep_dir, args.methods)
    invalid = [run_dir for run_dir in run_dirs if not _training_ready(run_dir)]
    if invalid:
        examples = ", ".join(str(path) for path in invalid[:3])
        raise RuntimeError(
            f"Training is incomplete or stale for {len(invalid)} run(s): {examples}"
        )

    provenance = _stage3_runtime_provenance()
    eval_fingerprint = provenance["pipeline_fingerprint"]
    tasks = [
        ScriptTask(
            Path(__file__).resolve(),
            (
                "--worker",
                f"--run-dir={run_dir}",
                f"--decoder-pairwise-block-size={args.decoder_pairwise_block_size}",
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
    incomplete = [path for path in run_dirs if not _is_complete(path, eval_fingerprint)]
    if return_code or incomplete:
        print(f"Evaluation incomplete: {len(incomplete)} run(s) lack valid artifacts.")
        return 1
    # A method subset controls new evaluation work, not the shared summary.
    # Rebuild it from every valid manifest evaluation so incremental additions
    # cannot erase methods that were completed by an earlier invocation.
    completed = _completed_run_dirs(sweep_dir, eval_fingerprint)
    _aggregate(sweep_dir, completed, completed)
    print(f"Evaluation summary: {sweep_dir / 'summary' / 'last'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
