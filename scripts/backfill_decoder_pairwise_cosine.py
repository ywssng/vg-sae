"""Compute Sparse but Wrong Eq. (4) from saved SAE checkpoints only."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs._sweep_io import manifest_run_dirs, read_json, read_rows, write_rows  # noqa: E402
from src.sae_evaluate import (  # noqa: E402
    decoder_atoms_from_model,
    decoder_pairwise_cosine_similarity,
)
from src.sae_sweep import load_checkpoint as load_stage1_checkpoint  # noqa: E402
from src.synthsaebench_sweep import (  # noqa: E402
    SYNTHSAEBENCH_DATA_KIND,
    load_checkpoint as load_stage2_checkpoint,
)


METRIC_NAME = "decoder_pairwise_cosine_similarity"
SIDECAR_NAME = "decoder_pairwise_cosine_metrics.csv"
METRIC_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        action="append",
        required=True,
        type=Path,
        help="Sweep root; repeat to backfill more than one root.",
    )
    parser.add_argument("--checkpoint", choices=("last", "best"), default="last")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _checkpoint_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _existing_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (str(row["method"]), str(row["run_id"])): row
        for row in read_rows(path)
    }


def _can_reuse(row: dict[str, str], checkpoint: Path, checkpoint_kind: str) -> bool:
    size, mtime_ns = _checkpoint_identity(checkpoint)
    try:
        value = float(row[METRIC_NAME])
        return (
            row.get("checkpoint_kind") == checkpoint_kind
            and int(row["metric_schema_version"]) == METRIC_SCHEMA_VERSION
            and int(row["checkpoint_size"]) == size
            and int(row["checkpoint_mtime_ns"]) == mtime_ns
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
        )
    except (KeyError, TypeError, ValueError):
        return False


def backfill_sweep(
    sweep_dir: Path,
    *,
    checkpoint_kind: str,
    device: str,
    block_size: int,
    force: bool,
) -> Path:
    root = sweep_dir.resolve()
    data_kind = str(read_json(root / "sweep_config.json")["data"]["kind"])
    is_stage2 = data_kind == SYNTHSAEBENCH_DATA_KIND
    if is_stage2 and checkpoint_kind != "last":
        raise ValueError("Stage-2 checkpoints only support checkpoint_kind='last'.")
    loader = load_stage2_checkpoint if is_stage2 else load_stage1_checkpoint
    destination = root / "summary" / checkpoint_kind / SIDECAR_NAME
    existing = {} if force else _existing_rows(destination)
    rows: list[dict[str, Any]] = []
    run_dirs = manifest_run_dirs(root)
    evaluated_rows = read_rows(destination.parent / "final_metrics.csv")
    evaluated_keys = [
        (str(row["method"]), str(row["run_id"])) for row in evaluated_rows
    ]
    target_keys = set(evaluated_keys)
    if len(target_keys) != len(evaluated_keys):
        raise ValueError("final_metrics.csv contains duplicate evaluated runs.")
    selected_runs: list[tuple[Path, dict[str, Any], tuple[str, str]]] = []
    for run_dir in run_dirs:
        bundle = read_json(run_dir / "config.json")
        spec = bundle["spec"]
        key = (str(spec["method"]), run_dir.name)
        if key in target_keys:
            selected_runs.append((run_dir, spec, key))
    selected_keys = {key for _, _, key in selected_runs}
    if selected_keys != target_keys:
        missing = sorted(target_keys - selected_keys)
        raise ValueError(
            "Evaluated runs are absent from the sweep manifest: "
            + ", ".join(f"{method}/{run_id}" for method, run_id in missing)
        )

    for index, (run_dir, spec, key) in enumerate(selected_runs, start=1):
        checkpoint = run_dir / "checkpoints" / f"{checkpoint_kind}.pt"
        run_id = key[1]
        previous = existing.get(key)
        if previous is not None and _can_reuse(
            previous, checkpoint, checkpoint_kind
        ):
            row: dict[str, Any] = dict(previous)
        else:
            model, _ = loader(checkpoint, device)
            value = decoder_pairwise_cosine_similarity(
                decoder_atoms_from_model(model), block_size=block_size
            )
            size, mtime_ns = _checkpoint_identity(checkpoint)
            row = {
                "method": spec["method"],
                "run_id": run_id,
                "seed": spec["seed"],
                "control_name": spec["control_name"],
                "control_value": spec["control_value"],
                "checkpoint_kind": checkpoint_kind,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "checkpoint_size": size,
                "checkpoint_mtime_ns": mtime_ns,
                "metric_definition": "Sparse but Wrong Eq. (4)",
                METRIC_NAME: value,
            }
            del model
        rows.append(row)
        print(f"[{index}/{len(selected_runs)}] {key[1]}")

    write_rows(destination, rows)
    print(f"Wrote {len(rows)} rows: {destination}")
    return destination


def main(args: argparse.Namespace) -> int:
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")
    for sweep_dir in args.sweep_dir:
        backfill_sweep(
            sweep_dir,
            checkpoint_kind=args.checkpoint,
            device=args.device,
            block_size=args.block_size,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
