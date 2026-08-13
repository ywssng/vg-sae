"""Run the five Stage-1 amplitude/frequency ablations end to end."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs._sweep_io import read_json  # noqa: E402
from src.sae_sweep_plot import plot_all  # noqa: E402


@dataclass(frozen=True)
class AblationCondition:
    name: str
    amplitude_mode: str
    frequency_skew: float


CONDITIONS = (
    AblationCondition("ablation2_constant", "constant", 0.5),
    AblationCondition("ablation2_uniform", "uniform", 0.5),
    AblationCondition("ablation3_uniformfreq", "exponential", 0.0),
    AblationCondition("ablation23_constant_uniformfreq", "constant", 0.0),
    AblationCondition("ablation23_uniform_uniformfreq", "uniform", 0.0),
)
BETA_MODES = ("profiled", "learned")
EXPECTED_RUNS = 273


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated condition names, or all.",
    )
    parser.add_argument(
        "--beta-modes",
        default="profiled,learned",
        help="Comma-separated profiled/learned modes.",
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--max-per-device", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def selected_conditions(requested: str) -> list[AblationCondition]:
    if requested == "all":
        return list(CONDITIONS)
    names = {name.strip() for name in requested.split(",") if name.strip()}
    known = {condition.name for condition in CONDITIONS}
    if unknown := names - known:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    return [condition for condition in CONDITIONS if condition.name in names]


def selected_beta_modes(requested: str) -> list[str]:
    modes = [mode.strip() for mode in requested.split(",") if mode.strip()]
    if not modes or set(modes) - set(BETA_MODES):
        raise ValueError("--beta-modes must contain profiled and/or learned.")
    if len(set(modes)) != len(modes):
        raise ValueError("--beta-modes contains a duplicate.")
    return modes


def sweep_root(condition: AblationCondition, beta_mode: str) -> Path:
    return PROJECT_ROOT / "outputs" / "runs" / (
        f"stage1_{condition.name}_beta_{beta_mode}"
        "_din128_gt1024_sae1024_sd001_seed0"
    )


def run_checked(command: list[str], *, environment: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def validate_and_plot(
    root: Path,
    condition: AblationCondition,
    beta_mode: str,
    *,
    make_plots: bool,
) -> None:
    manifest = read_json(root / "manifest.json")
    if len(manifest["runs"]) != EXPECTED_RUNS:
        raise RuntimeError(f"{root.name}: expected {EXPECTED_RUNS} manifest runs.")

    for checkpoint_kind in ("last", "best"):
        metrics_path = root / "summary" / checkpoint_kind / "final_metrics.csv"
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected_axes = {
            "beta_mode": beta_mode,
            "amplitude_mode": condition.amplitude_mode,
            "amplitude_scale": "1.0",
            "frequency_skew": str(condition.frequency_skew),
            "hard_metric_schema_version": "1",
        }
        if len(rows) != EXPECTED_RUNS:
            raise RuntimeError(
                f"{root.name}[{checkpoint_kind}]: expected {EXPECTED_RUNS} rows."
            )
        for key, value in expected_axes.items():
            observed = {row[key] for row in rows}
            if observed != {value}:
                raise RuntimeError(
                    f"{root.name}[{checkpoint_kind}]: {key}={observed}, "
                    f"expected {value!r}."
                )
        if make_plots:
            destination = root / "figures" / "hard_density" / checkpoint_kind
            figures, _ = plot_all(
                root,
                checkpoint_kind=checkpoint_kind,
                output_dir=destination,
                density_mode="hard",
            )
            for figure in figures.values():
                plt.close(figure)


def main(args: argparse.Namespace) -> int:
    conditions = selected_conditions(args.conditions)
    beta_modes = selected_beta_modes(args.beta_modes)
    environment = dict(os.environ)
    environment["WANDB_SILENT"] = "true"
    common = [
        "--methods",
        "all",
        "--devices",
        args.devices,
        "--max-per-device",
        str(args.max_per_device),
    ]
    if args.force:
        common.append("--force")

    for condition in conditions:
        for beta_mode in beta_modes:
            root = sweep_root(condition, beta_mode)
            print(f"\n=== {root.name}: train ===", flush=True)
            run_checked(
                [
                    sys.executable,
                    "-B",
                    str(PROJECT_ROOT / "runs" / "run_CustomData_sweep.py"),
                    "--output-dir",
                    str(root),
                    "--amplitude-mode",
                    condition.amplitude_mode,
                    "--frequency-skew",
                    str(condition.frequency_skew),
                    "--beta-mode",
                    beta_mode,
                    *common,
                ],
                environment=environment,
            )
            if args.skip_eval:
                continue

            print(f"\n=== {root.name}: last+best eval ===", flush=True)
            run_checked(
                [
                    sys.executable,
                    "-B",
                    str(PROJECT_ROOT / "runs" / "run_CustomData_sweep_eval.py"),
                    "--sweep-dir",
                    str(root),
                    *common,
                ],
                environment=environment,
            )
            validate_and_plot(
                root,
                condition,
                beta_mode,
                make_plots=not args.skip_plots,
            )
            print(f"Validated: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
