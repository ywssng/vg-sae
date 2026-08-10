"""File-layout and aggregation helpers for sweep launchers."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def runtime_provenance(project_root: Path, source_files: Iterable[str]) -> dict[str, Any]:
    """Fingerprint the code and package versions that define an artifact."""

    source_hash = hashlib.sha256()
    for relative_name in sorted(source_files):
        source_hash.update(relative_name.encode())
        source_hash.update((project_root / relative_name).read_bytes())
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    provenance = {
        "source_fingerprint": source_hash.hexdigest(),
        "git_revision": git_revision,
        "python": platform.python_version(),
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "scikit_learn": package_version("scikit-learn"),
        "pandas": package_version("pandas"),
        "sae_lens": package_version("sae-lens"),
    }
    provenance["pipeline_fingerprint"] = fingerprint(
        {key: value for key, value in provenance.items() if key != "git_revision"}
    )
    return provenance


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(materialized)
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_devices(requested: str) -> list[str]:
    if requested != "auto":
        devices = [item.strip() for item in requested.split(",") if item.strip()]
        if not devices:
            raise ValueError("--devices must name at least one device.")
        return devices

    import torch

    if torch.cuda.is_available():
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    return ["cpu"]


def manifest_run_dirs(sweep_dir: Path) -> list[Path]:
    manifest = read_json(sweep_dir / "manifest.json")
    return [sweep_dir / entry["relative_dir"] for entry in manifest["runs"]]


def aggregate_csv(run_dirs: Iterable[Path], relative_path: Path, destination: Path) -> None:
    rows = [row for run_dir in run_dirs for row in read_rows(run_dir / relative_path)]
    write_rows(destination, rows)
