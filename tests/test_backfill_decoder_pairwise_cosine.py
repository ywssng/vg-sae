from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import scripts.backfill_decoder_pairwise_cosine as backfill
from runs._sweep_io import write_json, write_rows


def test_checkpoint_only_backfill_writes_and_reuses_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "runs" / "topk" / "topk-1"
    checkpoint = run_dir / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    write_json(
        tmp_path / "sweep_config.json",
        {"data": {"kind": "stage1_custom_baseline"}},
    )
    skipped_run_dir = tmp_path / "runs" / "topk" / "topk-2"
    skipped_run_dir.mkdir(parents=True)
    write_json(
        tmp_path / "manifest.json",
        {
            "runs": [
                {"relative_dir": "runs/topk/topk-1"},
                {"relative_dir": "runs/topk/topk-2"},
            ]
        },
    )
    write_json(
        run_dir / "config.json",
        {
            "spec": {
                "method": "topk",
                "seed": 0,
                "control_name": "k",
                "control_value": 1,
            }
        },
    )
    write_json(
        skipped_run_dir / "config.json",
        {
            "spec": {
                "method": "topk",
                "seed": 0,
                "control_name": "k",
                "control_value": 2,
            }
        },
    )
    write_rows(
        tmp_path / "summary" / "last" / "final_metrics.csv",
        [{"method": "topk", "run_id": "topk-1"}],
    )
    calls = []

    def fake_loader(path: Path, device: str):
        calls.append((path, device))
        return SimpleNamespace(W_dec=torch.eye(2)), {}

    monkeypatch.setattr(backfill, "load_stage1_checkpoint", fake_loader)

    destination = backfill.backfill_sweep(
        tmp_path,
        checkpoint_kind="last",
        device="cpu",
        block_size=1,
        force=False,
    )
    backfill.backfill_sweep(
        tmp_path,
        checkpoint_kind="last",
        device="cpu",
        block_size=1,
        force=False,
    )

    row = pd.read_csv(destination).iloc[0]
    assert row[backfill.METRIC_NAME] == pytest.approx(0.0)
    assert row["metric_definition"] == "Sparse but Wrong Eq. (4)"
    assert row["metric_schema_version"] == backfill.METRIC_SCHEMA_VERSION
    assert len(pd.read_csv(destination)) == 1
    assert calls == [(checkpoint, "cpu")]
