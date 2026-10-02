from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts.colab_worker import run_plan, runtime_lock, validate_plan


SHA = "a" * 64


def command(task_id: str, code: str) -> dict:
    return {"id": task_id, "argv": [sys.executable, "-c", code]}


def test_resume_preserves_completed_task_and_reruns_failed_task(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    state_root = tmp_path / "drive"
    plan = {"version": 1, "tasks": [
        command("first", "import os,pathlib; p=pathlib.Path(os.environ['COLAB_TASK_OUTPUT'])/'count'; p.write_text(str(int(p.read_text())+1) if p.exists() else '1'); print('first log')"),
        command("second", "import os,pathlib,sys; p=pathlib.Path(os.environ['COLAB_TASK_OUTPUT'])/'attempted'; exists=p.exists(); p.write_text('yes'); sys.exit(0 if exists else 3)"),
    ]}
    assert run_plan(plan, work, state_root, SHA) == 3
    state = json.loads((state_root / "state.json").read_text())
    assert state["status"] == "failed"
    assert state["tasks"]["first"]["status"] == "completed"
    assert run_plan(plan, work, state_root, SHA) == 0
    assert run_plan(plan, work, state_root, SHA) == 0
    state = json.loads((state_root / "state.json").read_text())
    assert state["status"] == "completed"
    assert state["tasks"]["second"]["attempt"] == 2
    assert (state_root / "tasks/first/output/count").read_text() == "1"
    assert "first log" in (state_root / "tasks/first/stdout.log").read_text()


@pytest.mark.parametrize("change_source", [False, True])
def test_rejects_changed_plan_or_source_before_execution(tmp_path: Path, change_source: bool):
    state_root = tmp_path / "state"
    plan = {"version": 1, "tasks": [command("one", "pass")]}
    assert run_plan(plan, tmp_path, state_root, SHA) == 0
    before = (state_root / "state.json").read_bytes()
    if not change_source:
        plan["tasks"][0]["argv"][-1] = "raise Exception('must not execute')"
    with pytest.raises(ValueError, match="different plan or source"):
        run_plan(plan, tmp_path, state_root, "b" * 64 if change_source else SHA)
    assert (state_root / "state.json").read_bytes() == before


def test_local_lock_rejects_second_worker_and_releases(tmp_path: Path):
    state_root = tmp_path / "state"
    plan = {"version": 1, "tasks": [command("one", "pass")]}
    with runtime_lock(tmp_path, state_root):
        with pytest.raises(RuntimeError, match="Another worker"):
            run_plan(plan, tmp_path, state_root, SHA)
    assert not state_root.exists()
    assert run_plan(plan, tmp_path, state_root, SHA) == 0


def test_interrupted_running_record_is_retried(tmp_path: Path):
    state_root = tmp_path / "state"
    plan = {"version": 1, "tasks": [command("one", "pass")]}
    assert run_plan(plan, tmp_path, state_root, SHA) == 0
    path = state_root / "state.json"
    state = json.loads(path.read_text())
    state["status"] = state["tasks"]["one"]["status"] = "running"
    path.write_text(json.dumps(state))
    assert run_plan(plan, tmp_path, state_root, SHA) == 0
    assert json.loads(path.read_text())["tasks"]["one"]["attempt"] == 2


@pytest.mark.parametrize("tasks", [[], [{"id": "../escape", "argv": ["python"]}], [{"id": "ok", "argv": []}], [{"id": "ok", "argv": [1]}], [command("same", "pass"), command("same", "pass")]])
def test_invalid_task_plans_are_rejected(tasks):
    with pytest.raises(ValueError):
        validate_plan({"version": 1, "tasks": tasks})


def test_missing_executable_records_failure(tmp_path: Path):
    plan = {"version": 1, "tasks": [{"id": "missing", "argv": [str(tmp_path / "absent-executable")]}]}
    assert run_plan(plan, tmp_path, tmp_path / "state", SHA) == 127
    state = json.loads((tmp_path / "state/state.json").read_text())
    assert state["tasks"]["missing"]["status"] == "failed"
    assert "error" in state["tasks"]["missing"]
