"""Execute bounded task plans on Colab, persisting task completion to Drive.

Resume is at task boundaries: interrupted tasks run again from the beginning.
Commands should put durable results in the COLAB_TASK_OUTPUT directory. The
local lock prevents duplicate workers in one runtime; it is not a distributed
lock between different Colab runtimes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid


def validate_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise ValueError("Plan must have version 1")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Plan must contain a nonempty tasks list")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Each task must be an object")
        task_id = task.get("id")
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", task_id):
            raise ValueError("Task IDs must be safe names of 1 to 100 characters")
        if task_id in seen:
            raise ValueError(f"Duplicate task ID: {task_id}")
        seen.add(task_id)
        argv = task.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and "\0" not in arg for arg in argv) or not argv[0]:
            raise ValueError(f"Task {task_id} needs nonempty argv with string arguments")


def plan_hash(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def runtime_lock(work_root: Path, state_root: Path):
    """Lock locally, so an old Drive marker never blocks a replacement runtime."""
    lock_dir = work_root / ".colab-locks"
    lock_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(str(state_root.resolve()).encode()).hexdigest()
    with (lock_dir / f"{key}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another worker is already using this run in this runtime") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_plan(plan: dict, work_root: Path, state_root: Path, source_sha: str) -> int:
    validate_plan(plan)
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("source-sha must be a lowercase SHA-256 digest")
    work_root, state_root = Path(work_root).resolve(), Path(state_root).resolve()
    if not work_root.is_dir():
        raise ValueError("work-root must exist")
    with runtime_lock(work_root, state_root):
        state_root.mkdir(parents=True, exist_ok=True)
        state_path = state_root / "state.json"
        identity = {"version": 1, "plan_sha256": plan_hash(plan), "source_sha256": source_sha}
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or any(state.get(k) != v for k, v in identity.items()):
                raise ValueError("Existing run has a different plan or source; choose a new state-root")
            if not isinstance(state.get("tasks"), dict):
                raise ValueError("Invalid saved tasks state")
        else:
            state = {**identity, "created_at": timestamp(), "tasks": {}}
        state.update(status="running", updated_at=timestamp())
        atomic_json(state_path, state)
        for task in plan["tasks"]:
            task_id = task["id"]
            previous = state["tasks"].get(task_id, {})
            if previous.get("status") == "completed":
                print(f"Skipping completed task: {task_id}", flush=True)
                continue
            task_root = state_root / "tasks" / task_id
            output = task_root / "output"
            output.mkdir(parents=True, exist_ok=True)
            record = {"status": "running", "attempt": previous.get("attempt", 0) + 1, "started_at": timestamp()}
            state["tasks"][task_id] = record
            state["updated_at"] = timestamp()
            atomic_json(state_path, state)
            env = dict(os.environ, COLAB_TASK_OUTPUT=str(output), PYTHONUNBUFFERED="1")
            print(f"Running task: {task_id} (attempt {record['attempt']})", flush=True)
            try:
                with (task_root / "stdout.log").open("a", encoding="utf-8") as log:
                    log.write(f"\n--- Attempt {record['attempt']} at {record['started_at']} ---\n")
                    log.flush()
                    result = subprocess.run(task["argv"], cwd=work_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
                returncode = result.returncode
            except (OSError, KeyboardInterrupt) as exc:
                returncode = 130 if isinstance(exc, KeyboardInterrupt) else 127
                record["error"] = str(exc)
            record.update(status="completed" if returncode == 0 else "failed", returncode=returncode, finished_at=timestamp())
            state.update(status="running" if returncode == 0 else "failed", updated_at=timestamp())
            atomic_json(state_path, state)
            if returncode:
                print(f"Task failed: {task_id}; see {task_root / 'stdout.log'}", flush=True)
                return returncode if returncode > 0 else 128 - returncode
        state.update(status="completed", updated_at=timestamp())
        atomic_json(state_path, state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    try:
        return run_plan(json.loads(args.plan.read_text(encoding="utf-8")), args.work_root, args.state_root, args.source_sha)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"colab_worker: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
