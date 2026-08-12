"""Small subprocess scheduler shared by experiment launchers."""

from __future__ import annotations

import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def activate_worker_device(device: str) -> None:
    """Make an assigned CUDA device current before libraries initialize CUDA.

    Passing an indexed device to tensor constructors is not sufficient for all
    third-party loaders: some initialize the process's current CUDA device as
    well.  Selecting it at worker entry prevents nonzero-GPU workers from
    creating a stray primary context on GPU 0.
    """

    import torch

    normalized = torch.device(device)
    if normalized.type == "cuda":
        torch.cuda.set_device(normalized)


@dataclass(frozen=True)
class ScriptTask:
    script: Path
    args: tuple[str, ...]
    name: str

    def command(self, device: str) -> list[str]:
        return [sys.executable, str(self.script), *self.args, f"--device={device}"]


class ParallelExecutor:
    """Assign subprocesses to the least-loaded device as slots become free."""

    def __init__(
        self,
        tasks: Sequence[ScriptTask],
        devices: Sequence[str],
        max_per_device: int = 1,
    ) -> None:
        if not devices:
            raise ValueError("At least one device is required.")
        if max_per_device < 1:
            raise ValueError("max_per_device must be at least one.")
        self.tasks = list(tasks)
        self.devices = list(devices)
        self.max_per_device = max_per_device

    def run_all(self) -> int:
        pending = deque(self.tasks)
        running: dict[str, list[tuple[ScriptTask, subprocess.Popen[bytes]]]] = {
            device: [] for device in self.devices
        }
        failed: list[str] = []
        completed = 0

        print(
            f"Scheduling {len(pending)} task(s) on {len(self.devices)} device(s), "
            f"{self.max_per_device} per device."
        )
        try:
            while pending or any(running.values()):
                for device, jobs in running.items():
                    active = []
                    for task, process in jobs:
                        return_code = process.poll()
                        if return_code is None:
                            active.append((task, process))
                            continue
                        completed += 1
                        print(
                            f"Finished {task.name} on {device}: {return_code} "
                            f"[{completed}/{len(self.tasks)}]"
                        )
                        if return_code:
                            failed.append(task.name)
                    running[device] = active

                while pending:
                    available = [
                        device
                        for device in self.devices
                        if len(running[device]) < self.max_per_device
                    ]
                    if not available:
                        break
                    device = min(available, key=lambda item: len(running[item]))
                    task = pending.popleft()
                    command = task.command(device)
                    print(f"Launching {task.name} on {device}")
                    running[device].append((task, subprocess.Popen(command)))

                if pending or any(running.values()):
                    time.sleep(0.25)
        except KeyboardInterrupt:
            processes = [process for jobs in running.values() for _, process in jobs]
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

        if failed:
            print(f"{len(failed)} task(s) failed: {', '.join(failed)}")
            return 1
        return 0
