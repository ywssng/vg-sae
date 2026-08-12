from __future__ import annotations

import torch

from runs.gpu_scheduler import activate_worker_device


def test_activate_worker_device_selects_only_explicit_cuda_device(monkeypatch) -> None:
    calls: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)

    activate_worker_device("cuda:2")
    activate_worker_device("cpu")

    assert calls == [torch.device("cuda:2")]
