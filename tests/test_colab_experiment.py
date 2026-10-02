"""Transport and packaging checks without allocating Colab or training locally."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("colab_experiment", SCRIPTS / "colab_experiment.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture
def root(tmp_path, monkeypatch):
    for folder in ["src", "scripts", "configs", ".colab"]:
        (tmp_path / folder).mkdir()
    for name in ["src/model.py", "scripts/colab_worker.py", "scripts/colab_smoke.py", "configs/base.yaml"]:
        (tmp_path / name).write_text("# fixture\n")
    (tmp_path / ".colab/token.json").write_text("DO NOT UPLOAD")
    (tmp_path / "configs/colab_smoke.json").write_text(json.dumps({
        "version": 1, "tasks": [{"id": "one", "argv": ["python", "-c", "pass"]}]
    }))
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    return tmp_path


def test_bundle_excludes_credentials_and_preserves_original(root):
    target = root / ".colab/source.zip"
    digest = driver.bundle(target)
    assert driver.bundle(target) == digest
    before = target.read_bytes()
    with zipfile.ZipFile(target) as archive:
        assert ".colab/token.json" not in archive.namelist()
        assert "src/model.py" in archive.namelist()
    (root / "src/model.py").write_text("# changed")
    with pytest.raises(ValueError, match="Source changed"):
        driver.bundle(target)
    assert target.read_bytes() == before


def test_run_checks_remote_plan_even_when_cli_exits_zero(root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["driver", "run", "--session", "apostrophe'allowed"])
    calls = []

    def fake_cli(*args, **kwargs):
        calls.append(args)
        if args[0] == "exec":
            launch = Path(args[-1]).read_text()
            compile(launch, "remote_launcher", "exec")
            assert "--source-sha" in launch
        elif args[0] == "download":
            digest = (root / ".colab/runs/colab-smoke-v1/source.sha256").read_text().strip()
            Path(args[-1]).write_text(json.dumps({
                "status": "completed", "source_sha256": digest,
                "plan_sha256": "a-different-plan",
            }))

    monkeypatch.setattr(driver, "cli", fake_cli)
    with pytest.raises(SystemExit, match="completion not confirmed"):
        driver.main()
    assert [args[0] for args in calls] == ["upload", "exec", "download"]


def test_fetch_downloads_archive_without_kernel_execution(root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["driver", "fetch"])
    calls = []
    monkeypatch.setattr(driver, "cli", lambda *args, **kwargs: calls.append(args))
    driver.main()
    assert len(calls) == 1
    assert calls[0][0] == "download"
    assert calls[0][-2].endswith("/results.zip")


def test_unsafe_run_id_rejected_before_any_transport(root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["driver", "run", "--run-id", "../other"])
    monkeypatch.setattr(driver, "cli", lambda *a, **kw: pytest.fail("transport called"))
    with pytest.raises(SystemExit):
        driver.main()
