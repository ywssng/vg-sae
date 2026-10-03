"""Transport and packaging checks without allocating Colab or training locally."""
import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
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


def test_default_bundle_retains_legacy_bytes(root):
    expected = io.BytesIO()
    names = ["src/model.py", "configs/base.yaml", "scripts/colab_worker.py", "scripts/colab_smoke.py"]
    with zipfile.ZipFile(expected, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(zipfile.ZipInfo(name), (root / name).read_bytes())
    target = root / ".colab/default.zip"
    driver.bundle(target)
    assert target.read_bytes() == expected.getvalue()


def test_manifest_is_explicit_deterministic_and_resume_bound(root):
    (root / "scripts/research.py").write_text("# research task\n")
    (root / "configs/research.json").write_text('{"seed": 4}\n')
    (root / "scripts/unlisted.py").write_text("# not selected\n")
    manifest = root / "configs/source_manifest.json"
    manifest.write_text(json.dumps(["scripts/research.py", "configs/research.json", "src/model.py"]))
    target = root / ".colab/research.zip"
    digest = driver.bundle(target, manifest)
    original = target.read_bytes()
    with zipfile.ZipFile(target) as archive:
        assert "scripts/research.py" in archive.namelist()
        assert "configs/research.json" in archive.namelist()
        assert "scripts/unlisted.py" not in archive.namelist()
        assert archive.namelist().count("src/model.py") == 1
    manifest.write_text(json.dumps(["src/model.py", "configs/research.json", "scripts/research.py",
                                    "scripts/research.py"]))
    assert driver.bundle(target, manifest) == digest
    (root / "scripts/research.py").write_text("# changed scientific task\n")
    with pytest.raises(ValueError, match="Source changed"):
        driver.bundle(target, manifest)
    assert target.read_bytes() == original


@pytest.mark.parametrize("entry", [
    "../outside.py", "/tmp/outside.py", "src/../scripts/research.py", "src//model.py",
    "src\\model.py", ".colab/token.json", ".cache/data.json", "outputs/cache/data.npz",
    "configs/.env.production", "configs/service_account.json", "configs/credentials.json",
    "configs/client_secret.json", "configs/key.pem", 42,
])
def test_manifest_rejects_escape_credentials_and_cache_paths(root, entry):
    manifest = root / "configs/source_manifest.json"
    manifest.write_text(json.dumps([entry]))
    target = root / ".colab/rejected.zip"
    with pytest.raises(ValueError):
        driver.bundle(target, manifest)
    assert not target.exists()


@pytest.mark.parametrize("directory_link", [False, True])
def test_manifest_rejects_symlink_files_and_parent_directories(root, directory_link):
    if directory_link:
        (root / "alias").symlink_to(root / "src", target_is_directory=True)
        entry = "alias/model.py"
    else:
        (root / "scripts/alias.py").symlink_to(root / "src/model.py")
        entry = "scripts/alias.py"
    manifest = root / "configs/source_manifest.json"
    manifest.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="symlink"):
        driver.bundle(root / ".colab/rejected.zip", manifest)


def test_manifest_schema_and_location_are_validated(root):
    manifest = root / "configs/source_manifest.json"
    manifest.write_text('{"files": ["src/model.py"]}')
    with pytest.raises(ValueError, match="JSON list"):
        driver.bundle(root / ".colab/rejected.zip", manifest)
    with pytest.raises(ValueError, match="inside the repository"):
        driver.bundle(root / ".colab/rejected.zip", root.parent / "outside.json")


def test_cpu_only_changes_plan_identity_and_cannot_resume_default_run(root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["driver", "bundle"])
    driver.main()
    saved = root / ".colab/runs/colab-smoke-v1/plan.json"
    original = saved.read_bytes()
    monkeypatch.setattr(sys, "argv", ["driver", "bundle", "--cpu-only"])
    with pytest.raises(ValueError, match="Plan changed"):
        driver.main()
    assert saved.read_bytes() == original


def test_runtime_storage_cannot_resume_drive_plan_under_same_id(root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["driver", "bundle"])
    driver.main()
    saved = root / ".colab/runs/colab-smoke-v1/plan.json"
    original = saved.read_bytes()
    monkeypatch.setattr(sys, "argv", ["driver", "bundle", "--storage", "runtime"])
    with pytest.raises(ValueError, match="Plan changed"):
        driver.main()
    assert saved.read_bytes() == original


def cpu_guard_namespace(*, torch_gpu=False, command=None, returncode=0, stdout="", stderr="", env=None):
    cuda = SimpleNamespace(is_available=lambda: torch_gpu, device_count=lambda: int(torch_gpu))
    return {
        "os": SimpleNamespace(environ=dict(env or {})),
        "importlib": SimpleNamespace(import_module=lambda name: SimpleNamespace(cuda=cuda)),
        "shutil": SimpleNamespace(which=lambda name: command),
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr)),
    }


@pytest.mark.parametrize("kwargs", [
    {"torch_gpu": True},
    {"command": "/usr/bin/nvidia-smi", "stdout": "Tesla T4\n"},
    {"command": "/usr/bin/nvidia-smi", "returncode": 1, "stderr": "driver failure"},
    {"env": {"CUDA_VISIBLE_DEVICES": ""}},
    {"env": {"NVIDIA_VISIBLE_DEVICES": "none"}},
])
def test_cpu_guard_rejects_gpu_or_unverifiable_allocation(kwargs):
    namespace = cpu_guard_namespace(**kwargs)
    with pytest.raises(RuntimeError):
        exec(driver.CPU_ONLY_GUARD, namespace)
    assert "VG_SAE_COLAB_CPU_ONLY" not in namespace["os"].environ


@pytest.mark.parametrize("kwargs", [{}, {
    "command": "/usr/bin/nvidia-smi", "returncode": 6, "stdout": "No devices were found\n",
}])
def test_cpu_guard_accepts_cpu_and_can_run_twice_in_same_kernel(kwargs):
    namespace = cpu_guard_namespace(**kwargs)
    exec(driver.CPU_ONLY_GUARD, namespace)
    exec(driver.CPU_ONLY_GUARD, namespace)
    assert namespace["os"].environ["VG_SAE_COLAB_CPU_ONLY"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in namespace["os"].environ


def test_cpu_only_launcher_binds_guard_and_manifest_before_worker(root, monkeypatch):
    (root / "scripts/research.py").write_text("# additional source\n")
    manifest = root / "configs/source_manifest.json"
    manifest.write_text(json.dumps(["scripts/research.py"]))
    monkeypatch.setattr(sys, "argv", ["driver", "run", "--cpu-only", "--source-manifest", str(manifest)])

    def fake_cli(*args, **kwargs):
        if args[0] == "exec":
            launch = Path(args[-1]).read_text()
            compile(launch, "remote_launcher", "exec")
            assert launch.index(driver.CPU_ONLY_GUARD) < launch.index("archive.extractall(work)")
            assert launch.index(driver.CPU_ONLY_GUARD) < launch.index("'scripts/colab_worker.py'")
            assert "'cpu_only': True" in launch
        elif args[0] == "download":
            local = root / ".colab/runs/colab-smoke-v1"
            plan = json.loads((local / "plan.json").read_text())
            assert plan["colab_cpu_only"] is True
            with zipfile.ZipFile(local / "source.zip") as archive:
                assert "scripts/research.py" in archive.namelist()
            Path(args[-1]).write_text(json.dumps({
                "status": "completed", "source_sha256": (local / "source.sha256").read_text().strip(),
                "plan_sha256": driver.plan_hash(plan),
            }))

    monkeypatch.setattr(driver, "cli", fake_cli)
    driver.main()


def test_runtime_launch_skips_drive_mount_and_checks_completion(root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["driver", "run", "--cpu-only", "--storage", "runtime"])

    def fake_cli(*args, **kwargs):
        if args[0] == "exec":
            launch = Path(args[-1]).read_text()
            tree = ast.parse(launch)
            # Exercise the actual generated mount block with filesystem access
            # unavailable: runtime mode must not enter that branch.
            mount_guard = tree.body[1]
            assert isinstance(mount_guard, ast.If)
            exec(compile(ast.Module(body=[mount_guard], type_ignores=[]), "mount_guard", "exec"), {})
            assert "/content/vg-sae-runs/colab-smoke-v1" in launch
            assert "'storage': 'runtime'" in launch
        elif args[0] == "download":
            assert args[-2] == "/content/vg-sae-runs/colab-smoke-v1/state.json"
            local = root / ".colab/runs/colab-smoke-v1"
            plan = json.loads((local / "plan.json").read_text())
            assert plan["colab_storage"] == "runtime"
            Path(args[-1]).write_text(json.dumps({
                "status": "completed", "source_sha256": (local / "source.sha256").read_text().strip(),
                "plan_sha256": driver.plan_hash(plan),
            }))

    monkeypatch.setattr(driver, "cli", fake_cli)
    driver.main()


@pytest.mark.parametrize("action,remote_name", [("status", "state.json"), ("fetch", "results.zip")])
def test_runtime_status_and_fetch_use_file_api(root, monkeypatch, action, remote_name):
    monkeypatch.setattr(sys, "argv", ["driver", action, "--storage", "runtime"])
    calls = []

    def fake_cli(*args, **kwargs):
        calls.append(args)
        assert args[0] == "download"
        assert args[-2] == f"/content/vg-sae-runs/colab-smoke-v1/{remote_name}"
        if action == "status":
            Path(args[-1]).write_text('{"status": "running"}')

    monkeypatch.setattr(driver, "cli", fake_cli)
    driver.main()
    assert len(calls) == 1


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
