"""Upload a bounded source snapshot and run/resume it on an existing Colab VM.

No automatic allocation or execution retry: transport failures leave job state
uncertain. Download state before deciding whether to resume on a replacement VM.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from colab_worker import plan_hash, validate_plan

ROOT = Path(__file__).resolve().parents[1]
CLI = ["bash", str(ROOT / "scripts/colab")]


def cli(*args, timeout=180):
    return subprocess.run(CLI + list(args), cwd=ROOT, check=True, timeout=timeout)


def bundle(destination):
    # Never package credentials, caches, outputs, notebooks or the whole repo.
    files = sorted((ROOT / "src").glob("*.py")) + [
        ROOT / "configs/base.yaml", ROOT / "scripts/colab_worker.py",
        ROOT / "scripts/colab_smoke.py",
    ]
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError(f"Refusing symlink: {path}")
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix())
            archive.writestr(info, path.read_bytes())
    content = data.getvalue()
    if destination.exists() and destination.read_bytes() != content:
        raise ValueError("Source changed since this run was prepared; choose a new run-id.")
    destination.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["bundle", "run", "status", "fetch"])
    p.add_argument("--session", default="vg-sae")
    p.add_argument("--run-id", default="colab-smoke-v1")
    p.add_argument("--plan", type=Path, default=ROOT / "configs/colab_smoke.json")
    p.add_argument("--timeout", type=int, default=21600)
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        p.error("run-id must be a safe name (letters, digits, underscore, hyphen)")
    local = ROOT / ".colab" / "runs" / args.run_id
    local.mkdir(parents=True, exist_ok=True)
    remote_state = f"/content/drive/MyDrive/vg-sae-colab/{args.run_id}"
    if args.action == "status":
        cli("download", "-s", args.session, remote_state + "/state.json",
            str(local / "state.json"))
        print((local / "state.json").read_text())
        return
    if args.action == "fetch":
        # File API requests do not queue work on the busy Python kernel.
        cli("download", "-s", args.session, remote_state + "/results.zip",
            str(local / "results.zip"))
        return
    snapshot = local / "source.zip"
    digest = bundle(snapshot)
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    saved_plan = local / "plan.json"
    if saved_plan.exists() and plan_hash(json.loads(saved_plan.read_text())) != plan_hash(plan):
        raise ValueError("Plan changed; choose a new run-id.")
    (local / "plan.json").write_text(json.dumps(plan, indent=2))
    (local / "source.sha256").write_text(digest + "\n")
    print(f"Source SHA256: {digest}", flush=True)
    if args.action == "bundle":
        return
    remote_zip = f"/content/vg-sae-{digest}.zip"
    remote_work = f"/content/vg-sae-{digest}"
    cli("upload", "-s", args.session, str(snapshot), remote_zip)
    # Keep kernel busy with actual work; no keep-alive or detached dummy loops.
    code = f'''
import importlib, json, os, pathlib, platform, shutil, subprocess, sys, zipfile
drive = pathlib.Path('/content/drive/MyDrive')
if not drive.is_dir() or not os.path.ismount('/content/drive'):
    raise RuntimeError('Mount Google Drive first with the project Colab CLI drivemount command.')
for module, package in [('torch', 'torch'), ('numpy', 'numpy'), ('yaml', 'pyyaml')]:
    try:
        importlib.import_module(module)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
work = pathlib.Path({remote_work!r})
work.mkdir(exist_ok=True)
with zipfile.ZipFile({remote_zip!r}) as archive:
    archive.extractall(work)
plan = work / 'colab-plan-{args.run_id}.json'
plan.write_text({json.dumps(plan)!r})
subprocess.run([sys.executable, str(work / 'scripts/colab_worker.py'),
    '--plan', str(plan), '--work-root', str(work),
    '--state-root', {remote_state!r}, '--source-sha', {digest!r}], check=True)
state_root = pathlib.Path({remote_state!r})
shutil.copyfile({remote_zip!r}, state_root / 'source.zip')
shutil.copyfile(plan, state_root / 'plan.json')
environment = {{'python': platform.python_version(), 'executable': sys.executable,
               'packages': {{m: str(getattr(importlib.import_module(m), '__version__', 'unknown'))
                            for m in ['torch', 'numpy', 'yaml']}}}}
(state_root / 'environment.json').write_text(json.dumps(environment, indent=2))
with zipfile.ZipFile(state_root / 'results.zip.tmp', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(state_root.rglob('*')):
        if path.is_file() and path.name not in ['results.zip', 'results.zip.tmp']:
            archive.write(path, path.relative_to(state_root))
os.replace(state_root / 'results.zip.tmp', state_root / 'results.zip')
'''
    launcher = local / "launch.py"
    launcher.write_text(code)
    cli("exec", "-s", args.session, "--timeout", str(args.timeout), "-f",
        str(launcher), timeout=args.timeout + 60)
    # CLI 0.7.4 can return exit 0 after a remote Python exception.
    cli("download", "-s", args.session, remote_state + "/state.json",
        str(local / "state.json"))
    state = json.loads((local / "state.json").read_text())
    if (state.get("source_sha256") != digest
            or state.get("plan_sha256") != plan_hash(plan)
            or state.get("status") != "completed"):
        raise SystemExit("Remote completion not confirmed. Inspect status/logs before retrying.")
    print("Completed; results are saved in Google Drive:", remote_state)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        raise SystemExit(f"Colab command did not complete: {exc}. Do not blindly rerun; check status first.")
