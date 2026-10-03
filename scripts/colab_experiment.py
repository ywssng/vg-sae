"""Upload a bounded source snapshot and run/resume it on an existing Colab VM.

No automatic allocation or execution retry: transport failures leave job state
uncertain. Download state before deciding whether to resume on a replacement VM.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
import sys
import zipfile

from colab_worker import plan_hash, validate_plan

ROOT = Path(__file__).resolve().parents[1]
CLI = ["bash", str(ROOT / "scripts/colab")]

# This code runs in the remote launcher, before any planned task. Checking only
# tensor placement would miss a GPU allocation that still consumes quota.
CPU_ONLY_GUARD = '''
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    raise RuntimeError('Cannot verify CPU-only allocation with CUDA_VISIBLE_DEVICES set; inspect the runtime first.')
if os.environ.get('NVIDIA_VISIBLE_DEVICES', 'all').lower() in ('', 'none', 'void'):
    raise RuntimeError('Cannot verify CPU-only allocation with NVIDIA devices hidden.')
cuda = importlib.import_module('torch').cuda
if cuda.is_available() or cuda.device_count() != 0:
    raise RuntimeError('CPU-only plan refused: a GPU is available in this runtime.')
nvidia_smi = shutil.which('nvidia-smi')
if nvidia_smi:
    probe = subprocess.run([nvidia_smi, '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=15, check=False)
    diagnostic = (probe.stdout + probe.stderr).lower()
    if probe.returncode == 0 and probe.stdout.strip():
        raise RuntimeError('CPU-only plan refused: nvidia-smi reports an allocated GPU.')
    if probe.returncode != 0 and 'no devices were found' not in diagnostic:
        raise RuntimeError('CPU-only allocation could not be verified: nvidia-smi failed.')
os.environ['VG_SAE_COLAB_CPU_ONLY'] = '1'
'''

_EXCLUDED_COMPONENTS = frozenset({
    ".git", ".colab", ".cache", "__pycache__", ".venv", "venv", "node_modules",
    ".ssh", ".aws", ".azure", ".config", ".gnupg", "cache", "caches",
})
_CREDENTIAL_NAME = re.compile(
    r"(?:credentials?|secrets?|tokens?|oauth2?|auth|client[_-]secret|service[_-]account)"
    r"(?:[._-].*)?$|id_(?:rsa|ed25519)$",
    re.IGNORECASE,
)


def source_file(relative):
    """Validate an explicit repository file, including symlinked parent paths."""
    if not isinstance(relative, str) or not relative or "\\" in relative or "\0" in relative:
        raise ValueError("Source entries must be repository-relative POSIX file paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError(f"Source path must stay inside the repository: {relative}")
    for part in path.parts:
        if (part.lower() in _EXCLUDED_COMPONENTS or part.lower().startswith(".env")
                or _CREDENTIAL_NAME.fullmatch(part)
                or Path(part).suffix.lower() in {".pem", ".key", ".p12", ".pfx"}):
            raise ValueError(f"Credential or cache source path is excluded: {relative}")
    candidate = ROOT
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Refusing symlink: {relative}")
    if not candidate.resolve().is_relative_to(ROOT.resolve()) or not candidate.is_file():
        raise ValueError(f"Source must be an existing repository file: {relative}")
    return candidate


def manifest_sources(manifest):
    manifest = Path(manifest)
    if manifest.is_absolute():
        try:
            manifest = manifest.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError("Source manifest must be inside the repository") from exc
    manifest = source_file(manifest.as_posix())
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Source manifest must be a JSON list of repository-relative file paths")
    files = [source_file(entry) for entry in entries]
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def cli(*args, timeout=180):
    return subprocess.run(CLI + list(args), cwd=ROOT, check=True, timeout=timeout)


def bundle(destination, source_manifest=None):
    # Preserve the original default archive order/bytes. Additional inputs are
    # explicit files only; never recursively package credentials or caches.
    files = sorted((ROOT / "src").glob("*.py")) + [
        ROOT / "configs/base.yaml", ROOT / "scripts/colab_worker.py",
        ROOT / "scripts/colab_smoke.py",
    ]
    files = [source_file(path.relative_to(ROOT).as_posix()) for path in files]
    if source_manifest is not None:
        files += [path for path in manifest_sources(source_manifest) if path not in files]
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
    p.add_argument("--source-manifest", type=Path,
                   help="JSON list of additional repository-relative files to upload")
    p.add_argument("--cpu-only", action="store_true",
                   help="Bind the plan to CPU execution and reject a GPU runtime")
    p.add_argument("--storage", choices=("drive", "runtime"), default="drive",
                   help="State/result location; runtime storage requires an immediate fetch")
    p.add_argument("--timeout", type=int, default=21600)
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        p.error("run-id must be a safe name (letters, digits, underscore, hyphen)")
    local = ROOT / ".colab" / "runs" / args.run_id
    local.mkdir(parents=True, exist_ok=True)
    remote_base = "/content/drive/MyDrive/vg-sae-colab" if args.storage == "drive" else "/content/vg-sae-runs"
    remote_state = f"{remote_base}/{args.run_id}"
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
    digest = bundle(snapshot, args.source_manifest)
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.cpu_only:
        plan = {**plan, "colab_cpu_only": True}
    if args.storage == "runtime":
        plan = {**plan, "colab_storage": "runtime"}
    if plan.get("colab_storage", "drive") != args.storage:
        raise ValueError("Plan storage differs from --storage; select the matching storage explicitly")
    cpu_only = plan.get("colab_cpu_only", False)
    if not isinstance(cpu_only, bool):
        raise ValueError("colab_cpu_only must be a boolean")
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
if {args.storage == 'drive'!r}:
    drive = pathlib.Path('/content/drive/MyDrive')
    if not drive.is_dir() or not os.path.ismount('/content/drive'):
        raise RuntimeError('Mount Google Drive first with the project Colab CLI drivemount command.')
for module, package in [('torch', 'torch'), ('numpy', 'numpy'), ('yaml', 'pyyaml')]:
    try:
        importlib.import_module(module)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
{CPU_ONLY_GUARD if cpu_only else ''}
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
               'cpu_only': {cpu_only!r},
               'storage': {args.storage!r},
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
    print("Completed; results are saved in", args.storage, "storage:", remote_state)
    if args.storage == "runtime":
        print("Fetch the results now; runtime storage is lost when the Colab VM is removed.")


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        raise SystemExit(f"Colab command did not complete: {exc}. Do not blindly rerun; check status first.")
