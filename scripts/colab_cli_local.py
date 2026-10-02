"""Isolate the pinned official CLI's hardcoded home paths in this checkout."""
import importlib.metadata
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    if importlib.metadata.version("google-colab-cli") != "0.7.4":
        raise SystemExit("Run bash scripts/colab setup; this adapter requires CLI 0.7.4")
    os.umask(0o077)
    config = ROOT / ".colab" / "config"
    config.mkdir(parents=True, exist_ok=True)
    original = os.path.expanduser

    def project_path(path):
        # Upstream ignores XDG for tokens, logs, history and settings.
        if isinstance(path, str):
            prefix = "~/.config/colab-cli"
            if path == prefix or path.startswith(prefix + "/"):
                return str(config) + path[len(prefix):]
            if path == "~/.colab-cli-oauth-config.json":
                return str(config / "oauth-client.json")
        return original(path)

    os.path.expanduser = project_path
    from colab_cli.cli import main as official_main

    args = sys.argv[1:]
    if any(a == "--config" or a.startswith("--config=") for a in args):
        raise SystemExit("Session config is managed by this project-local adapter.")
    if args == ["login"]:
        args = ["usage"]
    sys.argv = [sys.argv[0], "--auth", "oauth2", "--config",
                str(config / "sessions.json"), *args]
    official_main()


if __name__ == "__main__":
    main()
