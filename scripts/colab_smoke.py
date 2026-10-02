"""Small Colab-only VG regression verification, not a research benchmark."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import yaml
    from src.train import run_from_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not Path("/content").is_dir() or not os.environ.get("COLAB_TASK_OUTPUT"):
        raise SystemExit("Run through the Colab worker on a Colab runtime.")
    out = Path(os.environ["COLAB_TASK_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path("configs/base.yaml").read_text())
    config["model"]["n_features"] = 16
    config["data"].update(n_features=16, n_train=64, n_test=64, rho_data=0.125)
    config["training"].update(
        seed=args.seed, max_steps=100, history_every=25,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_checkpoint=True, checkpoint_path=str(out / "model.pt"),
    )
    config_path = out / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = run_from_config(config_path, verbose=True)
    report = {"purpose": "infrastructure smoke test", "seed": args.seed,
              "python": platform.python_version(), "torch": torch.__version__,
              "device": config["training"]["device"],
              "evaluation": result.evaluation.__dict__}
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
