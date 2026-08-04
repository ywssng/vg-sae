"""Cache GPT-2 small layer-8 residual-stream activations for VG-SAE sweeps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gpt2_activations import ActivationCacheConfig, cache_gpt2_residual_activations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache GPT-2 residual-stream activations.")
    parser.add_argument("--output", default="outputs/gpt2/gpt2_layer8_resid.pt")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = cache_gpt2_residual_activations(
        ActivationCacheConfig(
            model_name=args.model_name,
            layer=args.layer,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            seed=args.seed,
            device=args.device,
        ),
        args.output,
    )
    print(path)


if __name__ == "__main__":
    main()
