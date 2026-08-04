"""Small GPT-2 residual-stream activation cache utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .sae_data import center_and_normalize


@dataclass
class ActivationCacheConfig:
    model_name: str = "gpt2"
    layer: int = 8
    max_tokens: int = 4096
    batch_size: int = 4
    sequence_length: int = 64
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


def _fallback_texts() -> list[str]:
    return [
        "Sparse autoencoders decompose transformer activations into interpretable latent features.",
        "The residual stream is a shared communication channel for attention heads and MLPs.",
        "Variational support uncertainty can be measured with Bernoulli gate variance.",
        "A recoverability transition is not the same object as the true number of features.",
    ]


def load_text_samples(config: ActivationCacheConfig, max_rows: int = 2048) -> list[str]:
    try:
        from datasets import load_dataset

        dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.dataset_split)
        texts = [str(row["text"]) for row in dataset.select(range(min(max_rows, len(dataset)))) if row.get("text")]
        return texts or _fallback_texts()
    except Exception:
        return _fallback_texts()


def collect_gpt2_residual_activations(config: ActivationCacheConfig) -> torch.Tensor:
    """Collect post-block residual activations from GPT-2-style hidden states."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
    model.eval()

    texts = load_text_samples(config)
    activations: list[torch.Tensor] = []
    token_count = 0
    for start in range(0, len(texts), config.batch_size):
        batch_texts = texts[start : start + config.batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config.sequence_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[config.layer + 1]
        mask = encoded["attention_mask"].bool()
        flat = hidden[mask].detach().to("cpu", dtype=config.dtype)
        activations.append(flat)
        token_count += flat.shape[0]
        if token_count >= config.max_tokens:
            break
    if not activations:
        raise RuntimeError("No activations were collected.")
    return torch.cat(activations, dim=0)[: config.max_tokens]


def cache_gpt2_residual_activations(config: ActivationCacheConfig, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_raw = collect_gpt2_residual_activations(config)
    x, mean, scale = center_and_normalize(x_raw)
    torch.save(
        {
            "activations": x,
            "raw_mean": mean,
            "raw_scale": scale,
            "model_name": config.model_name,
            "layer": config.layer,
            "max_tokens": config.max_tokens,
            "sequence_length": config.sequence_length,
        },
        path,
    )
    return path


def load_activation_cache(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    if "activations" not in payload:
        raise KeyError("Activation cache must contain an 'activations' tensor.")
    return payload["activations"]
