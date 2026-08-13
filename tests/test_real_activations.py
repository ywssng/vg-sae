from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

import src.real_activations as real_activations
from src.real_activations import (
    ResumableActivationProvider,
    _dataset_shard_position,
    iter_token_batches,
    live_activation_chunk_factory,
    load_real_language_model,
    load_pretokenized_dataset,
)
from src.real_activation_sweep import target_data_config


def _factory(values: torch.Tensor, chunks: tuple[int, ...]):
    def make(offset: int):
        cursor = offset
        chunk_index = 0
        while cursor < values.shape[0]:
            width = chunks[chunk_index % len(chunks)]
            yield values[cursor : min(cursor + width, values.shape[0])]
            cursor += width
            chunk_index += 1

    return make


def _provider(values: torch.Tensor) -> ResumableActivationProvider:
    return ResumableActivationProvider(
        _factory(values, (6, 10, 8)),
        total_tokens=values.shape[0],
        batch_size=4,
        d_in=values.shape[1],
        device="cpu",
        buffer_size=12,
        mix_fraction=0.5,
        seed=17,
        identity={"fixture": "resume"},
    )


def test_resumable_activation_provider_restores_exact_next_batch() -> None:
    values = torch.arange(40 * 3, dtype=torch.float32).reshape(40, 3)
    uninterrupted = _provider(values)
    prefix = [next(uninterrupted).clone() for _ in range(3)]
    state = copy.deepcopy(uninterrupted.state_dict())
    expected_tail = [batch.clone() for batch in uninterrupted]

    resumed = _provider(values)
    resumed.load_state_dict(state)
    actual_tail = [batch.clone() for batch in resumed]

    assert len(prefix) + len(actual_tail) == 10
    assert len(actual_tail) == len(expected_tail)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_tail, expected_tail)
    )
    assert resumed.tokens_yielded == 40


def test_resumable_activation_provider_rejects_mismatched_identity() -> None:
    values = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    provider = ResumableActivationProvider(
        _factory(values, (4,)),
        total_tokens=8,
        batch_size=4,
        d_in=3,
        device="cpu",
        buffer_size=4,
        mix_fraction=0.0,
        identity="first",
    )
    state = provider.state_dict()
    other = ResumableActivationProvider(
        _factory(values, (4,)),
        total_tokens=8,
        batch_size=4,
        d_in=3,
        device="cpu",
        buffer_size=4,
        mix_fraction=0.0,
        identity="second",
    )
    with pytest.raises(ValueError, match="identity"):
        other.load_state_dict(state)


def test_resumable_activation_provider_requires_complete_batches() -> None:
    values = torch.zeros(6, 2)
    with pytest.raises(ValueError, match="divisible"):
        ResumableActivationProvider(
            _factory(values, (3,)),
            total_tokens=6,
            batch_size=4,
            d_in=2,
            device="cpu",
            buffer_size=4,
            identity="bad-budget",
        )


def test_iter_token_batches_uses_exact_pretokenized_row_range() -> None:
    data = replace(
        target_data_config("llama-3.2-1b-layer7"), context_size=4
    )
    dataset = Dataset.from_dict(
        {"input_ids": [[row * 4 + index for index in range(4)] for row in range(5)]}
    ).to_iterable_dataset()
    batches = list(
        iter_token_batches(
            data,
            start_token=4,
            n_tokens=12,
            prompt_batch_size=2,
            device="cpu",
            dataset=dataset,
        )
    )
    assert [batch.tolist() for batch in batches] == [
        [[4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15]],
    ]


def test_pinned_stream_seeks_to_containing_shard(monkeypatch) -> None:
    data = replace(
        target_data_config("llama-3.2-1b-layer7"),
        dataset_shard_rows=(3, 2, 4),
        dataset_shard_path_pattern="parts/{index:02d}.parquet",
    )
    captured: dict[str, object] = {}
    fixture = Dataset.from_dict(
        {"input_ids": [[row] * data.context_size for row in range(6)]}
    ).to_iterable_dataset()

    def fake_load_dataset(path, **kwargs):
        captured.update(path=path, **kwargs)
        return fixture

    monkeypatch.setattr(real_activations, "load_dataset", fake_load_dataset)
    stream, local_row = load_pretokenized_dataset(data, start_row=4)

    assert stream is fixture
    assert local_row == 1
    assert captured["path"] == "parquet"
    assert captured["data_files"] == {
        "train": [
            (
                "https://huggingface.co/datasets/"
                f"{data.dataset_id}/resolve/{data.dataset_revision}/parts/01.parquet"
            ),
            (
                "https://huggingface.co/datasets/"
                f"{data.dataset_id}/resolve/{data.dataset_revision}/parts/02.parquet"
            ),
        ]
    }
    assert _dataset_shard_position(data, 0) == (0, 0)
    assert _dataset_shard_position(data, 2) == (0, 2)
    assert _dataset_shard_position(data, 3) == (1, 0)
    assert _dataset_shard_position(data, 8) == (2, 3)
    with pytest.raises(ValueError, match="outside"):
        _dataset_shard_position(data, 9)


def test_live_activation_factory_captures_only_configured_hook(monkeypatch) -> None:
    data = replace(
        target_data_config("llama-3.2-1b-layer7"),
        context_size=4,
        input_dim=2,
        hook_name="target.hook",
        layer=3,
    )
    calls: list[dict[str, object]] = []

    def token_batches(*args, **kwargs):
        yield torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    class FakeModel:
        def run_with_cache(self, tokens, **kwargs):
            calls.append(kwargs)
            activations = torch.stack((tokens.float(), tokens.float() + 10), dim=-1)
            return None, {"target.hook": activations}

    monkeypatch.setattr(real_activations, "iter_token_batches", token_batches)
    monkeypatch.setattr(
        real_activations, "model_input_device", lambda model: torch.device("cpu")
    )
    factory = live_activation_chunk_factory(
        data,
        FakeModel(),  # type: ignore[arg-type]
        start_token=0,
        total_tokens=4,
        prompt_batch_size=1,
        activation_device="cpu",
        autocast_lm=False,
    )
    chunks = list(factory(0))
    assert chunks[0].shape == (4, 2)
    assert chunks[0].dtype == torch.float32
    assert calls == [
        {
            "names_filter": ["target.hook"],
            "stop_at_layer": 4,
            "return_type": "logits",
            "prepend_bos": False,
        }
    ]


def test_language_model_load_keeps_reference_float32_weights(monkeypatch) -> None:
    data = target_data_config("llama-3.2-1b-layer7")
    captured: dict[str, object] = {}

    class FakeHFModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(use_cache=True)

    hf_model = FakeHFModel()

    def fake_model_load(model_id, **kwargs):
        captured.update(model_id=model_id, model_kwargs=kwargs)
        return hf_model

    def fake_tokenizer_load(model_id, **kwargs):
        captured.update(tokenizer_id=model_id, tokenizer_kwargs=kwargs)
        return object()

    class FakeProxy(torch.nn.Module):
        def __init__(self, model, tokenizer, hook_names):
            super().__init__()
            self.model = model
            captured.update(tokenizer=tokenizer, hook_names=hook_names)

    monkeypatch.setattr(
        real_activations.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_load,
    )
    monkeypatch.setattr(
        real_activations.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_load,
    )
    monkeypatch.setattr(real_activations, "HookedProxyLM", FakeProxy)

    model = load_real_language_model(data, "cpu")

    assert captured["model_id"] == data.model_id
    assert captured["model_kwargs"] == {
        "revision": data.model_revision,
        "torch_dtype": torch.float32,
    }
    assert captured["tokenizer_kwargs"] == {"revision": data.model_revision}
    assert captured["hook_names"] == [data.hook_name]
    assert hf_model.config.use_cache is False
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
