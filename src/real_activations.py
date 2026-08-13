"""Pinned, resumable activation streams for the Stage-3 real-model sweep.

The public SAELens activation store is intentionally not modified here.  This
module supplies the one extra property needed by the long Stage-3 runs: the
shuffle buffer itself is serializable, so a rolling checkpoint resumes at the
exact stream position and serves the same already-buffered activation batches
rather than merely returning to a nearby dataset row.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import accumulate
import json
from typing import Any

import torch
from datasets import IterableDataset, load_dataset
from sae_lens.load_model import HookedProxyLM
from transformer_lens import HookedRootModule
from transformers import AutoModelForCausalLM, AutoTokenizer

from .real_activation_sweep import RealActivationDataConfig


ActivationChunkFactory = Callable[[int], Iterator[torch.Tensor]]


def _tensor_to_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().cpu().clone()


class ResumableActivationProvider(Iterator[torch.Tensor]):
    """Mix and batch a finite activation stream with exact buffered state.

    ``chunk_factory(offset)`` must deterministically reproduce the source and
    begin ``offset`` activations after its configured start.  The mixing rule
    matches SAELens' ``mixing_buffer`` while using a private RNG so unrelated
    model operations cannot perturb activation ordering.
    """

    def __init__(
        self,
        chunk_factory: ActivationChunkFactory,
        *,
        total_tokens: int,
        batch_size: int,
        d_in: int,
        device: torch.device | str,
        buffer_size: int,
        mix_fraction: float = 0.5,
        seed: int = 0,
        identity: Mapping[str, Any] | str,
    ) -> None:
        if total_tokens <= 0 or batch_size <= 0 or d_in <= 0:
            raise ValueError("total_tokens, batch_size, and d_in must be positive.")
        if total_tokens % batch_size:
            raise ValueError("total_tokens must be divisible by batch_size.")
        if buffer_size < batch_size:
            raise ValueError("buffer_size must be at least batch_size.")
        if not 0.0 <= mix_fraction <= 1.0:
            raise ValueError("mix_fraction must lie in [0, 1].")
        self.chunk_factory = chunk_factory
        self.total_tokens = int(total_tokens)
        self.batch_size = int(batch_size)
        self.d_in = int(d_in)
        self.device = torch.device(device)
        self.buffer_size = int(buffer_size)
        self.mix_fraction = float(mix_fraction)
        self.seed = int(seed)
        self.identity = (
            identity
            if isinstance(identity, str)
            else json.dumps(identity, sort_keys=True, separators=(",", ":"))
        )

        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self._source_consumed = 0
        self._tokens_yielded = 0
        self._source_exhausted = False
        self._source = iter(self.chunk_factory(0))
        self._storage_buffer: torch.Tensor | None = None
        self._serving_buffer: torch.Tensor | None = None
        self._serving_offset = 0

    def __iter__(self) -> ResumableActivationProvider:
        return self

    @property
    def source_tokens_consumed(self) -> int:
        return self._source_consumed

    @property
    def tokens_yielded(self) -> int:
        return self._tokens_yielded

    @torch.no_grad()
    def _read_chunk(self) -> torch.Tensor | None:
        if self._source_exhausted:
            return None
        try:
            chunk = next(self._source)
        except StopIteration:
            self._source_exhausted = True
            return None
        if chunk.ndim != 2 or chunk.shape[1] != self.d_in:
            raise ValueError(
                "Activation source returned shape "
                f"{tuple(chunk.shape)}; expected (tokens, {self.d_in})."
            )
        if chunk.shape[0] == 0:
            raise ValueError("Activation source returned an empty chunk.")
        remaining = self.total_tokens - self._source_consumed
        if chunk.shape[0] > remaining:
            chunk = chunk[:remaining]
        chunk = chunk.detach().to(self.device)
        self._source_consumed += int(chunk.shape[0])
        if self._source_consumed >= self.total_tokens:
            self._source_exhausted = True
        return chunk

    @torch.no_grad()
    def _prepare_serving_buffer(self) -> bool:
        while True:
            chunk = self._read_chunk()
            if chunk is not None:
                self._storage_buffer = (
                    chunk
                    if self._storage_buffer is None
                    else torch.cat((self._storage_buffer, chunk), dim=0)
                )
                if self._storage_buffer.shape[0] < self.buffer_size:
                    continue

                if self.mix_fraction > 0.0:
                    permutation = torch.randperm(
                        self._storage_buffer.shape[0], generator=self._generator
                    ).to(self._storage_buffer.device)
                    self._storage_buffer = self._storage_buffer[permutation]
                keep_for_mixing = int(self.buffer_size * self.mix_fraction)
                num_to_serve = self._storage_buffer.shape[0] - keep_for_mixing
                num_batches = max(1, num_to_serve // self.batch_size)
                cutoff = num_batches * self.batch_size
                self._serving_buffer = self._storage_buffer[:cutoff]
                self._storage_buffer = self._storage_buffer[cutoff:]
                self._serving_offset = 0
                return True

            if self._storage_buffer is None:
                return False
            num_batches = self._storage_buffer.shape[0] // self.batch_size
            if num_batches == 0:
                raise RuntimeError(
                    "Activation source ended with an incomplete batch; token "
                    "budget and source rows disagree."
                )
            cutoff = num_batches * self.batch_size
            self._serving_buffer = self._storage_buffer[:cutoff]
            remainder = self._storage_buffer[cutoff:]
            self._storage_buffer = remainder if remainder.numel() else None
            self._serving_offset = 0
            return True

    def __next__(self) -> torch.Tensor:
        if self._tokens_yielded >= self.total_tokens:
            raise StopIteration
        if (
            self._serving_buffer is None
            or self._serving_offset >= self._serving_buffer.shape[0]
        ):
            self._serving_buffer = None
            self._serving_offset = 0
            if not self._prepare_serving_buffer():
                raise RuntimeError(
                    "Activation source ended before the configured token budget."
                )
        assert self._serving_buffer is not None
        start = self._serving_offset
        stop = start + self.batch_size
        batch = self._serving_buffer[start:stop]
        if batch.shape[0] != self.batch_size:
            raise RuntimeError("Internal activation buffer produced a partial batch.")
        self._serving_offset = stop
        self._tokens_yielded += self.batch_size
        return batch

    def state_dict(self) -> dict[str, Any]:
        """Return CPU state sufficient to reproduce the next buffered batch."""

        remaining_serving = (
            None
            if self._serving_buffer is None
            else self._serving_buffer[self._serving_offset :]
        )
        if remaining_serving is not None and not remaining_serving.numel():
            remaining_serving = None
        return {
            "format_version": 1,
            "identity": self.identity,
            "total_tokens": self.total_tokens,
            "batch_size": self.batch_size,
            "d_in": self.d_in,
            "buffer_size": self.buffer_size,
            "mix_fraction": self.mix_fraction,
            "seed": self.seed,
            "source_consumed": self._source_consumed,
            "tokens_yielded": self._tokens_yielded,
            "source_exhausted": self._source_exhausted,
            "generator_state": self._generator.get_state().cpu(),
            "storage_buffer": _tensor_to_cpu(self._storage_buffer),
            # Drop the already-consumed prefix to bound rolling-checkpoint size.
            "serving_buffer": _tensor_to_cpu(remaining_serving),
            "serving_offset": 0,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""

        expected = {
            "format_version": 1,
            "identity": self.identity,
            "total_tokens": self.total_tokens,
            "batch_size": self.batch_size,
            "d_in": self.d_in,
            "buffer_size": self.buffer_size,
            "mix_fraction": self.mix_fraction,
            "seed": self.seed,
        }
        mismatched = [name for name, value in expected.items() if state.get(name) != value]
        if mismatched:
            raise ValueError(
                "Activation checkpoint does not match this stream: "
                + ", ".join(mismatched)
            )
        source_consumed = int(state["source_consumed"])
        tokens_yielded = int(state["tokens_yielded"])
        if not 0 <= tokens_yielded <= source_consumed <= self.total_tokens:
            raise ValueError("Activation checkpoint counters are inconsistent.")
        self._source_consumed = source_consumed
        self._tokens_yielded = tokens_yielded
        self._source_exhausted = bool(state["source_exhausted"])
        self._generator.set_state(state["generator_state"].cpu())
        storage = state.get("storage_buffer")
        serving = state.get("serving_buffer")
        self._storage_buffer = None if storage is None else storage.to(self.device)
        self._serving_buffer = None if serving is None else serving.to(self.device)
        self._serving_offset = int(state["serving_offset"])
        if self._serving_buffer is None and self._serving_offset:
            raise ValueError("Activation checkpoint has an invalid serving offset.")
        if (
            self._serving_buffer is not None
            and not 0 <= self._serving_offset <= self._serving_buffer.shape[0]
        ):
            raise ValueError("Activation checkpoint serving offset is out of range.")
        self._source = iter(self.chunk_factory(self._source_consumed))


def model_input_device(model: HookedRootModule) -> torch.device:
    """Return the device hosting a wrapped Hugging Face input embedding."""

    underlying = getattr(model, "model", model)
    if hasattr(underlying, "get_input_embeddings"):
        embeddings = underlying.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    return next(model.parameters()).device


def load_real_language_model(
    data: RealActivationDataConfig,
    device: torch.device | str,
    *,
    dtype: torch.dtype | None = None,
) -> HookedRootModule:
    """Load the pinned HF checkpoint through SAELens' proxy hook interface."""

    resolved_device = torch.device(device)
    if dtype is None:
        # The paper runner supplied no from_pretrained dtype override: weights
        # remain float32 and CUDA forward passes use the configured BF16
        # autocast.  Loading resident weights as BF16 would change activations.
        dtype = torch.float32
    if data.model_class_name != "AutoModelForCausalLM":
        raise ValueError("Stage-3 real activations require AutoModelForCausalLM.")
    hf_model = AutoModelForCausalLM.from_pretrained(
        data.model_id,
        revision=data.model_revision,
        torch_dtype=dtype,
    )
    hf_model.config.use_cache = False
    hf_model = hf_model.to(resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(
        data.model_id, revision=data.model_revision
    )
    model = HookedProxyLM(hf_model, tokenizer, hook_names=[data.hook_name])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _dataset_shard_position(
    data: RealActivationDataConfig, start_row: int
) -> tuple[int, int]:
    """Map one global row offset to its immutable Parquet shard and local row."""

    if start_row < 0:
        raise ValueError("start_row must be nonnegative.")
    cumulative = tuple(accumulate(data.dataset_shard_rows))
    if not cumulative or start_row >= cumulative[-1]:
        raise ValueError("start_row lies outside the pinned tokenized dataset.")
    shard_index = bisect_right(cumulative, start_row)
    prior_rows = 0 if shard_index == 0 else cumulative[shard_index - 1]
    return shard_index, start_row - prior_rows


def _dataset_shard_url(data: RealActivationDataConfig, index: int) -> str:
    path = data.dataset_shard_path_pattern.format(index=index)
    return (
        f"https://huggingface.co/datasets/{data.dataset_id}/resolve/"
        f"{data.dataset_revision}/{path}"
    )


def load_pretokenized_dataset(
    data: RealActivationDataConfig,
    *,
    start_row: int = 0,
) -> tuple[IterableDataset, int]:
    """Open only the pinned Parquet shard range that contains ``start_row``.

    The datasets contain 64 immutable ordered shards.  Starting at the
    containing shard bounds resume/evaluation seeking to at most one shard,
    while retaining byte-for-byte global row order and avoiding a local cache.
    """

    shard_index, local_row = _dataset_shard_position(data, start_row)
    data_files = [
        _dataset_shard_url(data, index)
        for index in range(shard_index, len(data.dataset_shard_rows))
    ]
    dataset = load_dataset(
        "parquet",
        data_files={data.dataset_split: data_files},
        split=data.dataset_split,
        streaming=True,
    )
    if not isinstance(dataset, IterableDataset):
        raise TypeError("Expected datasets.load_dataset(..., streaming=True) to stream.")
    return dataset, local_row


def _token_rows(
    data: RealActivationDataConfig,
    *,
    start_token: int,
    n_tokens: int,
    dataset: IterableDataset | None = None,
) -> Iterator[Sequence[int]]:
    if start_token < 0 or n_tokens < 0:
        raise ValueError("Token offsets and lengths must be nonnegative.")
    if start_token % data.context_size or n_tokens % data.context_size:
        raise ValueError("Pretokenized offsets and lengths must align to context_size.")
    rows_to_skip = start_token // data.context_size
    rows_to_take = n_tokens // data.context_size
    if dataset is None:
        stream, rows_to_skip = load_pretokenized_dataset(
            data, start_row=rows_to_skip
        )
    else:
        stream = dataset
    iterator = iter(stream.skip(rows_to_skip).take(rows_to_take))
    for index, row in enumerate(iterator):
        if "input_ids" not in row:
            raise ValueError(f"Dataset row {index} has no 'input_ids' field.")
        token_ids = row["input_ids"]
        if len(token_ids) != data.context_size:
            raise ValueError(
                f"Dataset row has {len(token_ids)} tokens; expected {data.context_size}."
            )
        yield token_ids


def iter_token_batches(
    data: RealActivationDataConfig,
    *,
    start_token: int,
    n_tokens: int,
    prompt_batch_size: int,
    device: torch.device | str,
    dataset: IterableDataset | None = None,
) -> Iterator[torch.Tensor]:
    """Yield pinned pretokenized contexts without adding another BOS token."""

    if prompt_batch_size <= 0:
        raise ValueError("prompt_batch_size must be positive.")
    rows = _token_rows(
        data,
        start_token=start_token,
        n_tokens=n_tokens,
        dataset=dataset,
    )
    batch: list[Sequence[int]] = []
    for token_ids in rows:
        batch.append(token_ids)
        if len(batch) == prompt_batch_size:
            yield torch.tensor(batch, dtype=torch.long, device=device)
            batch = []
    if batch:
        yield torch.tensor(batch, dtype=torch.long, device=device)


def live_activation_chunk_factory(
    data: RealActivationDataConfig,
    model: HookedRootModule,
    *,
    start_token: int,
    total_tokens: int,
    prompt_batch_size: int,
    activation_device: torch.device | str,
    autocast_lm: bool,
) -> ActivationChunkFactory:
    """Build a deterministic offset-addressable live-activation source."""

    if start_token % data.context_size or total_tokens % data.context_size:
        raise ValueError("Live activation ranges must align to context_size.")
    input_device = model_input_device(model)
    destination = torch.device(activation_device)

    def factory(consumed_tokens: int) -> Iterator[torch.Tensor]:
        if consumed_tokens < 0 or consumed_tokens > total_tokens:
            raise ValueError("Activation source offset is out of range.")
        if consumed_tokens % data.context_size:
            raise ValueError("Activation source offsets must align to context_size.")
        remaining = total_tokens - consumed_tokens
        token_batches = iter_token_batches(
            data,
            start_token=start_token + consumed_tokens,
            n_tokens=remaining,
            prompt_batch_size=prompt_batch_size,
            device=input_device,
        )
        for tokens in token_batches:
            with torch.inference_mode(), torch.autocast(
                device_type=input_device.type,
                dtype=torch.bfloat16,
                enabled=autocast_lm and input_device.type == "cuda",
            ):
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[data.hook_name],
                    stop_at_layer=data.layer + 1,
                    return_type="logits",
                    prepend_bos=False,
                )
            activations = cache[data.hook_name]
            if activations.ndim != 3 or activations.shape[-1] != data.input_dim:
                raise ValueError(
                    f"Hook {data.hook_name!r} returned {tuple(activations.shape)}; "
                    f"expected (prompts, context, {data.input_dim})."
                )
            # SAELens' live ActivationsStore copies LM outputs into its default
            # float32 stacked buffer before yielding optimizer batches.
            yield activations.detach().reshape(-1, data.input_dim).to(
                device=destination, dtype=torch.float32
            )

    return factory


def make_live_activation_provider(
    data: RealActivationDataConfig,
    model: HookedRootModule,
    *,
    start_token: int,
    total_tokens: int,
    batch_size: int,
    prompt_batch_size: int,
    n_batches_in_buffer: int,
    activation_device: torch.device | str,
    mix_fraction: float,
    seed: int,
    autocast_lm: bool,
) -> ResumableActivationProvider:
    """Create the finite provider used by Stage-3 training or evaluation."""

    if n_batches_in_buffer <= 0:
        raise ValueError("n_batches_in_buffer must be positive.")
    identity = {
        "model_id": data.model_id,
        "model_revision": data.model_revision,
        "hook_name": data.hook_name,
        "dataset_id": data.dataset_id,
        "dataset_revision": data.dataset_revision,
        "dataset_shard_path_pattern": data.dataset_shard_path_pattern,
        "dataset_shard_rows": data.dataset_shard_rows,
        "start_token": start_token,
        "total_tokens": total_tokens,
        "context_size": data.context_size,
    }
    return ResumableActivationProvider(
        live_activation_chunk_factory(
            data,
            model,
            start_token=start_token,
            total_tokens=total_tokens,
            prompt_batch_size=prompt_batch_size,
            activation_device=activation_device,
            autocast_lm=autocast_lm,
        ),
        total_tokens=total_tokens,
        batch_size=batch_size,
        d_in=data.input_dim,
        device=activation_device,
        # Match SAELens ActivationsStore: n_batches_in_buffer is counted in
        # context-length activation blocks, not optimizer batches.
        buffer_size=max(batch_size, n_batches_in_buffer * data.context_size),
        mix_fraction=mix_fraction,
        seed=seed,
        identity=identity,
    )


__all__ = [
    "ActivationChunkFactory",
    "ResumableActivationProvider",
    "iter_token_batches",
    "live_activation_chunk_factory",
    "load_pretokenized_dataset",
    "load_real_language_model",
    "make_live_activation_provider",
    "model_input_device",
]
