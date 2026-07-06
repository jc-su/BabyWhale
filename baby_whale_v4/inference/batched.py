"""Continuous batched decoding for ``Engine.fork``.

After ``fork()`` prefills a shared prompt at B=1, this module tiles the
resulting :class:`DynamicKVCache` to B=N and runs a single batched forward
pass per decode step instead of N independent ones. The win at our scale
is kernel-dispatch reduction: where N separate ``model(inp, cache)`` calls
each pay MLX launch overhead, one batched ``model(inp, cache)`` with
``inp.shape == [N, 1]`` pays once. The educational point is the SGLang
``fork`` pattern fully realized — prefill once, decode N branches
together.

What this implementation does NOT yet do:
* Drop finished rows mid-batch. A branch that hits EOS continues to occupy
  a cache row (no further tokens are emitted for it) until the entire
  group is done. Production engines repack the active set; we keep it
  simple at the cost of decoding "filler" forward passes once a single
  branch finishes early.
* Ragged sequence lengths. All branches must share the same prompt prefix
  length, which is the case for ``fork``-style group rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from baby_whale_v4.cache import DynamicKVCache
from baby_whale_v4.inference.engine import (
    GenerationOptions,
    RequestState,
    _filter_min_p,
    _filter_top_k,
    _filter_top_p,
)
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.typing import array_to_int_tuple


def tile_cache(cache: DynamicKVCache, n: int) -> DynamicKVCache:
    """Tile a ``B=1`` cache to ``B=n`` along the batch dim.

    Concatenation (vs broadcast) materializes the tensor so downstream
    ``append`` ops grow each row independently — exactly what we want once
    branches start diverging on token IDs.
    """
    if n <= 0:
        raise ValueError("tile_cache n must be positive")
    new_keys: list[mx.array | None] = []
    new_values: list[mx.array | None] = []
    new_latents: list[mx.array | None] = []
    for k, v in zip(cache.keys, cache.values, strict=True):
        if k is None or v is None:
            new_keys.append(None)
            new_values.append(None)
            continue
        if k.shape[0] != 1:
            raise ValueError(f"tile_cache expects B=1 input, got batch={k.shape[0]}")
        new_keys.append(mx.concatenate([k] * n, axis=0))
        new_values.append(mx.concatenate([v] * n, axis=0))
    latents = cache.latents or [None] * len(cache.keys)
    for lat in latents:
        if lat is None:
            new_latents.append(None)
        else:
            if lat.shape[0] != 1:
                raise ValueError("tile_cache expects B=1 latents")
            new_latents.append(mx.concatenate([lat] * n, axis=0))
    return DynamicKVCache(keys=new_keys, values=new_values, latents=new_latents)


@dataclass
class BatchedDecodeState:
    """N-branch decode state sharing a single batched ``DynamicKVCache``.

    Created by ``Engine.fork_batched()``. Each step advances every branch
    in parallel; per-branch outputs accumulate in ``generated`` and
    ``captured_log_probs``. A branch is "finished" once it hit EOS or its
    ``max_new_tokens`` budget; further steps skip token-emission for it
    but still run its forward (until the whole group finishes).
    """

    prompt_ids: list[int]
    options: GenerationOptions
    n_branches: int
    cache: DynamicKVCache
    last_logits: mx.array  # [N, V]
    generated: list[list[int]] = field(default_factory=list)
    captured_log_probs: list[list[float]] = field(default_factory=list)
    finished: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_branches <= 0:
            raise ValueError("n_branches must be positive")
        if self.last_logits.ndim != 2 or self.last_logits.shape[0] != self.n_branches:
            raise ValueError(
                f"last_logits must be shape [N={self.n_branches}, V]; "
                f"got {tuple(self.last_logits.shape)}"
            )
        if not self.generated:
            self.generated = [[] for _ in range(self.n_branches)]
        if not self.captured_log_probs:
            self.captured_log_probs = [[] for _ in range(self.n_branches)]
        if not self.finished:
            self.finished = [False] * self.n_branches

    @property
    def all_finished(self) -> bool:
        return all(self.finished)

    def total_emitted(self) -> list[int]:
        return [len(g) for g in self.generated]


def decode_step_batched(model: BabyWhaleV4Model, state: BatchedDecodeState) -> None:
    """Advance every branch by one token via a single batched forward.

    Even after some branches finish, the batch keeps its row (the cache
    can't lose a row without rebuilding). The finished branches still
    decode forward but their emitted tokens are discarded.
    """
    if state.all_finished:
        return

    opts = state.options
    scaled = state.last_logits / opts.temperature
    log_norm = mx.logsumexp(scaled, axis=-1, keepdims=True)  # [N, 1]

    if opts.mode == "greedy":
        tokens = mx.argmax(scaled, axis=-1)  # [N]
    elif opts.mode == "sample":
        logits = scaled
        if opts.top_k is not None:
            logits = _filter_top_k(logits, opts.top_k)
        if opts.top_p is not None:
            logits = _filter_top_p(logits, opts.top_p)
        if opts.min_p is not None:
            logits = _filter_min_p(logits, opts.min_p)
        tokens = mx.random.categorical(logits, axis=-1)  # [N]
    else:
        # "speculative" decoding has no batched path here; use deterministic greedy decode.
        tokens = mx.argmax(scaled, axis=-1)

    # Materialize the per-branch token IDs and log-probs for bookkeeping.
    tok_ids: list[int] = list(array_to_int_tuple(tokens))
    log_probs_per_branch = [float(scaled[i, tid] - log_norm[i, 0]) for i, tid in enumerate(tok_ids)]
    for i, tid in enumerate(tok_ids):
        if state.finished[i]:
            continue
        state.generated[i].append(tid)
        state.captured_log_probs[i].append(log_probs_per_branch[i])
        if opts.eos_id is not None and tid == opts.eos_id:
            state.finished[i] = True
            continue
        if len(state.generated[i]) >= opts.max_new_tokens:
            state.finished[i] = True

    if state.all_finished:
        return

    # One batched forward: shape [N, 1] in, logits shape [N, 1, V] out.
    inp = tokens.reshape(state.n_branches, 1).astype(mx.int32)
    out = model(inp, cache=state.cache)
    state.last_logits = out.logits[:, -1, :]


def generate_batched(
    model: BabyWhaleV4Model, state: BatchedDecodeState, *, max_steps: int = 10_000
) -> BatchedDecodeState:
    """Run :func:`decode_step_batched` until every branch is finished."""
    for _ in range(max_steps):
        if state.all_finished:
            return state
        decode_step_batched(model, state)
    if not state.all_finished:
        raise RuntimeError("generate_batched exceeded max_steps")
    return state


def _stack_caches(caches: list[DynamicKVCache]) -> DynamicKVCache:
    """Stack N same-length ``B=1`` caches into one ``B=N`` cache along the batch dim.

    Every cache must share identical per-layer shapes (same sequence length) —
    the cohort invariant the scheduler enforces before batching.
    """
    n_layer = len(caches[0].keys)
    per_latents = [c.latents or [None] * len(c.keys) for c in caches]
    keys: list[mx.array | None] = []
    values: list[mx.array | None] = []
    latents: list[mx.array | None] = []
    for layer in range(n_layer):
        layer_keys = [c.keys[layer] for c in caches]
        layer_values = [c.values[layer] for c in caches]
        if layer_keys[0] is None:
            keys.append(None)
            values.append(None)
        else:
            keys.append(mx.concatenate([k for k in layer_keys if k is not None], axis=0))
            values.append(mx.concatenate([v for v in layer_values if v is not None], axis=0))
        layer_latents = [per_latents[j][layer] for j in range(len(caches))]
        if layer_latents[0] is None:
            latents.append(None)
        else:
            latents.append(
                mx.concatenate([lat for lat in layer_latents if lat is not None], axis=0)
            )
    return DynamicKVCache(keys=keys, values=values, latents=latents)


def _slice_cache(cache: DynamicKVCache, i: int) -> DynamicKVCache:
    """Extract row ``i`` of a ``B=N`` cache back into a ``B=1`` cache."""
    latents = cache.latents or [None] * len(cache.keys)
    return DynamicKVCache(
        keys=[None if k is None else k[i : i + 1] for k in cache.keys],
        values=[None if v is None else v[i : i + 1] for v in cache.values],
        latents=[None if lat is None else lat[i : i + 1] for lat in latents],
    )


def decode_group_batched(model: BabyWhaleV4Model, states: list[RequestState]) -> None:
    """Advance a cohort of same-length ``RequestState``s by one token via ONE
    batched forward, scattering the new token / KV / logits back to each state.

    Requires every state to share the same sequence length and sampling params
    (the scheduler groups on exactly that; uniform length ⇒ identical positions
    and attention mask across the batch, so no model changes are needed). Greedy
    output is token-identical to per-request :meth:`Engine.decode_step`; sampling
    differs only because the batched ``mx.random.categorical`` draws once for the
    batch instead of once per request.
    """
    n = len(states)
    if n == 0:
        return
    opts = states[0].options
    if opts.mode == "speculative":
        raise ValueError("batched decode does not support speculative mode")

    dense_caches: list[DynamicKVCache] = []
    logits_rows: list[mx.array] = []
    for state in states:
        if state.last_logits is None:
            raise RuntimeError("decode_group_batched requires prefilled states")
        if not isinstance(state.cache, DynamicKVCache):
            raise TypeError("batched decode requires DynamicKVCache states")
        dense_caches.append(state.cache)
        logits_rows.append(state.last_logits)

    scaled = mx.concatenate(logits_rows, axis=0) / opts.temperature  # [N, V]
    log_norm = mx.logsumexp(scaled, axis=-1, keepdims=True)  # [N, 1]
    if opts.mode == "greedy":
        tokens = mx.argmax(scaled, axis=-1)
    else:
        logits = scaled
        if opts.top_k is not None:
            logits = _filter_top_k(logits, opts.top_k)
        if opts.top_p is not None:
            logits = _filter_top_p(logits, opts.top_p)
        if opts.min_p is not None:
            logits = _filter_min_p(logits, opts.min_p)
        tokens = mx.random.categorical(logits, axis=-1)
    tok_ids = list(array_to_int_tuple(tokens))

    for i, state in enumerate(states):
        tid = tok_ids[i]
        state.generated.append(tid)
        state.captured_log_probs.append(float(scaled[i, tid] - log_norm[i, 0]))
        if (state.options.eos_id is not None and tid == state.options.eos_id) or (
            state.total_emitted >= state.options.max_new_tokens
        ):
            state.finished = True

    # One batched forward advances every row's cache (finished rows included —
    # the scheduler drops them from the cohort on the next tick).
    batched_cache = _stack_caches(dense_caches)
    inp = tokens.reshape(n, 1).astype(mx.int32)
    out = model(inp, cache=batched_cache)
    new_last = out.logits[:, -1, :]
    for i, state in enumerate(states):
        state.cache = _slice_cache(batched_cache, i)
        state.last_logits = new_last[i : i + 1]


def _leftpad_stack_caches(caches: list[DynamicKVCache], lengths: list[int]) -> DynamicKVCache:
    """Left-pad each request's KV to the max length and stack to ``B=N``.

    Left-padding aligns every row at the right edge: an old key at column ``c``
    keeps the rotation it was given at its own position, and (because RoPE is
    relative) the query-vs-key relative positions come out uniform per column, so
    no key needs re-rotating. Padding columns are zeros, masked out at attention.
    """
    t_max = max(lengths)
    n_layer = len(caches[0].keys)
    keys: list[mx.array | None] = []
    values: list[mx.array | None] = []
    for layer in range(n_layer):
        rows_k: list[mx.array] = []
        rows_v: list[mx.array] = []
        for cache, length in zip(caches, lengths, strict=True):
            k = cache.keys[layer]
            v = cache.values[layer]
            if k is None or v is None:
                raise ValueError("ragged batched decode needs a fully-populated dense cache")
            if length < t_max:
                pad = mx.zeros((1, k.shape[1], t_max - length, k.shape[3]), dtype=k.dtype)
                k = mx.concatenate([pad, k], axis=2)
                v = mx.concatenate([mx.zeros_like(pad), v], axis=2)
            rows_k.append(k)
            rows_v.append(v)
        keys.append(mx.concatenate(rows_k, axis=0))
        values.append(mx.concatenate(rows_v, axis=0))
    return DynamicKVCache(keys=keys, values=values, latents=[None] * n_layer)


def _slice_unpad_cache(cache: DynamicKVCache, i: int, true_len: int) -> DynamicKVCache:
    """Row ``i`` of a batched cache, keeping only its rightmost ``true_len`` columns."""
    return DynamicKVCache(
        keys=[None if k is None else k[i : i + 1, :, -true_len:, :] for k in cache.keys],
        values=[None if v is None else v[i : i + 1, :, -true_len:, :] for v in cache.values],
        latents=[None] * len(cache.keys),
    )


def decode_ragged_batched(model: BabyWhaleV4Model, states: list[RequestState]) -> None:
    """Advance a cohort of *different-length* RequestStates by one token via ONE
    batched forward — the mixed-length generalization of :func:`decode_group_batched`.

    Left-pads the per-request caches to a common length, rotates each row's new
    query at its own position, and masks each row to its own causal + sliding
    window. Greedy output is token-identical to per-request decode. Requires an
    all-``sliding_mqa`` model (the compressed/latent attention variants have no
    ragged-mask path and raise).
    """
    n = len(states)
    if n == 0:
        return
    schedule = model.config.effective_layer_schedule
    if any(kind != "sliding_mqa" for kind in schedule):
        raise ValueError("ragged batched decode requires an all-sliding_mqa model")
    opts = states[0].options
    if opts.mode == "speculative":
        raise ValueError("batched decode does not support speculative mode")

    dense_caches: list[DynamicKVCache] = []
    logits_rows: list[mx.array] = []
    lengths: list[int] = []
    for state in states:
        if state.last_logits is None:
            raise RuntimeError("decode_ragged_batched requires prefilled states")
        if not isinstance(state.cache, DynamicKVCache):
            raise TypeError("batched decode requires DynamicKVCache states")
        dense_caches.append(state.cache)
        logits_rows.append(state.last_logits)
        lengths.append(state.cache.max_sequence_length())

    scaled = mx.concatenate(logits_rows, axis=0) / opts.temperature  # [N, V]
    log_norm = mx.logsumexp(scaled, axis=-1, keepdims=True)
    if opts.mode == "greedy":
        tokens = mx.argmax(scaled, axis=-1)
    else:
        logits = scaled
        if opts.top_k is not None:
            logits = _filter_top_k(logits, opts.top_k)
        if opts.top_p is not None:
            logits = _filter_top_p(logits, opts.top_p)
        if opts.min_p is not None:
            logits = _filter_min_p(logits, opts.min_p)
        tokens = mx.random.categorical(logits, axis=-1)
    tok_ids = list(array_to_int_tuple(tokens))
    for i, state in enumerate(states):
        tid = tok_ids[i]
        state.generated.append(tid)
        state.captured_log_probs.append(float(scaled[i, tid] - log_norm[i, 0]))
        if (state.options.eos_id is not None and tid == state.options.eos_id) or (
            state.total_emitted >= state.options.max_new_tokens
        ):
            state.finished = True

    # Per-row new-token positions (its own length), and a per-row causal+sliding
    # mask over the appended (T_max + 1)-length key axis.
    t_max = max(lengths)
    t_total = t_max + 1
    positions = mx.array([[length] for length in lengths], dtype=mx.int32)  # [N, 1]
    lengths_after = mx.array([length + 1 for length in lengths], dtype=mx.int32)  # [N]
    attend = mx.minimum(lengths_after, model.config.sliding_window)
    start = t_total - attend  # [N]
    cols = mx.arange(t_total)[None, :]
    key_mask = (cols >= start[:, None])[:, None, None, :]  # [N, 1, 1, t_total] bool

    batched_cache = _leftpad_stack_caches(dense_caches, lengths)
    inp = tokens.reshape(n, 1).astype(mx.int32)
    out = model(inp, cache=batched_cache, positions=positions, key_mask=key_mask)
    new_last = out.logits[:, -1, :]
    for i, state in enumerate(states):
        state.cache = _slice_unpad_cache(batched_cache, i, lengths[i] + 1)
        state.last_logits = new_last[i : i + 1]
