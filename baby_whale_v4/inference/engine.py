from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import mlx.core as mx

if TYPE_CHECKING:
    from baby_whale_v4.inference.batched import BatchedDecodeState

from baby_whale_v4.cache import DynamicKVCache
from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.inference.kv_offload import (
    KVOffloadReport,
    load_kv_cache_npz,
    save_kv_cache_npz,
)
from baby_whale_v4.inference.paged_kv import PagedKVCache, PagedKVPool
from baby_whale_v4.inference.prefix_cache import PrefixCache, PrefixCacheKey
from baby_whale_v4.inference.radix_cache import RadixKVCache
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.typing import (
    GenerationMode,
    RequestId,
    TokenizerHash,
    array_to_int_tuple,
    assert_never,
    ensure_in,
)

_GENERATION_MODES: tuple[GenerationMode, ...] = get_args(GenerationMode)

# Pure logit-filter helpers, compiled once with mx.compile to amortize MLX
# kernel dispatch overhead — which dominates wall-clock at tiny-model scale.
# Each takes a `[1, V]` logits tensor and returns the same shape with
# rejected positions set to -1e9 (so a subsequent ``mx.random.categorical``
# never selects them).


@mx.compile
def _filter_top_k(logits: mx.array, top_k: int) -> mx.array:
    values = mx.topk(logits, top_k, axis=-1)
    floor = mx.min(values, axis=-1, keepdims=True)
    return mx.where(logits < floor, mx.full(logits.shape, -1e9), logits)


@mx.compile
def _filter_top_p(logits: mx.array, top_p: float) -> mx.array:
    # Standard nucleus sampling: sort, take the smallest prefix whose
    # cumulative probability is ≥ top_p, mask the tail.
    sorted_logits = mx.sort(logits, axis=-1)  # ascending
    sorted_logits = sorted_logits[..., ::-1]  # descending
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumprobs = mx.cumsum(sorted_probs, axis=-1)
    # Anything strictly above the cutoff is dropped; the first position to
    # cross top_p stays so we always keep at least one token.
    mask = cumprobs - sorted_probs >= top_p
    cutoff_per_row = mx.where(mask, sorted_logits, mx.full(sorted_logits.shape, mx.inf))
    cutoff = mx.min(cutoff_per_row, axis=-1, keepdims=True)
    return mx.where(logits < cutoff, mx.full(logits.shape, -1e9), logits)


@mx.compile
def _filter_min_p(logits: mx.array, min_p: float) -> mx.array:
    # Min-p (Nguyen et al. 2024): keep tokens with prob ≥ min_p · p_max.
    # Operates on probabilities in a numerically-stable way.
    probs = mx.softmax(logits, axis=-1)
    threshold = min_p * mx.max(probs, axis=-1, keepdims=True)
    return mx.where(probs < threshold, mx.full(logits.shape, -1e9), logits)


@dataclass
class GenerationOptions:
    max_new_tokens: int = 32
    mode: GenerationMode = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    eos_id: int | None = None

    def __post_init__(self) -> None:
        ensure_in("mode", self.mode, _GENERATION_MODES)
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.min_p is not None and not (0.0 <= self.min_p < 1.0):
            raise ValueError("min_p must be in [0, 1)")


@dataclass
class RequestState:
    request_id: RequestId
    prompt_ids: list[int]
    options: GenerationOptions
    generated: list[int] = field(default_factory=list)
    captured_log_probs: list[float] = field(default_factory=list)
    cache: DynamicKVCache | PagedKVCache | None = None
    prefilled: int = 0
    last_logits: mx.array | None = None
    finished: bool = False
    used_prefix_cache: bool = False
    cancelled: bool = False

    def cancel(self) -> None:
        """Signal that no more work should be done on this request.

        The scheduler checks this flag once per tick; an in-flight kernel
        call is not interrupted but no further steps are scheduled. This is
        the SSE-disconnect / agent-timeout escape hatch.
        """
        self.cancelled = True
        self.finished = True

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if self.prefilled < 0:
            raise ValueError("prefilled must be non-negative")
        if self.prefilled > len(self.prompt_ids):
            raise ValueError("prefilled cannot exceed prompt length")

    @property
    def total_emitted(self) -> int:
        return len(self.generated)

    @property
    def remaining_prefill(self) -> int:
        return max(0, len(self.prompt_ids) - self.prefilled)

    @property
    def remaining_decode(self) -> int:
        return max(0, self.options.max_new_tokens - self.total_emitted)


class Engine:
    def __init__(
        self,
        *,
        model: BabyWhaleV4Model,
        config: BabyWhaleV4Config,
        tokenizer_hash: TokenizerHash,
        prefix_cache: PrefixCache | None = None,
        radix_cache: RadixKVCache | None = None,
        paged_pool: PagedKVPool | None = None,
    ):
        if sum(c is not None for c in (prefix_cache, radix_cache, paged_pool)) > 1:
            raise ValueError(
                "Engine accepts at most one of prefix_cache / radix_cache / "
                "paged_pool; they are distinct KV storage/reuse strategies."
            )
        if paged_pool is not None:
            self._validate_paged_pool(paged_pool, config)
        self.model = model
        self.config = config
        self.tokenizer_hash = tokenizer_hash
        self.prefix_cache = prefix_cache
        self.radix_cache = radix_cache
        self.paged_pool = paged_pool

    @staticmethod
    def _validate_paged_pool(pool: PagedKVPool, config: BabyWhaleV4Config) -> None:
        if "mla" in config.effective_layer_schedule:
            raise ValueError(
                "paged_pool is not supported for models with 'mla' layers "
                "(the paged cache has no latent path); use a DynamicKVCache."
            )
        pc = pool.config
        actual = (pc.n_layer, pc.n_heads, pc.head_dim)
        expected = (config.n_layer, config.n_kv_head, config.head_dim)
        if actual != expected:
            raise ValueError(
                "paged_pool dimensions do not match the model: pool "
                f"(n_layer, n_heads, head_dim)={actual}, model needs {expected} "
                "(n_heads must equal config.n_kv_head). Build it with "
                "PagedKVConfig.from_model_config(config)."
            )

    def _new_cache(self) -> DynamicKVCache | PagedKVCache:
        if self.paged_pool is not None:
            return PagedKVCache(pool=self.paged_pool)
        return self.model.empty_cache()

    @property
    def device(self) -> str:
        return self.model.device

    def new_request(
        self, request_id: RequestId, prompt_ids: list[int], options: GenerationOptions
    ) -> RequestState:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if len(prompt_ids) > self.config.context_length:
            raise ValueError("prompt exceeds context_length")
        if len(prompt_ids) + options.max_new_tokens > self.config.context_length:
            raise ValueError("prompt plus generation exceeds context_length")
        if options.top_k is not None and options.top_k > self.config.vocab_size:
            raise ValueError("top_k must be <= vocab_size")
        state = RequestState(
            request_id=request_id,
            prompt_ids=list(prompt_ids),
            options=options,
            cache=self._new_cache(),
        )
        self._try_prefix_cache_warm(state)
        return state

    def _try_prefix_cache_warm(self, state: RequestState) -> None:
        # Radix cache short-circuits the hash cache (and they're mutually
        # exclusive — see __init__ guard).
        if self.radix_cache is not None:
            hit = self.radix_cache.match(state.prompt_ids)
            if hit is not None:
                n_tokens, cache, last_logits = hit
                state.cache = cache
                state.prefilled = n_tokens
                state.last_logits = last_logits
                state.used_prefix_cache = True
            return
        if self.prefix_cache is None:
            return
        for n in range(len(state.prompt_ids), 0, -1):
            key = PrefixCacheKey.build(
                prefix_ids=state.prompt_ids[:n],
                config=self.config,
                tokenizer_hash=self.tokenizer_hash,
            )
            hit = self.prefix_cache.get(key)
            if hit is not None:
                n_tokens, cache, last_logits = hit
                state.cache = cache
                state.prefilled = n_tokens
                state.last_logits = last_logits
                state.used_prefix_cache = True
                return

    def prefill_chunk(self, state: RequestState, chunk_size: int) -> int:
        if state.finished:
            return 0
        remaining = state.remaining_prefill
        if remaining == 0:
            return 0
        n = min(chunk_size, remaining)
        start = state.prefilled
        end = start + n
        chunk_ids = mx.array([state.prompt_ids[start:end]], dtype=mx.int32)
        out = self.model(chunk_ids, cache=state.cache)
        state.prefilled = end
        state.last_logits = out.logits[:, -1, :]
        return n

    def commit_prefix_cache(self, state: RequestState) -> None:
        if state.cache is None or state.prefilled <= 0 or state.last_logits is None:
            return
        if not isinstance(state.cache, DynamicKVCache):
            # Paged pools don't participate in prefix/radix reuse (mutually
            # exclusive with them at construction) — nothing to commit.
            return
        if self.radix_cache is not None:
            self.radix_cache.insert(
                state.prompt_ids[: state.prefilled],
                state.cache,
                state.prefilled,
                state.last_logits,
            )
            return
        if self.prefix_cache is None:
            return
        key = PrefixCacheKey.build(
            prefix_ids=state.prompt_ids[: state.prefilled],
            config=self.config,
            tokenizer_hash=self.tokenizer_hash,
        )
        self.prefix_cache.put(key, state.cache, state.prefilled, state.last_logits)

    def decode_step(self, state: RequestState) -> int | None:
        if state.finished:
            return None
        if state.remaining_prefill > 0:
            raise RuntimeError("decode_step called before prefill complete")
        if state.last_logits is None:
            raise RuntimeError("no last_logits; prefill must run at least once")
        # Capture the rollout-time log-prob of the about-to-be-sampled token. RL
        # algorithms need π_old computed against the policy that actually drew
        # the sample, not against π_θ after the gradient step.
        scaled = state.last_logits / state.options.temperature
        log_norm = mx.logsumexp(scaled, axis=-1, keepdims=True)
        token = self._sample(state.last_logits, state.options)
        token_id = int(token[0])
        log_prob = float(scaled[0, token_id] - log_norm[0, 0])
        state.captured_log_probs.append(log_prob)
        state.generated.append(token_id)
        if state.options.eos_id is not None and token_id == state.options.eos_id:
            state.finished = True
            return token_id
        if state.total_emitted >= state.options.max_new_tokens:
            state.finished = True
        if not state.finished:
            inp = token.reshape(1, 1).astype(mx.int32)
            out = self.model(inp, cache=state.cache)
            state.last_logits = out.logits[:, -1, :]
        return token_id

    def _sample(self, logits: mx.array, options: GenerationOptions) -> mx.array:
        logits = logits / options.temperature
        match options.mode:
            case "greedy":
                return mx.argmax(logits, axis=-1).reshape(1)
            case "sample":
                # Apply truncation filters in vLLM's published order:
                # top-k → top-p → min-p. Each narrows the candidate set.
                if options.top_k is not None:
                    logits = _filter_top_k(logits, options.top_k)
                if options.top_p is not None:
                    logits = _filter_top_p(logits, options.top_p)
                if options.min_p is not None:
                    logits = _filter_min_p(logits, options.min_p)
                return mx.random.categorical(logits, axis=-1).reshape(1)
            case "speculative":
                raise ValueError(
                    "speculative decoding is not available in the per-step decode "
                    "loop; call Engine.generate(...) with mode='speculative', which "
                    "routes to model.spec_decode()"
                )
            case _:
                assert_never(options.mode)

    def generate(
        self,
        prompt_ids: list[int],
        options: GenerationOptions,
        *,
        request_id: str = "req",
    ) -> list[int]:
        state = self.generate_with_state(prompt_ids, options, request_id=request_id)
        tokens = list(state.generated)
        self._release(state)
        return tokens

    def generate_with_state(
        self,
        prompt_ids: list[int],
        options: GenerationOptions,
        *,
        request_id: str = "req",
    ) -> RequestState:
        """Run a full generation and return the final :class:`RequestState`.

        Callers that need the stop reason, captured log-probs, or the KV cache
        (the HTTP server's ``finish_reason``, RL rollout) use this; plain
        :meth:`generate` is the token-list convenience wrapper. When the engine
        uses a ``paged_pool`` the returned state still owns pool blocks — call
        ``state.cache.free()`` when done (``generate`` does this for you).
        """
        if options.mode == "speculative":
            return self._generate_speculative(prompt_ids, options, request_id=request_id)
        state = self.new_request(RequestId(request_id), prompt_ids, options)
        while state.remaining_prefill > 0:
            self.prefill_chunk(state, chunk_size=self.config.context_length)
        self.commit_prefix_cache(state)
        while not state.finished and state.remaining_decode > 0:
            self.decode_step(state)
        return state

    def _generate_speculative(
        self, prompt_ids: list[int], options: GenerationOptions, *, request_id: str
    ) -> RequestState:
        """Real MTP speculative decode via :meth:`BabyWhaleV4Model.spec_decode`.

        The stepwise engine loop can't draft (it emits one token per forward),
        so speculative decoding runs as a self-contained model call and the
        drafted tail is packed back into a RequestState for a uniform return
        type. Greedy speculative decode is token-identical to greedy sampling;
        the win is fewer forwards (see ``SpecDecodeResult.acceptance_rate``).
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        prefix = mx.array([list(prompt_ids)], dtype=mx.int32)
        result = self.model.spec_decode(prefix, max_new_tokens=options.max_new_tokens)
        generated = list(array_to_int_tuple(result.tokens[0, len(prompt_ids) :]))
        # spec_decode ignores EOS; truncate at the first EOS (inclusive) so the
        # output matches what the stepwise decoder would have emitted.
        if options.eos_id is not None and options.eos_id in generated:
            generated = generated[: generated.index(options.eos_id) + 1]
        state = RequestState(
            request_id=RequestId(request_id),
            prompt_ids=list(prompt_ids),
            options=options,
        )
        state.generated = generated
        state.finished = True
        return state

    def _release(self, state: RequestState) -> None:
        """Return a paged request's blocks to the pool. No-op for dense caches
        (GC reclaims them) and None caches (speculative)."""
        if isinstance(state.cache, PagedKVCache):
            state.cache.free()

    def fork(
        self,
        prompt_ids: list[int],
        n: int,
        options: GenerationOptions,
        *,
        request_id_prefix: str = "fork",
    ) -> list[RequestState]:
        """Generate ``n`` parallel completions of ``prompt_ids``.

        Prefills the prompt **once** (committing its KV to ``radix_cache``
        if present), then spawns ``n`` independent decode requests. With a
        radix cache attached, every branch hits the cached prompt KV — the
        prefill is shared, the decode diverges.

        This is the SGLang ``fork`` primitive surface: ideal for RL
        rollouts (sample N continuations from one prompt) and agent loops
        (one system prompt, many user-turn branches).
        """
        if n <= 0:
            raise ValueError("fork n must be positive")

        # Anchor: prefill once with a real request so the radix cache learns
        # the prompt KV. We then deliberately do *not* run its decode loop —
        # it's discarded after the cache write.
        anchor = self.new_request(RequestId(f"{request_id_prefix}-anchor"), prompt_ids, options)
        while anchor.remaining_prefill > 0:
            self.prefill_chunk(anchor, chunk_size=self.config.context_length)
        self.commit_prefix_cache(anchor)

        # Children: each is a fresh request that hits the radix cache for
        # the full prompt and starts decoding from the shared last_logits.
        branches: list[RequestState] = []
        for i in range(n):
            state = self.new_request(RequestId(f"{request_id_prefix}-{i}"), prompt_ids, options)
            branches.append(state)
        return branches

    def decode_step_group(self, states: list[RequestState]) -> list[int | None]:
        """Advance every state in ``states`` by one decode step.

        Semantically equivalent to calling ``decode_step(state)`` in a
        Python loop. Use :meth:`fork_batched` + ``decode_step_batched``
        for true single-forward batching across branches.
        """
        out: list[int | None] = []
        for state in states:
            out.append(self.decode_step(state))
        return out

    def fork_batched(
        self,
        prompt_ids: list[int],
        n: int,
        options: GenerationOptions,
        *,
        request_id_prefix: str = "fork",
    ) -> BatchedDecodeState:
        """Continuous-batched analog of :meth:`fork`.

        Prefills the prompt once at B=1, tiles the resulting KV cache to
        B=N, and returns a :class:`BatchedDecodeState` that lets
        ``decode_step_batched`` advance every branch with **one** batched
        forward call per step (instead of N).
        """

        from baby_whale_v4.inference.batched import BatchedDecodeState, tile_cache

        if n <= 0:
            raise ValueError("fork_batched n must be positive")

        anchor = self.new_request(RequestId(f"{request_id_prefix}-anchor"), prompt_ids, options)
        while anchor.remaining_prefill > 0:
            self.prefill_chunk(anchor, chunk_size=self.config.context_length)
        self.commit_prefix_cache(anchor)

        if anchor.cache is None or anchor.last_logits is None:
            raise RuntimeError("anchor prefill did not produce a usable cache")
        if not isinstance(anchor.cache, DynamicKVCache):
            raise TypeError("fork_batched requires a dense DynamicKVCache (no paged_pool)")
        batched_cache = tile_cache(anchor.cache, n)
        batched_last_logits = mx.concatenate([anchor.last_logits] * n, axis=0)
        return BatchedDecodeState(
            prompt_ids=list(prompt_ids),
            options=options,
            n_branches=n,
            cache=batched_cache,
            last_logits=batched_last_logits,
        )

    def offload_request(self, state: RequestState, path: Path | str) -> KVOffloadReport:
        """Snapshot a request's KV cache to disk (npz + manifest).

        Dense (:class:`DynamicKVCache`) only — paged requests live in the shared
        pool and are not offloaded per-request. Pair with :meth:`reload_request`
        to evict a long conversation's KV and resume it later.
        """
        if not isinstance(state.cache, DynamicKVCache):
            raise TypeError(
                "offload_request requires a DynamicKVCache; a paged request's KV "
                "lives in the shared pool and is not offloaded per-request"
            )
        return save_kv_cache_npz(state.cache, path)

    def reload_request(
        self,
        request_id: RequestId,
        prompt_ids: list[int],
        options: GenerationOptions,
        path: Path | str,
        *,
        prefilled: int,
        last_logits: mx.array,
    ) -> RequestState:
        """Rebuild a decode-ready :class:`RequestState` from an offloaded cache.

        ``prefilled`` and ``last_logits`` come from the pre-offload state — the
        KV blob alone doesn't carry the next-step logits — so decode resumes
        exactly where it left off.
        """
        cache = load_kv_cache_npz(path, expected_n_layer=self.config.n_layer)
        return RequestState(
            request_id=request_id,
            prompt_ids=list(prompt_ids),
            options=options,
            cache=cache,
            prefilled=prefilled,
            last_logits=last_logits,
        )
