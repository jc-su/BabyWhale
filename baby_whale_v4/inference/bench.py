from dataclasses import dataclass
from time import perf_counter

from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.inference.engine import Engine, GenerationOptions
from baby_whale_v4.inference.paged_kv import PagedKVConfig, PagedKVPool
from baby_whale_v4.inference.prefix_cache import PrefixCache
from baby_whale_v4.inference.scheduler import RequestScheduler
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.quantization import apply_weight_quantization
from baby_whale_v4.typing import QuantMode, TokenizerHash


@dataclass(frozen=True)
class InferenceBenchmark:
    requests: int
    prompt_tokens: int
    generated_tokens: int
    total_ms: float
    decode_tokens_per_sec: float
    prefill_steps: int
    decode_steps: int
    completed: int
    prefix_cache_hits: int
    prefix_cache_misses: int

    def __post_init__(self) -> None:
        if self.requests <= 0:
            raise ValueError("benchmark requests must be positive")
        if self.prompt_tokens <= 0:
            raise ValueError("benchmark prompt_tokens must be positive")
        if self.generated_tokens < 0:
            raise ValueError("benchmark generated_tokens must be non-negative")
        if self.total_ms <= 0:
            raise ValueError("benchmark total_ms must be positive")
        if self.decode_tokens_per_sec < 0:
            raise ValueError("decode_tokens_per_sec must be non-negative")


def benchmark_scheduler(
    *,
    engine: Engine,
    prompts: list[list[int]],
    options: GenerationOptions,
    prefill_chunk: int,
) -> InferenceBenchmark:
    if not prompts:
        raise ValueError("benchmark prompts must be non-empty")
    scheduler = RequestScheduler(engine, prefill_chunk=prefill_chunk)
    for idx, prompt in enumerate(prompts):
        scheduler.submit(f"bench-{idx}", prompt, options)
    prefix_cache = engine.prefix_cache
    hits_before = 0 if prefix_cache is None else prefix_cache.hits
    misses_before = 0 if prefix_cache is None else prefix_cache.misses
    start = perf_counter()
    completed = scheduler.run_until_done()
    elapsed_ms = (perf_counter() - start) * 1000.0
    generated_tokens = sum(state.total_emitted for state in completed)
    elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
    hits_after = 0 if prefix_cache is None else prefix_cache.hits
    misses_after = 0 if prefix_cache is None else prefix_cache.misses
    return InferenceBenchmark(
        requests=len(prompts),
        prompt_tokens=sum(len(prompt) for prompt in prompts),
        generated_tokens=generated_tokens,
        total_ms=elapsed_ms,
        decode_tokens_per_sec=generated_tokens / elapsed_s,
        prefill_steps=scheduler.stats.prefill_steps,
        decode_steps=scheduler.stats.decode_steps,
        completed=scheduler.stats.completed,
        prefix_cache_hits=hits_after - hits_before,
        prefix_cache_misses=misses_after - misses_before,
    )


@dataclass(frozen=True)
class InferenceComparison:
    """One :class:`InferenceBenchmark` per named config, run on one prompt suite."""

    rows: tuple[tuple[str, InferenceBenchmark], ...]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("InferenceComparison must have at least one row")

    def as_rows(self) -> list[dict[str, object]]:
        return [
            {
                "config": name,
                "decode_tokens_per_sec": round(bench.decode_tokens_per_sec, 2),
                "total_ms": round(bench.total_ms, 2),
                "generated_tokens": bench.generated_tokens,
                "prefix_cache_hits": bench.prefix_cache_hits,
                "prefix_cache_misses": bench.prefix_cache_misses,
            }
            for name, bench in self.rows
        ]

    def fastest(self) -> str:
        return max(self.rows, key=lambda row: row[1].decode_tokens_per_sec)[0]


def compare_inference_configs(
    *,
    model: BabyWhaleV4Model,
    config: BabyWhaleV4Config,
    tokenizer_hash: TokenizerHash,
    prompts: list[list[int]],
    options: GenerationOptions,
    prefill_chunk: int = 4,
    prefix_capacity: int = 64,
    quant_modes: tuple[QuantMode, ...] = (),
) -> InferenceComparison:
    """Run the same prompt suite under several inference configs (milestone #4).

    Always compares the KV strategies — no reuse, hash prefix cache, and (for
    non-MLA models) the paged pool — on one prompt set so decode throughput and
    cache hit-rates are directly comparable. Any weight-only ``quant_modes`` are
    additionally benchmarked by mutating the shared model in place then resetting
    it to ``none``.
    """
    if not prompts:
        raise ValueError("compare_inference_configs requires a non-empty prompt suite")

    def _bench(engine: Engine) -> InferenceBenchmark:
        return benchmark_scheduler(
            engine=engine, prompts=prompts, options=options, prefill_chunk=prefill_chunk
        )

    rows: list[tuple[str, InferenceBenchmark]] = [
        ("no-cache", _bench(Engine(model=model, config=config, tokenizer_hash=tokenizer_hash))),
        (
            "prefix-cache",
            _bench(
                Engine(
                    model=model,
                    config=config,
                    tokenizer_hash=tokenizer_hash,
                    prefix_cache=PrefixCache(capacity=prefix_capacity),
                )
            ),
        ),
    ]
    if "mla" not in config.effective_layer_schedule:
        pool = PagedKVPool(PagedKVConfig.from_model_config(config, block_size=16, n_blocks=256))
        rows.append(
            (
                "paged",
                _bench(
                    Engine(
                        model=model,
                        config=config,
                        tokenizer_hash=tokenizer_hash,
                        paged_pool=pool,
                    )
                ),
            )
        )
    for qmode in quant_modes:
        if qmode == "none":
            continue
        apply_weight_quantization(model, qmode)
        try:
            rows.append(
                (
                    f"quant:{qmode}",
                    _bench(Engine(model=model, config=config, tokenizer_hash=tokenizer_hash)),
                )
            )
        finally:
            apply_weight_quantization(model, "none")
    return InferenceComparison(rows=tuple(rows))
