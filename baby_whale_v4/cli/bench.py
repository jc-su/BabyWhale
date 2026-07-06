import argparse
import json

from baby_whale_v4.cli._common import (
    FP4_CACHE_CHOICES,
    FP4_GRAD_CHOICES,
    FP4_MASTER_DTYPE_CHOICES,
    QUANT_CHOICES,
)


def _bench_fp4_training(args: argparse.Namespace) -> None:
    from baby_whale_v4.training.fp4_native import benchmark_custom_vjp_fp4_training

    bench = benchmark_custom_vjp_fp4_training(
        batch=args.batch_size,
        input_dims=args.input_dims,
        output_dims=args.output_dims,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        max_ratio=args.max_ratio,
        weight_grad=args.weight_grad,
        cache_policy=args.cache_policy,
    )
    print(
        json.dumps(
            {
                "dense_ms": bench.dense_ms,
                "fp4_ms": bench.fp4_ms,
                "ratio": bench.ratio,
                "max_ratio": bench.max_ratio,
                "weight_grad": args.weight_grad,
                "cache_policy": args.cache_policy,
                "passed": bench.passed,
            },
            indent=2,
        )
    )


def _bench_fp4_memory(args: argparse.Namespace) -> None:
    from baby_whale_v4.training.fp4_native import benchmark_fp4_training_memory

    bench = benchmark_fp4_training_memory(
        batch=args.batch_size,
        input_dims=args.input_dims,
        output_dims=args.output_dims,
        baseline=args.baseline,
        max_peak_ratio=args.max_peak_ratio,
        weight_grad=args.weight_grad,
        cache_policy=args.cache_policy,
        optimizer=args.optimizer,
        fp4_master_dtype=args.fp4_master_dtype,
    )
    print(
        json.dumps(
            {
                "dense_fp32_peak_bytes": bench.dense_fp32_peak_bytes,
                "dense_fp32_active_bytes": bench.dense_fp32_active_bytes,
                "dense_bf16_peak_bytes": bench.dense_bf16_peak_bytes,
                "dense_bf16_active_bytes": bench.dense_bf16_active_bytes,
                "fp4_peak_bytes": bench.fp4_peak_bytes,
                "fp4_active_bytes": bench.fp4_active_bytes,
                "baseline": bench.baseline,
                "baseline_peak_bytes": bench.baseline_peak_bytes,
                "baseline_active_bytes": bench.baseline_active_bytes,
                "peak_ratio": bench.peak_ratio,
                "active_ratio": bench.active_ratio,
                "max_peak_ratio": bench.max_peak_ratio,
                "weight_grad": bench.weight_grad,
                "cache_policy": bench.cache_policy,
                "optimizer": bench.optimizer,
                "fp4_master_dtype": bench.fp4_master_dtype,
                "passed": bench.passed,
            },
            indent=2,
        )
    )


def _bench_rollout(args: argparse.Namespace) -> None:
    import time

    import mlx.core as mx

    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.data import ByteTokenizer
    from baby_whale_v4.inference.engine import GenerationOptions
    from baby_whale_v4.rl import InProcessRolloutEngine, RolloutRequest

    mx.random.seed(0)
    tok = ByteTokenizer()
    cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=args.context_length)
    model = BabyWhaleV4Model(cfg)
    model.eval()
    engine = InProcessRolloutEngine(
        model=model,
        config=cfg,
        tokenizer_hash=tok.hash_signature(),
        prefix_cache_capacity=args.prefix_cache_capacity,
        prefill_chunk=args.prefill_chunk,
    )

    prompt_ids = tuple(tok.encode(args.prompt))
    options = GenerationOptions(max_new_tokens=args.max_new_tokens, mode="sample")
    requests = [
        RolloutRequest(prompt_ids=prompt_ids, options=options) for _ in range(args.group_size)
    ]

    start = time.perf_counter()
    samples = engine.generate_batch(requests)
    total_ms = (time.perf_counter() - start) * 1000.0
    total_decode_tokens = sum(len(s.response_ids) for s in samples)
    print(
        json.dumps(
            {
                "group_size": args.group_size,
                "prompt_tokens": len(prompt_ids),
                "decode_tokens_total": total_decode_tokens,
                "total_ms": total_ms,
                "decode_tokens_per_sec": (total_decode_tokens / total_ms * 1000.0)
                if total_ms > 0
                else 0.0,
                "prefix_cache_hits": engine.prefix_cache.hits,
                "prefix_cache_misses": engine.prefix_cache.misses,
            },
            indent=2,
        )
    )


def _bench_inference(args: argparse.Namespace) -> None:
    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.data import ByteTokenizer
    from baby_whale_v4.inference import Engine, GenerationOptions, PrefixCache, benchmark_scheduler
    from baby_whale_v4.quantization import apply_weight_quantization

    tok = ByteTokenizer()
    cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=args.context_length)
    model = BabyWhaleV4Model(cfg)
    model.eval()
    if args.quant != "none":
        apply_weight_quantization(model, args.quant)
    prefix_cache = PrefixCache(capacity=args.prefix_cache_capacity) if args.prefix_cache else None
    engine = Engine(
        model=model,
        config=cfg,
        tokenizer_hash=tok.hash_signature(),
        prefix_cache=prefix_cache,
    )
    prompts = [tok.encode(args.prompt) for _ in range(args.requests)]
    bench = benchmark_scheduler(
        engine=engine,
        prompts=prompts,
        options=GenerationOptions(max_new_tokens=args.max_new_tokens, mode="greedy"),
        prefill_chunk=args.prefill_chunk,
    )
    print(
        json.dumps(
            {
                "requests": bench.requests,
                "prompt_tokens": bench.prompt_tokens,
                "generated_tokens": bench.generated_tokens,
                "total_ms": bench.total_ms,
                "decode_tokens_per_sec": bench.decode_tokens_per_sec,
                "prefill_steps": bench.prefill_steps,
                "decode_steps": bench.decode_steps,
                "completed": bench.completed,
                "prefix_cache_hits": bench.prefix_cache_hits,
                "prefix_cache_misses": bench.prefix_cache_misses,
                "quant": args.quant,
            },
            indent=2,
        )
    )


def _bench_compare(args: argparse.Namespace) -> None:
    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.data import ByteTokenizer
    from baby_whale_v4.inference import GenerationOptions, compare_inference_configs

    tok = ByteTokenizer()
    cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=args.context_length)
    model = BabyWhaleV4Model(cfg)
    model.eval()
    # Identical prompts maximize prefix-cache hits so the comparison is legible.
    prompts = [tok.encode(args.prompt) for _ in range(args.requests)]
    quant_modes = () if args.quant == "none" else (args.quant,)
    comparison = compare_inference_configs(
        model=model,
        config=cfg,
        tokenizer_hash=tok.hash_signature(),
        prompts=prompts,
        options=GenerationOptions(max_new_tokens=args.max_new_tokens, mode="greedy"),
        prefill_chunk=args.prefill_chunk,
        quant_modes=quant_modes,
    )
    print(json.dumps({"rows": comparison.as_rows(), "fastest": comparison.fastest()}, indent=2))


def register(sub: argparse._SubParsersAction) -> None:
    bench_fp4 = sub.add_parser(
        "bench-fp4-training",
        help="Benchmark custom-VJP FP4 training against dense MLX training.",
    )
    bench_fp4.add_argument("--batch-size", type=int, default=8)
    bench_fp4.add_argument("--input-dims", type=int, default=128)
    bench_fp4.add_argument("--output-dims", type=int, default=128)
    bench_fp4.add_argument("--warmup-steps", type=int, default=8)
    bench_fp4.add_argument("--timed-steps", type=int, default=20)
    bench_fp4.add_argument("--max-ratio", type=float, default=1.5)
    bench_fp4.add_argument("--weight-grad", choices=FP4_GRAD_CHOICES, default="mlx")
    bench_fp4.add_argument("--cache-policy", choices=FP4_CACHE_CHOICES, default="reuse")
    bench_fp4.set_defaults(func=_bench_fp4_training)

    bench_fp4_mem = sub.add_parser(
        "bench-fp4-memory",
        help="Benchmark FP4 training peak memory against dense fp32/bf16 training.",
    )
    bench_fp4_mem.add_argument("--batch-size", type=int, default=32)
    bench_fp4_mem.add_argument("--input-dims", type=int, default=1024)
    bench_fp4_mem.add_argument("--output-dims", type=int, default=1024)
    bench_fp4_mem.add_argument("--baseline", choices=FP4_MASTER_DTYPE_CHOICES, default="bf16")
    bench_fp4_mem.add_argument("--max-peak-ratio", type=float, default=1.0)
    bench_fp4_mem.add_argument("--weight-grad", choices=FP4_GRAD_CHOICES, default="mlx")
    bench_fp4_mem.add_argument("--cache-policy", choices=FP4_CACHE_CHOICES, default="reuse")
    bench_fp4_mem.add_argument(
        "--optimizer", choices=("none", "adamw", "adafactor", "muon"), default="none"
    )
    bench_fp4_mem.add_argument(
        "--fp4-master-dtype", choices=FP4_MASTER_DTYPE_CHOICES, default="bf16"
    )
    bench_fp4_mem.set_defaults(func=_bench_fp4_memory)

    bench_inf = sub.add_parser(
        "bench-inference",
        help="Benchmark chunked prefill, decode, prefix cache, and quantized inference.",
    )
    bench_inf.add_argument("--prompt", default="hello from baby whale")
    bench_inf.add_argument("--requests", type=int, default=2)
    bench_inf.add_argument("--max-new-tokens", type=int, default=8)
    bench_inf.add_argument("--context-length", type=int, default=128)
    bench_inf.add_argument("--prefill-chunk", type=int, default=8)
    bench_inf.add_argument("--prefix-cache", action="store_true")
    bench_inf.add_argument("--prefix-cache-capacity", type=int, default=16)
    bench_inf.add_argument("--quant", choices=QUANT_CHOICES, default="none")
    bench_inf.set_defaults(func=_bench_inference)

    bench_cmp = sub.add_parser(
        "bench-compare",
        help="Compare inference configs (no-cache / prefix-cache / paged / quant) on one prompt suite.",
    )
    bench_cmp.add_argument("--prompt", default="hello from baby whale")
    bench_cmp.add_argument("--requests", type=int, default=4)
    bench_cmp.add_argument("--max-new-tokens", type=int, default=8)
    bench_cmp.add_argument("--context-length", type=int, default=128)
    bench_cmp.add_argument("--prefill-chunk", type=int, default=8)
    bench_cmp.add_argument("--quant", choices=QUANT_CHOICES, default="none")
    bench_cmp.set_defaults(func=_bench_compare)

    bench_rollout = sub.add_parser(
        "bench-rollout",
        help="Benchmark InProcessRolloutEngine: prefix-cache hits, group throughput.",
    )
    bench_rollout.add_argument("--prompt", default="hello from baby whale")
    bench_rollout.add_argument("--group-size", type=int, default=8)
    bench_rollout.add_argument("--max-new-tokens", type=int, default=8)
    bench_rollout.add_argument("--context-length", type=int, default=128)
    bench_rollout.add_argument("--prefill-chunk", type=int, default=8)
    bench_rollout.add_argument("--prefix-cache-capacity", type=int, default=16)
    bench_rollout.set_defaults(func=_bench_rollout)
