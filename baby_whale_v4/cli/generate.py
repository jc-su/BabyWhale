import argparse
import json
import sys

from baby_whale_v4.cli._common import QUANT_CHOICES, RUNTIME_CHOICES


def _generate(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.config import config_for_inference
    from baby_whale_v4.data import ByteTokenizer, Message, load_tokenizer, render_chat_prompt
    from baby_whale_v4.device import ensure_runtime_matches
    from baby_whale_v4.inference.engine import Engine, GenerationOptions
    from baby_whale_v4.inference.prefix_cache import PrefixCache
    from baby_whale_v4.quantization import apply_weight_quantization
    from baby_whale_v4.training import load_checkpoint

    if (args.prompt is None) == (args.user is None):
        raise ValueError("generate requires exactly one of --prompt or --user")

    mx.random.seed(args.seed)
    tokenizer = (
        load_tokenizer(args.tokenizer_path) if args.tokenizer_path is not None else ByteTokenizer()
    )

    if args.from_checkpoint is not None:
        ckpt = load_checkpoint(args.from_checkpoint)
        cfg = config_for_inference(ckpt.config)
        if cfg.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"checkpoint vocab {cfg.vocab_size} != tokenizer vocab {tokenizer.vocab_size}"
            )
        ensure_runtime_matches(cfg.backend, args.runtime)
        model = BabyWhaleV4Model(cfg)
        model.update(ckpt.model_state)
    else:
        cfg = BabyWhaleV4Config.tiny(
            vocab_size=tokenizer.vocab_size,
            context_length=args.context_length,
        )
        ensure_runtime_matches(cfg.backend, args.runtime)
        model = BabyWhaleV4Model(cfg)
    model.eval()
    if args.quant != "none":
        apply_weight_quantization(model, args.quant)

    if args.user is not None:
        prompt_text = render_chat_prompt([Message("user", args.user)])
    else:
        prompt_text = args.prompt
    prompt_ids = tokenizer.encode(prompt_text)

    engine = Engine(
        model=model,
        config=cfg,
        tokenizer_hash=tokenizer.hash_signature(),
        prefix_cache=PrefixCache(capacity=8),
    )
    opts = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    gen_ids = engine.generate(prompt_ids, opts)
    completion = tokenizer.decode(gen_ids)

    print(completion)
    print(
        json.dumps(
            {
                "model": cfg.name,
                "config_hash": cfg.config_hash()[:12],
                "from_checkpoint": args.from_checkpoint,
                "tokenizer": tokenizer.kind,
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(gen_ids),
                "mode": args.mode,
            }
        ),
        file=sys.stderr,
    )


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "generate",
        help="One-shot in-process generation against a checkpoint (per-stage inspection).",
    )
    p.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Load a trained .bw4 checkpoint. When set, --context-length is ignored.",
    )
    p.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Trained tokenizer JSON. Defaults to ByteTokenizer.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Raw text prompt (no chat template). Mutually exclusive with --user.",
    )
    p.add_argument(
        "--user",
        type=str,
        default=None,
        help="User turn text; wrapped in <|user|>...<|eot|><|assistant|>. "
        "Mutually exclusive with --prompt.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument(
        "--quant",
        choices=QUANT_CHOICES,
        default="none",
    )
    p.add_argument(
        "--runtime",
        choices=RUNTIME_CHOICES,
        default="mlx-metal",
        help="Concrete MLX runtime. Use 'mlx-cuda' only with an MLX CUDA wheel on Linux/NVIDIA.",
    )
    p.add_argument("--context-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_generate)
