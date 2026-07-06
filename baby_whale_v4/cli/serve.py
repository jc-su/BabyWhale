import argparse
import sys

from baby_whale_v4.cli._common import QUANT_CHOICES, RUNTIME_CHOICES


def _serve(args: argparse.Namespace) -> None:
    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.config import config_for_inference
    from baby_whale_v4.data import ByteTokenizer, load_tokenizer
    from baby_whale_v4.device import ensure_runtime_matches
    from baby_whale_v4.inference.server import serve
    from baby_whale_v4.quantization import apply_weight_quantization
    from baby_whale_v4.training import load_checkpoint

    tokenizer = (
        load_tokenizer(args.tokenizer_path) if args.tokenizer_path is not None else ByteTokenizer()
    )
    if args.from_checkpoint is not None:
        ckpt = load_checkpoint(args.from_checkpoint)
        cfg = config_for_inference(ckpt.config)
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
    print(
        f"serving on http://{args.host}:{args.port}  "
        f"(model={cfg.name} config_hash={cfg.config_hash()[:12]} "
        f"tokenizer={tokenizer.kind} vocab={tokenizer.vocab_size})",
        file=sys.stderr,
    )
    serve(model=model, config=cfg, tokenizer=tokenizer, host=args.host, port=args.port)


def register(sub: argparse._SubParsersAction) -> None:
    serve_p = sub.add_parser("serve", help="Run a local HTTP inference server.")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--context-length", type=int, default=64)
    serve_p.add_argument("--quant", choices=QUANT_CHOICES, default="none")
    serve_p.add_argument(
        "--runtime",
        choices=RUNTIME_CHOICES,
        default="mlx-metal",
        help="Concrete MLX runtime. Use 'mlx-cuda' only with an MLX CUDA wheel on Linux/NVIDIA.",
    )
    serve_p.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Load a trained .bw4 checkpoint. When set, --context-length is ignored.",
    )
    serve_p.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to a trained tokenizer JSON. Defaults to ByteTokenizer.",
    )
    serve_p.set_defaults(func=_serve)
