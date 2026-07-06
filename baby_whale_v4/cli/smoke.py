import argparse
import json


def _smoke(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model

    config = BabyWhaleV4Config.tiny(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
    )
    model = BabyWhaleV4Model(config)
    model.eval()
    input_ids = mx.random.randint(0, config.vocab_size, (args.batch_size, args.seq_len))
    out = model(input_ids)
    print(
        json.dumps(
            {
                "model": config.name,
                "shape": list(out.logits.shape),
                "params": model.num_parameters(),
                "backend": config.backend,
                "precision": config.precision,
                "config_hash": config.config_hash(),
            },
            indent=2,
        )
    )


def _smoke_hybrid(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model

    cfg = BabyWhaleV4Config.hybrid_tiny(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        hc_mult=args.hc_mult,
        mtp_heads=args.mtp_heads,
    )
    model = BabyWhaleV4Model(cfg)
    model.eval()
    input_ids = mx.random.randint(0, cfg.vocab_size, (1, args.seq_len))
    out = model(input_ids)
    print(
        json.dumps(
            {
                "model": cfg.name,
                "schedule": list(cfg.effective_layer_schedule),
                "hc_mult": cfg.hc_mult,
                "mtp_heads": cfg.mtp_heads,
                "shape": list(out.logits.shape),
                "n_mtp_logits": len(out.mtp_logits),
                "params": model.num_parameters(),
            },
            indent=2,
        )
    )


def register(sub: argparse._SubParsersAction) -> None:
    smoke = sub.add_parser("smoke", help="Tiny forward pass.")
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--seq-len", type=int, default=8)
    smoke.add_argument("--vocab-size", type=int, default=128)
    smoke.add_argument("--context-length", type=int, default=32)
    smoke.set_defaults(func=_smoke)

    hybrid = sub.add_parser("smoke-hybrid", help="Hybrid (sliding/HCA/CSA) forward pass.")
    hybrid.add_argument("--vocab-size", type=int, default=128)
    hybrid.add_argument("--context-length", type=int, default=64)
    hybrid.add_argument("--seq-len", type=int, default=24)
    hybrid.add_argument("--hc-mult", type=int, default=1, dest="hc_mult")
    hybrid.add_argument("--mtp-heads", type=int, default=0, dest="mtp_heads")
    hybrid.set_defaults(func=_smoke_hybrid)
