from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.data.tokenizer import Tokenizer

from baby_whale_v4.cli._common import (
    OPTIMIZER_CHOICES,
    PRECISION_CHOICES,
    QUANT_CHOICES,
    RUNTIME_CHOICES,
)


def _train_pretrain(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.data import SyntheticCopyDataset, load_tokenizer, pack_normalized_jsonl
    from baby_whale_v4.data.dataset import TensorPairDataset
    from baby_whale_v4.device import ensure_runtime_matches
    from baby_whale_v4.training import (
        ContextCurriculum,
        PretrainConfig,
        load_checkpoint,
        pretrain,
        pretrain_with_curriculum,
    )

    mx.random.seed(0)
    curriculum: ContextCurriculum | None = (
        ContextCurriculum.parse(args.context_curriculum)
        if args.context_curriculum is not None
        else None
    )
    if curriculum is not None and args.train_jsonl is None:
        raise ValueError("--context-curriculum requires --train-jsonl")
    if curriculum is not None and args.from_checkpoint is not None:
        raise ValueError(
            "--context-curriculum cannot be combined with --from-checkpoint; "
            "the curriculum builds the model fresh at max_context_length"
        )

    eval_ds: TensorPairDataset | None = None
    build_dataset: Callable[[int], TensorPairDataset] | None = None
    if args.train_jsonl is None:
        vocab_size = args.vocab_size
        context_length = args.seq_len
        ds: TensorPairDataset = SyntheticCopyDataset(
            n_samples=args.n_samples, seq_len=args.seq_len, vocab_size=args.vocab_size, seed=0
        )
        dataset_kind = "synthetic_copy"
    else:
        tok = load_tokenizer(args.tokenizer_path)
        vocab_size = tok.vocab_size
        if curriculum is not None:
            context_length = curriculum.max_context_length

            def build_dataset(block_size: int) -> TensorPairDataset:
                return pack_normalized_jsonl(args.train_jsonl, tokenizer=tok, block_size=block_size)

            ds = build_dataset(curriculum.phases[0].context_length)
            eval_ds = (
                None
                if args.eval_jsonl is None
                else pack_normalized_jsonl(
                    args.eval_jsonl,
                    tokenizer=tok,
                    block_size=curriculum.max_context_length,
                )
            )
        else:
            block_size = args.block_size
            context_length = block_size
            ds = pack_normalized_jsonl(args.train_jsonl, tokenizer=tok, block_size=block_size)
            eval_ds = (
                None
                if args.eval_jsonl is None
                else pack_normalized_jsonl(args.eval_jsonl, tokenizer=tok, block_size=block_size)
            )
        dataset_kind = "normalized_jsonl"

    starting_model: BabyWhaleV4Model | None = None
    if args.from_checkpoint is not None:
        ckpt = load_checkpoint(args.from_checkpoint)
        cfg = ckpt.config
        if cfg.vocab_size != vocab_size:
            raise ValueError(
                f"checkpoint vocab_size {cfg.vocab_size} != tokenizer/dataset vocab_size {vocab_size}; "
                "rebuild with a matching tokenizer or pretrain from scratch"
            )
        ensure_runtime_matches(cfg.backend, args.runtime)
        starting_model = BabyWhaleV4Model(cfg)
        starting_model.update(ckpt.model_state)
    else:
        cfg_overrides: dict[str, object] = {
            "backend": args.backend,
            "precision": args.precision,
            "quant_mode": args.quant,
            "activation_checkpoint": args.activation_checkpoint,
        }
        if args.n_layer is not None:
            cfg_overrides["n_layer"] = args.n_layer
        if args.n_embd is not None:
            cfg_overrides["n_embd"] = args.n_embd
        if args.n_head is not None:
            cfg_overrides["n_head"] = args.n_head
        if args.moe_intermediate_size is not None:
            cfg_overrides["moe_intermediate_size"] = args.moe_intermediate_size
        if args.sliding_window is not None:
            cfg_overrides["sliding_window"] = args.sliding_window
        cfg = BabyWhaleV4Config.from_dict(
            {
                **BabyWhaleV4Config.tiny(
                    vocab_size=vocab_size,
                    context_length=context_length,
                ).to_dict(),
                **cfg_overrides,
            }
        )
        ensure_runtime_matches(cfg.backend, args.runtime)

    out_dir = Path(args.out_dir)
    pretrain_cfg = PretrainConfig(
        lr=args.lr,
        optimizer=args.optimizer,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        log_every=max(1, args.max_steps // 4),
        seed=0,
        device="mlx",
    )
    if curriculum is not None:
        assert build_dataset is not None  # narrowed by control flow above
        pretrain_with_curriculum(
            config=cfg,
            pretrain_config=pretrain_cfg,
            curriculum=curriculum,
            build_dataset=build_dataset,
            eval_dataset=eval_ds,
            out_dir=out_dir,
            model=starting_model,
        )
    else:
        pretrain(
            config=cfg,
            pretrain_config=pretrain_cfg,
            train_dataset=ds,
            eval_dataset=eval_ds,
            out_dir=out_dir,
            model=starting_model,
        )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "max_steps": args.max_steps,
                "dataset_kind": dataset_kind,
                "vocab_size": cfg.vocab_size,
                "context_length": cfg.context_length,
                "curriculum": (
                    None
                    if curriculum is None
                    else [
                        {"context_length": p.context_length, "n_tokens": p.n_tokens}
                        for p in curriculum.phases
                    ]
                ),
            }
        )
    )


def _train_midtrain(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.data import DatasetMixtureSource, load_tokenizer, pack_mixture_jsonl
    from baby_whale_v4.device import ensure_runtime_matches
    from baby_whale_v4.training import MidtrainConfig, load_checkpoint, midtrain

    if not args.train_jsonl:
        raise ValueError("midtrain requires at least one --train-jsonl")
    mx.random.seed(0)
    tokenizer = load_tokenizer(args.tokenizer_path)
    sources = [
        DatasetMixtureSource(path=Path(path), repeat=args.source_repeat)
        for path in args.train_jsonl
    ]
    ds = pack_mixture_jsonl(sources, tokenizer=tokenizer, block_size=args.block_size)

    starting_model: BabyWhaleV4Model | None = None
    if args.from_checkpoint is not None:
        ckpt = load_checkpoint(args.from_checkpoint)
        cfg = ckpt.config
        if cfg.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"checkpoint vocab_size {cfg.vocab_size} != tokenizer vocab_size {tokenizer.vocab_size}"
            )
        ensure_runtime_matches(cfg.backend, args.runtime)
        starting_model = BabyWhaleV4Model(cfg)
        starting_model.update(ckpt.model_state)
    else:
        cfg_overrides: dict[str, object] = {
            "backend": args.backend,
            "precision": args.precision,
            "quant_mode": args.quant,
            "activation_checkpoint": args.activation_checkpoint,
        }
        if args.n_layer is not None:
            cfg_overrides["n_layer"] = args.n_layer
        if args.n_embd is not None:
            cfg_overrides["n_embd"] = args.n_embd
        if args.n_head is not None:
            cfg_overrides["n_head"] = args.n_head
        if args.moe_intermediate_size is not None:
            cfg_overrides["moe_intermediate_size"] = args.moe_intermediate_size
        if args.sliding_window is not None:
            cfg_overrides["sliding_window"] = args.sliding_window
        cfg = BabyWhaleV4Config.from_dict(
            {
                **BabyWhaleV4Config.tiny(
                    vocab_size=tokenizer.vocab_size,
                    context_length=args.block_size,
                ).to_dict(),
                **cfg_overrides,
            }
        )
        ensure_runtime_matches(cfg.backend, args.runtime)

    out_dir = Path(args.out_dir)
    midtrain(
        config=cfg,
        midtrain_config=MidtrainConfig(
            lr=args.lr,
            optimizer=args.optimizer,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            log_every=max(1, args.max_steps // 4),
            seed=0,
            device="mlx",
        ),
        train_dataset=ds,
        out_dir=out_dir,
        model=starting_model,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "max_steps": args.max_steps,
                "sources": [str(source.path) for source in sources],
                "vocab_size": cfg.vocab_size,
                "context_length": cfg.context_length,
            }
        )
    )


def _load_reward_fn(args: argparse.Namespace):
    import importlib.util
    import sys

    import mlx.core as mx

    if args.reward_token is not None:
        target = args.reward_token

        def fn(sample: mx.array) -> float:
            return float(mx.sum(mx.equal(sample, target)))

        return fn
    if args.reward_module is not None:
        spec = importlib.util.spec_from_file_location("user_reward", args.reward_module)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load reward module {args.reward_module}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["user_reward"] = module
        spec.loader.exec_module(module)
        fn = getattr(module, "reward_fn", None)
        if not callable(fn):
            raise ValueError("reward module must expose a callable `reward_fn(sample) -> float`")
        return fn
    raise ValueError("either --reward-token or --reward-module is required")


def _rl_setup_model(
    args: argparse.Namespace,
) -> tuple[BabyWhaleV4Config, BabyWhaleV4Model, Tokenizer, int]:
    """Build (cfg, model, tokenizer, starting_step) for any RL trainer.

    If --from-checkpoint is set, the config and weights come from the .bw4 and
    activation_checkpoint is disabled (KV-cache incompatibility). Otherwise we
    build a fresh tiny config sized to the tokenizer + --context-length.
    """
    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.config import config_for_inference
    from baby_whale_v4.data import ByteTokenizer, load_tokenizer
    from baby_whale_v4.training import load_checkpoint

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
        model = BabyWhaleV4Model(cfg)
        model.update(ckpt.model_state)
        return cfg, model, tokenizer, int(ckpt.step)
    cfg = BabyWhaleV4Config.tiny(
        vocab_size=tokenizer.vocab_size, context_length=args.context_length
    )
    model = BabyWhaleV4Model(cfg)
    return cfg, model, tokenizer, 0


def _train_grpo(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4.training import GRPOConfig, grpo
    from baby_whale_v4.training.checkpoint import save_checkpoint

    mx.random.seed(args.seed)
    cfg, model, tokenizer, starting_step = _rl_setup_model(args)
    prompts = [mx.array(tokenizer.encode(p), dtype=mx.int32) for p in args.prompt]
    reward_fn = _load_reward_fn(args)

    out_dir = Path(args.out_dir)
    grpo(
        model=model,
        prompts=prompts,
        reward_fn=reward_fn,
        grpo_config=GRPOConfig(
            lr=args.lr,
            beta_kl=args.beta_kl,
            group_size=args.group_size,
            response_len=args.response_len,
            max_steps=args.max_steps,
            log_every=max(1, args.max_steps // 4),
            temperature=args.temperature,
            seed=args.seed,
        ),
        out_dir=out_dir,
    )
    save_checkpoint(
        out_dir / "final.bw4",
        config=cfg,
        model=model,
        optimizer=None,
        scheduler=None,
        step=starting_step + args.max_steps,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "max_steps": args.max_steps,
                "final_checkpoint": str(out_dir / "final.bw4"),
            }
        )
    )


def _train_ppo(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4.training.checkpoint import save_checkpoint
    from baby_whale_v4.training.ppo import PPOConfig, ppo

    mx.random.seed(args.seed)
    cfg, model, tokenizer, starting_step = _rl_setup_model(args)
    prompts = [mx.array(tokenizer.encode(p), dtype=mx.int32) for p in args.prompt]
    reward_fn = _load_reward_fn(args)

    out_dir = Path(args.out_dir)
    ppo(
        model=model,
        prompts=prompts,
        reward_fn=reward_fn,
        ppo_config=PPOConfig(
            lr=args.lr,
            clip_eps=args.clip_eps,
            beta_kl=args.beta_kl,
            group_size=args.group_size,
            response_len=args.response_len,
            max_steps=args.max_steps,
            log_every=max(1, args.max_steps // 4),
            temperature=args.temperature,
            seed=args.seed,
        ),
        out_dir=out_dir,
    )
    save_checkpoint(
        out_dir / "final.bw4",
        config=cfg,
        model=model,
        optimizer=None,
        scheduler=None,
        step=starting_step + args.max_steps,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "max_steps": args.max_steps,
                "final_checkpoint": str(out_dir / "final.bw4"),
            }
        )
    )


def _train_rloo(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4.training.checkpoint import save_checkpoint
    from baby_whale_v4.training.rloo import RLOOConfig, rloo

    mx.random.seed(args.seed)
    cfg, model, tokenizer, starting_step = _rl_setup_model(args)
    prompts = [mx.array(tokenizer.encode(p), dtype=mx.int32) for p in args.prompt]
    reward_fn = _load_reward_fn(args)

    out_dir = Path(args.out_dir)
    rloo(
        model=model,
        prompts=prompts,
        reward_fn=reward_fn,
        rloo_config=RLOOConfig(
            lr=args.lr,
            beta_kl=args.beta_kl,
            group_size=args.group_size,
            response_len=args.response_len,
            max_steps=args.max_steps,
            log_every=max(1, args.max_steps // 4),
            temperature=args.temperature,
            seed=args.seed,
        ),
        out_dir=out_dir,
    )
    save_checkpoint(
        out_dir / "final.bw4",
        config=cfg,
        model=model,
        optimizer=None,
        scheduler=None,
        step=starting_step + args.max_steps,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "max_steps": args.max_steps,
                "final_checkpoint": str(out_dir / "final.bw4"),
            }
        )
    )


def _train_code_agent(args: argparse.Namespace) -> None:
    import dataclasses

    import mlx.core as mx

    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.config import config_for_inference
    from baby_whale_v4.data import load_tokenizer
    from baby_whale_v4.rl import CodeRewardConfig, load_problems_from_jsonl
    from baby_whale_v4.training import GRPOConfig, code_grpo, load_checkpoint

    mx.random.seed(args.seed)
    tokenizer = load_tokenizer(args.tokenizer_path)
    problems = load_problems_from_jsonl(args.problems_jsonl)
    if args.limit is not None:
        problems = problems[: args.limit]
    if not problems:
        raise ValueError("no problems loaded; check --problems-jsonl and --limit")

    if args.from_checkpoint is not None:
        ckpt = load_checkpoint(args.from_checkpoint)
        cfg = dataclasses.replace(config_for_inference(ckpt.config), precision=args.precision)
        if cfg.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"checkpoint vocab {cfg.vocab_size} != tokenizer vocab {tokenizer.vocab_size}"
            )
        model = BabyWhaleV4Model(cfg)
        model.update(ckpt.model_state)
    else:
        cfg_overrides: dict[str, object] = {
            "precision": args.precision,
        }
        if args.n_layer is not None:
            cfg_overrides["n_layer"] = args.n_layer
        if args.n_embd is not None:
            cfg_overrides["n_embd"] = args.n_embd
        if args.n_head is not None:
            cfg_overrides["n_head"] = args.n_head
        cfg = BabyWhaleV4Config.from_dict(
            {
                **BabyWhaleV4Config.tiny(
                    vocab_size=tokenizer.vocab_size,
                    context_length=args.context_length,
                ).to_dict(),
                **cfg_overrides,
            }
        )
        model = BabyWhaleV4Model(cfg)

    out_dir = Path(args.out_dir)
    code_grpo(
        model=model,
        problems=problems,
        tokenizer=tokenizer,
        grpo_config=GRPOConfig(
            lr=args.lr,
            beta_kl=args.beta_kl,
            group_size=args.group_size,
            response_len=args.response_len,
            max_steps=args.max_steps,
            log_every=max(1, args.max_steps // 4),
            temperature=args.temperature,
            seed=args.seed,
        ),
        out_dir=out_dir,
        reward_config=CodeRewardConfig(
            timeout_sec=args.timeout_sec,
            partial_credit=args.partial_credit,
        ),
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "max_steps": args.max_steps,
                "n_problems": len(problems),
                "tokenizer": str(args.tokenizer_path),
            }
        )
    )


def _train_distill(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.config import config_for_inference
    from baby_whale_v4.data import ByteTokenizer, load_tokenizer
    from baby_whale_v4.training import DistillConfig, distill, load_checkpoint, make_reference
    from baby_whale_v4.training.checkpoint import save_checkpoint

    mx.random.seed(args.seed)

    tokenizer = (
        load_tokenizer(args.tokenizer_path) if args.tokenizer_path is not None else ByteTokenizer()
    )

    # Student: from checkpoint or fresh tiny.
    if args.from_checkpoint is not None:
        s_ckpt = load_checkpoint(args.from_checkpoint)
        cfg = config_for_inference(s_ckpt.config)
        if cfg.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"student vocab {cfg.vocab_size} != tokenizer vocab {tokenizer.vocab_size}"
            )
        student = BabyWhaleV4Model(cfg)
        student.update(s_ckpt.model_state)
        starting_step = int(s_ckpt.step)
    else:
        cfg = BabyWhaleV4Config.tiny(
            vocab_size=tokenizer.vocab_size, context_length=args.context_length
        )
        student = BabyWhaleV4Model(cfg)
        starting_step = 0

    # Teacher: from a separate checkpoint, or self-distillation if absent.
    if args.teacher_checkpoint is not None:
        t_ckpt = load_checkpoint(args.teacher_checkpoint)
        t_cfg = config_for_inference(t_ckpt.config)
        if t_cfg.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"teacher vocab {t_cfg.vocab_size} != student vocab {cfg.vocab_size} — "
                "tokenizers must match"
            )
        teacher = BabyWhaleV4Model(t_cfg)
        teacher.update(t_ckpt.model_state)
        teacher.eval()
    else:
        teacher = make_reference(student)

    # Prompts: --prompts-jsonl beats inline --prompt (educational distill
    # typically runs on the same prompt distribution as RL/SFT).
    if args.problems_jsonl is not None:
        from baby_whale_v4.rl import load_problems_from_jsonl

        problems = load_problems_from_jsonl(args.problems_jsonl)
        if args.limit is not None:
            problems = problems[: args.limit]
        if not problems:
            raise ValueError("no problems loaded from --problems-jsonl")
        prompts = [mx.array(tokenizer.encode(p.prompt), dtype=mx.int32) for p in problems]
    elif args.prompt:
        prompts = [mx.array(tokenizer.encode(p), dtype=mx.int32) for p in args.prompt]
    else:
        raise ValueError("distill requires --prompt or --problems-jsonl")

    reward_fn = None
    if args.reward_token is not None or args.reward_module is not None:
        reward_fn = _load_reward_fn(args)

    out_dir = Path(args.out_dir)
    distill(
        student=student,
        teacher=teacher,
        prompts=prompts,
        reward_fn=reward_fn,
        distill_config=DistillConfig(
            lr=args.lr,
            teacher_temperature=args.teacher_temperature,
            student_temperature=args.student_temperature,
            group_size=args.group_size,
            response_len=args.response_len,
            max_steps=args.max_steps,
            log_every=max(1, args.max_steps // 4),
            temperature=args.temperature,
            reward_threshold=args.reward_threshold,
            seed=args.seed,
        ),
        out_dir=out_dir,
    )
    save_checkpoint(
        out_dir / "final.bw4",
        config=cfg,
        model=student,
        optimizer=None,
        scheduler=None,
        step=starting_step + args.max_steps,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "max_steps": args.max_steps,
                "n_prompts": len(prompts),
                "teacher": "self" if args.teacher_checkpoint is None else args.teacher_checkpoint,
                "final_checkpoint": str(out_dir / "final.bw4"),
            }
        )
    )


def _train_sft(args: argparse.Namespace) -> None:
    import json as _json
    import re as _re

    import mlx.core as mx

    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.cli._args import SFTArgs
    from baby_whale_v4.config import config_for_inference
    from baby_whale_v4.data import (
        ByteTokenizer,
        ChatExample,
        Message,
        SFTDataset,
        load_tokenizer,
    )
    from baby_whale_v4.rl import load_problems_from_jsonl
    from baby_whale_v4.training import SFTConfig, load_checkpoint, sft
    from baby_whale_v4.training.checkpoint import save_checkpoint

    a = SFTArgs.from_namespace(args)
    mx.random.seed(a.seed)

    # Tokenizer: trained tokenizer if given, otherwise ByteTokenizer
    tokenizer = (
        load_tokenizer(a.tokenizer_path) if a.tokenizer_path is not None else ByteTokenizer()
    )

    # Examples: pick exactly one source
    sources = [bool(a.user), bool(a.chat_jsonl), bool(a.problems_jsonl)]
    if sum(1 for s in sources if s) != 1:
        raise ValueError(
            "sft requires exactly one source: --user/--assistant pairs OR --chat-jsonl OR --problems-jsonl"
        )

    examples: list[ChatExample] = []
    if a.user:
        if len(a.user) != len(a.assistant):
            raise ValueError("--user and --assistant must be repeated the same number of times")
        for u, asst in zip(a.user, a.assistant, strict=True):
            examples.append(ChatExample(messages=[Message("user", u), Message("assistant", asst)]))
    elif a.chat_jsonl is not None:
        for line in Path(a.chat_jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _json.loads(line)
            msgs = row.get("messages") or []
            ms: list[Message] = []
            for m in msgs:
                role = m.get("role")
                content = m.get("content")
                if role not in ("system", "user", "assistant", "tool") or not isinstance(
                    content, str
                ):
                    raise ValueError("chat-jsonl rows must have messages[].role and string content")
                ms.append(Message(role=role, content=content))
            if not ms:
                continue
            examples.append(ChatExample(messages=ms))
    else:
        # Auto-build chat pairs from CodeProblem JSONL
        assert a.problems_jsonl is not None
        for problem in load_problems_from_jsonl(a.problems_jsonl):
            if not problem.canonical_solution:
                continue
            user_text = _re.sub(r"\s*```python\s*$", "", problem.prompt).strip()
            assistant_text = f"```python\n{problem.canonical_solution.strip()}\n```"
            examples.append(
                ChatExample(
                    messages=[
                        Message("user", user_text),
                        Message("assistant", assistant_text),
                    ]
                )
            )

    if not examples:
        raise ValueError("no examples loaded; check inputs")
    ds = SFTDataset(examples=examples, tokenizer=tokenizer, block_size=a.block_size)

    # Model: from checkpoint (inherit config) or fresh tiny
    if a.from_checkpoint is not None:
        ckpt = load_checkpoint(a.from_checkpoint)
        cfg = config_for_inference(ckpt.config)
        if cfg.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"checkpoint vocab_size {cfg.vocab_size} != tokenizer vocab_size {tokenizer.vocab_size}"
            )
        model = BabyWhaleV4Model(cfg)
        model.update(ckpt.model_state)
        starting_step = ckpt.step
    else:
        cfg = BabyWhaleV4Config.tiny(vocab_size=tokenizer.vocab_size, context_length=a.block_size)
        model = BabyWhaleV4Model(cfg)
        starting_step = 0

    out_dir = Path(a.out_dir)
    sft(
        config=cfg,
        sft_config=SFTConfig(
            lr=a.lr,
            batch_size=a.batch_size,
            max_steps=a.max_steps,
            log_every=max(1, a.max_steps // 4),
            seed=a.seed,
        ),
        train_dataset=ds,
        out_dir=out_dir,
        model=model,
    )
    # Save a final.bw4 the same way pretrain/midtrain do
    final = out_dir / "final.bw4"
    save_checkpoint(
        final,
        config=cfg,
        model=model,
        optimizer=None,
        scheduler=None,
        step=starting_step + a.max_steps,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "max_steps": a.max_steps,
                "examples": len(examples),
                "final_checkpoint": str(final),
            }
        )
    )


def _train_dpo(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
    from baby_whale_v4.config import config_for_inference
    from baby_whale_v4.data import ByteTokenizer, load_tokenizer
    from baby_whale_v4.training import DPOConfig, DPOExample, dpo, load_checkpoint
    from baby_whale_v4.training.checkpoint import save_checkpoint
    from baby_whale_v4.training.dpo import dpo_examples_from_jsonl

    mx.random.seed(args.seed)

    tokenizer = (
        load_tokenizer(args.tokenizer_path) if args.tokenizer_path is not None else ByteTokenizer()
    )

    # Examples: exactly one of --input-jsonl or inline --prompt/--chosen/--rejected.
    if args.input_jsonl is not None:
        if args.prompt or args.chosen or args.rejected:
            raise ValueError(
                "dpo: --input-jsonl is mutually exclusive with --prompt/--chosen/--rejected"
            )
        examples = dpo_examples_from_jsonl(
            args.input_jsonl,
            tokenizer,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
        )
    else:
        if not args.prompt:
            raise ValueError(
                "dpo requires either --input-jsonl or at least one --prompt/--chosen/--rejected triple"
            )
        if not (len(args.prompt) == len(args.chosen) == len(args.rejected)):
            raise ValueError("--prompt, --chosen, --rejected must repeat the same number of times")
        examples = [
            DPOExample(
                prompt=mx.array(tokenizer.encode(p), dtype=mx.int32),
                chosen=mx.array(tokenizer.encode(c), dtype=mx.int32),
                rejected=mx.array(tokenizer.encode(r), dtype=mx.int32),
            )
            for p, c, r in zip(args.prompt, args.chosen, args.rejected, strict=True)
        ]

    if args.from_checkpoint is not None:
        ckpt = load_checkpoint(args.from_checkpoint)
        cfg = config_for_inference(ckpt.config)
        if cfg.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"checkpoint vocab {cfg.vocab_size} != tokenizer vocab {tokenizer.vocab_size}"
            )
        model = BabyWhaleV4Model(cfg)
        model.update(ckpt.model_state)
        starting_step = ckpt.step
    else:
        cfg = BabyWhaleV4Config.tiny(
            vocab_size=tokenizer.vocab_size, context_length=args.context_length
        )
        model = BabyWhaleV4Model(cfg)
        starting_step = 0

    out_dir = Path(args.out_dir)
    dpo(
        model=model,
        examples=examples,
        dpo_config=DPOConfig(
            lr=args.lr,
            beta=args.beta,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            log_every=max(1, args.max_steps // 4),
            seed=args.seed,
        ),
        out_dir=out_dir,
    )
    save_checkpoint(
        out_dir / "final.bw4",
        config=cfg,
        model=model,
        optimizer=None,
        scheduler=None,
        step=starting_step + args.max_steps,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "max_steps": args.max_steps,
                "examples": len(examples),
                "final_checkpoint": str(out_dir / "final.bw4"),
            }
        )
    )


def register(sub: argparse._SubParsersAction) -> None:
    pre = sub.add_parser("pretrain", help="Pretrain on synthetic or normalized JSONL data.")
    pre.add_argument("--vocab-size", type=int, default=32)
    pre.add_argument("--seq-len", type=int, default=16)
    pre.add_argument("--block-size", type=int, default=16)
    pre.add_argument("--n-samples", type=int, default=8)
    pre.add_argument("--train-jsonl", default=None)
    pre.add_argument("--eval-jsonl", default=None)
    pre.add_argument("--tokenizer-path", default=None)
    pre.add_argument("--max-steps", type=int, default=40)
    pre.add_argument("--batch-size", type=int, default=4)
    pre.add_argument("--grad-accum", type=int, default=1)
    pre.add_argument("--lr", type=float, default=3e-3)
    pre.add_argument("--precision", choices=PRECISION_CHOICES, default="fp32")
    pre.add_argument("--optimizer", choices=OPTIMIZER_CHOICES, default="adamw")
    pre.add_argument("--quant", choices=QUANT_CHOICES, default="none")
    pre.add_argument("--activation-checkpoint", action="store_true")
    pre.add_argument("--device", choices=["mlx"], default="mlx")
    pre.add_argument(
        "--backend",
        choices=["mlx"],
        default="mlx",
        help="Framework backend. Only 'mlx' is supported.",
    )
    pre.add_argument(
        "--runtime",
        choices=RUNTIME_CHOICES,
        default="mlx-metal",
        help="Concrete MLX runtime. Use 'mlx-cuda' only with an MLX CUDA wheel on Linux/NVIDIA.",
    )
    pre.add_argument("--n-layer", type=int, default=None, help="Override default tiny n_layer.")
    pre.add_argument("--n-embd", type=int, default=None, help="Override default tiny n_embd.")
    pre.add_argument("--n-head", type=int, default=None, help="Override default tiny n_head.")
    pre.add_argument("--moe-intermediate-size", type=int, default=None)
    pre.add_argument("--sliding-window", type=int, default=None)
    pre.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Load weights from this .bw4 checkpoint and start a fresh optimizer; size flags are ignored.",
    )
    pre.add_argument(
        "--context-curriculum",
        type=str,
        default=None,
        help=(
            "V4-style native long-context curriculum. Comma-separated "
            "'len:tokens' phases (e.g. '384:50M,768:50M,1536:50M'); accepts K/M/B suffixes. "
            "The model is built at the max length; each phase re-packs the JSONL at "
            "its own block size and trains until n_tokens are consumed. Requires --train-jsonl."
        ),
    )
    pre.add_argument("--out-dir", type=str, required=True)
    pre.set_defaults(func=_train_pretrain)

    mid = sub.add_parser("midtrain", help="Continue training on normalized JSONL mixtures.")
    mid.add_argument("--train-jsonl", action="append", required=True)
    mid.add_argument("--tokenizer-path", default=None)
    mid.add_argument("--block-size", type=int, default=128)
    mid.add_argument("--source-repeat", type=int, default=1)
    mid.add_argument("--max-steps", type=int, default=40)
    mid.add_argument("--batch-size", type=int, default=4)
    mid.add_argument("--grad-accum", type=int, default=1)
    mid.add_argument("--lr", type=float, default=3e-4)
    mid.add_argument("--precision", choices=PRECISION_CHOICES, default="fp32")
    mid.add_argument("--optimizer", choices=OPTIMIZER_CHOICES, default="adamw")
    mid.add_argument("--quant", choices=QUANT_CHOICES, default="none")
    mid.add_argument("--activation-checkpoint", action="store_true")
    mid.add_argument(
        "--backend",
        choices=["mlx"],
        default="mlx",
        help="Framework backend. Only 'mlx' is supported.",
    )
    mid.add_argument(
        "--runtime",
        choices=RUNTIME_CHOICES,
        default="mlx-metal",
        help="Concrete MLX runtime. Use 'mlx-cuda' only with an MLX CUDA wheel on Linux/NVIDIA.",
    )
    mid.add_argument("--n-layer", type=int, default=None)
    mid.add_argument("--n-embd", type=int, default=None)
    mid.add_argument("--n-head", type=int, default=None)
    mid.add_argument("--moe-intermediate-size", type=int, default=None)
    mid.add_argument("--sliding-window", type=int, default=None)
    mid.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Load weights from this .bw4 checkpoint and start a fresh optimizer.",
    )
    mid.add_argument("--out-dir", type=str, required=True)
    mid.set_defaults(func=_train_midtrain)

    g = sub.add_parser("grpo", help="GRPO training over byte-tokenized prompts.")
    g.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Prompt text. Repeat for multi-prompt training.",
    )
    g.add_argument("--context-length", type=int, default=64)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--beta-kl", type=float, default=0.0)
    g.add_argument("--group-size", type=int, default=4)
    g.add_argument("--response-len", type=int, default=8)
    g.add_argument("--max-steps", type=int, default=20)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument(
        "--reward-token",
        type=int,
        default=None,
        help="Reward = count of this token id in the response.",
    )
    g.add_argument(
        "--reward-module",
        type=str,
        default=None,
        help="Path to a Python file exposing a `reward_fn(sample) -> float` callable.",
    )
    g.add_argument("--out-dir", type=str, required=True)
    g.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Continue from a .bw4 checkpoint (typically the SFT/DPO model). "
        "When set, --context-length is taken from the checkpoint.",
    )
    g.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Trained tokenizer JSON. Defaults to ByteTokenizer.",
    )
    g.set_defaults(func=_train_grpo)

    p_ppo = sub.add_parser("ppo", help="Clipped-IS PPO training over byte-tokenized prompts.")
    p_ppo.add_argument("--prompt", action="append", required=True)
    p_ppo.add_argument("--context-length", type=int, default=64)
    p_ppo.add_argument("--lr", type=float, default=5e-4)
    p_ppo.add_argument("--clip-eps", type=float, default=0.2)
    p_ppo.add_argument("--beta-kl", type=float, default=0.0)
    p_ppo.add_argument("--group-size", type=int, default=4)
    p_ppo.add_argument("--response-len", type=int, default=8)
    p_ppo.add_argument("--max-steps", type=int, default=20)
    p_ppo.add_argument("--temperature", type=float, default=1.0)
    p_ppo.add_argument("--seed", type=int, default=0)
    p_ppo.add_argument("--reward-token", type=int, default=None)
    p_ppo.add_argument("--reward-module", type=str, default=None)
    p_ppo.add_argument("--out-dir", type=str, required=True)
    p_ppo.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Continue from a .bw4 checkpoint. When set, --context-length is "
        "taken from the checkpoint.",
    )
    p_ppo.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Trained tokenizer JSON. Defaults to ByteTokenizer.",
    )
    p_ppo.set_defaults(func=_train_ppo)

    p_rloo = sub.add_parser(
        "rloo", help="REINFORCE Leave-One-Out training over byte-tokenized prompts."
    )
    p_rloo.add_argument("--prompt", action="append", required=True)
    p_rloo.add_argument("--context-length", type=int, default=64)
    p_rloo.add_argument("--lr", type=float, default=5e-4)
    p_rloo.add_argument("--beta-kl", type=float, default=0.0)
    p_rloo.add_argument("--group-size", type=int, default=4)
    p_rloo.add_argument("--response-len", type=int, default=8)
    p_rloo.add_argument("--max-steps", type=int, default=20)
    p_rloo.add_argument("--temperature", type=float, default=1.0)
    p_rloo.add_argument("--seed", type=int, default=0)
    p_rloo.add_argument("--reward-token", type=int, default=None)
    p_rloo.add_argument("--reward-module", type=str, default=None)
    p_rloo.add_argument("--out-dir", type=str, required=True)
    p_rloo.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Continue from a .bw4 checkpoint. When set, --context-length is "
        "taken from the checkpoint.",
    )
    p_rloo.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Trained tokenizer JSON. Defaults to ByteTokenizer.",
    )
    p_rloo.set_defaults(func=_train_rloo)

    s = sub.add_parser("sft", help="Supervised fine-tuning on (user, assistant) chat pairs.")
    s.add_argument(
        "--user",
        action="append",
        default=[],
        help="User turn text. Repeat with matching --assistant for inline examples.",
    )
    s.add_argument("--assistant", action="append", default=[])
    s.add_argument(
        "--chat-jsonl",
        type=str,
        default=None,
        help='JSONL of {"messages":[{"role":..,"content":..}, ...]} chat-format examples.',
    )
    s.add_argument(
        "--problems-jsonl",
        type=str,
        default=None,
        help="CodeProblem JSONL (produced by prepare-code-tasks). Auto-builds (user, assistant) pairs.",
    )
    s.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Trained tokenizer JSON. Defaults to ByteTokenizer.",
    )
    s.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Continue from a .bw4 checkpoint (parity with pretrain/midtrain).",
    )
    s.add_argument("--block-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=5e-3)
    s.add_argument("--batch-size", type=int, default=2)
    s.add_argument("--max-steps", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-dir", type=str, required=True)
    s.set_defaults(func=_train_sft)

    p_code = sub.add_parser(
        "train-code-agent",
        help="GRPO with verifiable code-execution rewards on MBPP/HumanEval-style problems.",
    )
    p_code.add_argument(
        "--problems-jsonl",
        required=True,
        help="Path to a CodeProblem JSONL (produced by `prepare-code-tasks`).",
    )
    p_code.add_argument("--tokenizer-path", required=True)
    p_code.add_argument("--context-length", type=int, default=512)
    p_code.add_argument("--lr", type=float, default=5e-5)
    p_code.add_argument("--beta-kl", type=float, default=0.0)
    p_code.add_argument("--group-size", type=int, default=4)
    p_code.add_argument("--response-len", type=int, default=128)
    p_code.add_argument("--max-steps", type=int, default=20)
    p_code.add_argument("--temperature", type=float, default=0.8)
    p_code.add_argument("--timeout-sec", type=float, default=5.0)
    p_code.add_argument("--partial-credit", action="store_true")
    p_code.add_argument("--limit", type=int, default=None, help="Use only the first N problems.")
    p_code.add_argument("--precision", choices=PRECISION_CHOICES, default="bf16")
    p_code.add_argument("--n-layer", type=int, default=None)
    p_code.add_argument("--n-embd", type=int, default=None)
    p_code.add_argument("--n-head", type=int, default=None)
    p_code.add_argument("--seed", type=int, default=0)
    p_code.add_argument("--out-dir", type=str, required=True)
    p_code.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Continue from a .bw4 checkpoint. When set, model arch flags "
        "(--context-length/--n-layer/--n-embd/--n-head) are taken from the "
        "checkpoint and ignored.",
    )
    p_code.set_defaults(func=_train_code_agent)

    p_dist = sub.add_parser(
        "distill",
        help="On-policy distillation: student generates, teacher labels, KL loss.",
    )
    p_dist.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Inline prompt. Repeat for multi-prompt distill. Mutually exclusive with --problems-jsonl.",
    )
    p_dist.add_argument(
        "--problems-jsonl",
        type=str,
        default=None,
        help="CodeProblem JSONL — use problem prompts as the distill prompt distribution.",
    )
    p_dist.add_argument("--limit", type=int, default=None, help="Use only first N problems.")
    p_dist.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Student starting weights. .bw4 from sft/dpo/grpo. Default: fresh tiny.",
    )
    p_dist.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="Teacher .bw4 (frozen). Default: self-distillation against student snapshot.",
    )
    p_dist.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Trained tokenizer JSON. Defaults to ByteTokenizer.",
    )
    p_dist.add_argument("--context-length", type=int, default=64)
    p_dist.add_argument("--lr", type=float, default=5e-4)
    p_dist.add_argument("--teacher-temperature", type=float, default=1.0)
    p_dist.add_argument("--student-temperature", type=float, default=1.0)
    p_dist.add_argument("--group-size", type=int, default=4)
    p_dist.add_argument("--response-len", type=int, default=8)
    p_dist.add_argument("--max-steps", type=int, default=20)
    p_dist.add_argument("--temperature", type=float, default=1.0)
    p_dist.add_argument("--reward-threshold", type=float, default=0.0)
    p_dist.add_argument("--seed", type=int, default=0)
    p_dist.add_argument("--reward-token", type=int, default=None)
    p_dist.add_argument("--reward-module", type=str, default=None)
    p_dist.add_argument("--out-dir", type=str, required=True)
    p_dist.set_defaults(func=_train_distill)

    d = sub.add_parser("dpo", help="DPO over (prompt, chosen, rejected) triples.")
    d.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Inline prompt. Repeat with matching --chosen/--rejected. Mutually exclusive with --input-jsonl.",
    )
    d.add_argument("--chosen", action="append", default=[])
    d.add_argument("--rejected", action="append", default=[])
    d.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help='Preference JSONL with rows like {"kind":"preference","prompt":..,"chosen":..,"rejected":..}.',
    )
    d.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Continue from a .bw4 checkpoint (typically the SFT model).",
    )
    d.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Trained tokenizer JSON. Defaults to ByteTokenizer.",
    )
    d.add_argument("--context-length", type=int, default=32)
    d.add_argument("--max-prompt-tokens", type=int, default=128)
    d.add_argument("--max-response-tokens", type=int, default=128)
    d.add_argument("--lr", type=float, default=1e-3)
    d.add_argument("--beta", type=float, default=0.1)
    d.add_argument("--batch-size", type=int, default=2)
    d.add_argument("--max-steps", type=int, default=20)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out-dir", type=str, required=True)
    d.set_defaults(func=_train_dpo)
