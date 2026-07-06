"""Per-stage evaluation subcommands.

Each subcommand answers a single "is this stage GOOD?" question with a number:
  * eval-tokenizer  -> bytes-per-token, fertility
  * eval-bpb        -> held-out perplexity / bits-per-byte
  * eval-code       -> HumanEval/MBPP-style pass@1
  * eval-ifeval     -> minimal verifiable instruction-following accuracy
  * eval-dpo        -> DPO reward accuracy / margin on a preference JSONL
  * eval-rl-health  -> threshold check on a GRPO/PPO/RLOO metrics JSONL
  * eval-parity     -> Engine.generate vs. naive greedy decode parity
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from typing import Any


def _load_tokenizer(path: str | None) -> Any:
    from baby_whale_v4.data import ByteTokenizer, load_tokenizer

    return load_tokenizer(path) if path is not None else ByteTokenizer()


def _load_model_from_ckpt(ckpt_path: str, tokenizer: Any) -> tuple[Any, Any]:
    from baby_whale_v4 import BabyWhaleV4Model
    from baby_whale_v4.config import config_for_inference
    from baby_whale_v4.training import load_checkpoint

    ckpt = load_checkpoint(ckpt_path)
    cfg = config_for_inference(ckpt.config)
    if cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"checkpoint vocab {cfg.vocab_size} != tokenizer vocab {tokenizer.vocab_size}"
        )
    model = BabyWhaleV4Model(cfg)
    model.update(ckpt.model_state)
    return model, cfg


def _iter_jsonl(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            yield line_no, json.loads(stripped)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


# ---------- 1) eval-tokenizer ---------------------------------------------


def _eval_tokenizer(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer(args.tokenizer_path)
    total_bytes = 0
    total_tokens = 0
    total_words = 0
    n_lines = 0
    for _, rec in _iter_jsonl(args.input_jsonl):
        text = str(rec.get(args.text_field, ""))
        if not text:
            continue
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokenizer.encode(text))
        total_words += len(text.split())
        n_lines += 1
        if args.limit is not None and n_lines >= args.limit:
            break
    if total_tokens == 0:
        raise ValueError("scored 0 tokens; check --input-jsonl and --text-field")
    _emit(
        {
            "tokenizer_kind": tokenizer.kind,
            "vocab_size": tokenizer.vocab_size,
            "n_lines": n_lines,
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
            "total_words": total_words,
            "bytes_per_token": round(total_bytes / total_tokens, 4),
            "fertility_tokens_per_word": (
                round(total_tokens / total_words, 4) if total_words else None
            ),
        }
    )


# ---------- 2) eval-bpb ---------------------------------------------------


def _eval_bpb(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4.training.dpo import _log_softmax

    tokenizer = _load_tokenizer(args.tokenizer_path)
    model, cfg = _load_model_from_ckpt(args.from_checkpoint, tokenizer)
    model.eval()

    block_size = args.block_size
    if block_size + 1 > cfg.context_length:
        raise ValueError(
            f"block-size {block_size} + 1 exceeds checkpoint context_length {cfg.context_length}"
        )

    total_loss_nats = 0.0
    total_tokens_in_loss = 0
    total_bytes_in_loss = 0
    n_blocks = 0
    for _, rec in _iter_jsonl(args.input_jsonl):
        text = str(rec.get(args.text_field, ""))
        if not text:
            continue
        text_bytes = text.encode("utf-8")
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue
        # Walk fixed-size blocks. The byte share for each block is approximated
        # by tokens_in_block / total_tokens_in_text * total_bytes_in_text so the
        # per-byte denominator is consistent.
        for start in range(0, len(ids) - 1, block_size):
            chunk = ids[start : start + block_size + 1]
            if len(chunk) < 2:
                continue
            x = mx.array([chunk[:-1]], dtype=mx.int32)
            y = mx.array([chunk[1:]], dtype=mx.int32)
            logits = model(x).logits
            log_probs = _log_softmax(logits)
            tgt = mx.take_along_axis(log_probs, y[:, :, None], axis=-1).squeeze(-1)
            block_tokens = int(tgt.shape[1])
            total_loss_nats += float(-mx.sum(tgt))
            total_tokens_in_loss += block_tokens
            total_bytes_in_loss += round(block_tokens / len(ids) * len(text_bytes))
            n_blocks += 1
            if args.limit_blocks is not None and n_blocks >= args.limit_blocks:
                break
        if args.limit_blocks is not None and n_blocks >= args.limit_blocks:
            break

    if total_tokens_in_loss == 0:
        raise ValueError("scored 0 tokens; check --input-jsonl, --text-field, --block-size")
    mean_loss_nats = total_loss_nats / total_tokens_in_loss
    tokens_per_byte = total_tokens_in_loss / total_bytes_in_loss if total_bytes_in_loss else 0.0
    bpb = (mean_loss_nats / math.log(2)) * tokens_per_byte
    _emit(
        {
            "from_checkpoint": args.from_checkpoint,
            "tokenizer_kind": tokenizer.kind,
            "n_blocks": n_blocks,
            "tokens_scored": total_tokens_in_loss,
            "bytes_scored": total_bytes_in_loss,
            "mean_loss_nats": round(mean_loss_nats, 4),
            "perplexity": round(math.exp(mean_loss_nats), 4),
            "bits_per_byte": round(bpb, 4),
            "tokens_per_byte": round(tokens_per_byte, 4),
        }
    )


# ---------- 3) eval-code (HumanEval/MBPP pass@1) -------------------------


def _eval_code(args: argparse.Namespace) -> None:
    from baby_whale_v4.data import Message, render_chat_prompt
    from baby_whale_v4.inference.engine import Engine, GenerationOptions
    from baby_whale_v4.inference.prefix_cache import PrefixCache
    from baby_whale_v4.rl import load_problems_from_jsonl
    from baby_whale_v4.rl.code_exec import execute_python_with_tests
    from baby_whale_v4.rl.code_reward import extract_python_code

    tokenizer = _load_tokenizer(args.tokenizer_path)
    model, cfg = _load_model_from_ckpt(args.from_checkpoint, tokenizer)
    model.eval()

    problems = load_problems_from_jsonl(args.problems_jsonl)
    if args.limit is not None:
        problems = problems[: args.limit]
    if not problems:
        raise ValueError("no problems to evaluate")

    engine = Engine(
        model=model,
        config=cfg,
        tokenizer_hash=tokenizer.hash_signature(),
        prefix_cache=PrefixCache(capacity=16),
    )
    opts = GenerationOptions(max_new_tokens=args.max_new_tokens, mode="greedy")

    n_passed = 0
    n_total = 0
    details: list[dict[str, Any]] = []
    for problem in problems:
        prompt_text = (
            render_chat_prompt([Message("user", problem.prompt)])
            if args.chat_template
            else problem.prompt
        )
        prompt_ids = tokenizer.encode(prompt_text)
        allowed = cfg.context_length - args.max_new_tokens - 1
        if allowed <= 0:
            details.append({"problem_id": problem.problem_id, "skipped": "context too short"})
            continue
        if len(prompt_ids) > allowed:
            prompt_ids = prompt_ids[-allowed:]
        try:
            gen_ids = engine.generate(prompt_ids, opts, request_id=problem.problem_id)
        except ValueError as exc:
            details.append({"problem_id": problem.problem_id, "error": str(exc)})
            n_total += 1
            continue
        completion = tokenizer.decode(gen_ids)
        code = extract_python_code(completion)
        try:
            result = execute_python_with_tests(
                solution_code=code,
                tests=list(problem.tests),
                timeout_sec=args.timeout_sec,
            )
            passed = bool(result.passed)
            n_pass_tests = int(result.n_passed)
            n_tests = int(result.n_tests)
        except Exception as exc:  # sandbox / decode-time errors
            passed = False
            n_pass_tests = 0
            n_tests = len(problem.tests)
            details.append({"problem_id": problem.problem_id, "exec_error": str(exc)[:120]})
        n_total += 1
        if passed:
            n_passed += 1
        details.append(
            {
                "problem_id": problem.problem_id,
                "passed": passed,
                "n_passed_tests": n_pass_tests,
                "n_tests": n_tests,
            }
        )

    pass_at_1 = n_passed / n_total if n_total else 0.0
    payload = {
        "from_checkpoint": args.from_checkpoint,
        "problems_jsonl": args.problems_jsonl,
        "chat_template": args.chat_template,
        "n_passed": n_passed,
        "n_total": n_total,
        "pass_at_1": round(pass_at_1, 4),
    }
    if args.show_details:
        payload["details"] = details
    _emit(payload)


# ---------- 4) eval-ifeval (minimal verifiable instructions) -------------


def _ifeval_instructions() -> list[tuple[str, Any, str]]:
    """Built-in instruction set with deterministic verifiers.

    Each item is (prompt, verifier(response: str) -> bool, kind).
    Kept small so the educational impl runs in seconds and the success criteria
    are obvious from the verifier source.
    """

    def _starts_with_a(r: str) -> bool:
        s = r.strip()
        return bool(s) and s[0].lower() == "a"

    def _ends_with_period(r: str) -> bool:
        return r.strip().endswith(".")

    def _is_yes_or_no(r: str) -> bool:
        return r.strip().upper() in {"YES", "NO"}

    def _contains_apple(r: str) -> bool:
        return "apple" in r.lower()

    def _no_banana(r: str) -> bool:
        return "banana" not in r.lower()

    def _exactly_three_words(r: str) -> bool:
        return len(r.strip().split()) == 3

    def _at_most_ten_words(r: str) -> bool:
        return 0 < len(r.strip().split()) <= 10

    def _json_with_name(r: str) -> bool:
        try:
            data = json.loads(r.strip())
        except ValueError, TypeError:
            return False
        return isinstance(data, dict) and "name" in data

    return [
        ("Respond with only the word YES or only the word NO.", _is_yes_or_no, "exact_choice"),
        ("Include the word 'apple' in your response.", _contains_apple, "keyword_present"),
        (
            "Do not include the word 'banana' in your response.",
            _no_banana,
            "keyword_absent",
        ),
        ("Respond with exactly three words.", _exactly_three_words, "word_count_exact"),
        ("Respond with at most ten words.", _at_most_ten_words, "word_count_max"),
        ("Start your response with the letter A.", _starts_with_a, "startswith"),
        ("End your response with a period.", _ends_with_period, "endswith"),
        (
            "Respond with a valid JSON object that has a 'name' field.",
            _json_with_name,
            "json_valid",
        ),
    ]


def _eval_ifeval(args: argparse.Namespace) -> None:
    from baby_whale_v4.data import Message, render_chat_prompt
    from baby_whale_v4.inference.engine import Engine, GenerationOptions
    from baby_whale_v4.inference.prefix_cache import PrefixCache

    tokenizer = _load_tokenizer(args.tokenizer_path)
    model, cfg = _load_model_from_ckpt(args.from_checkpoint, tokenizer)
    model.eval()

    engine = Engine(
        model=model,
        config=cfg,
        tokenizer_hash=tokenizer.hash_signature(),
        prefix_cache=PrefixCache(capacity=8),
    )
    opts = GenerationOptions(max_new_tokens=args.max_new_tokens, mode="greedy")
    instructions = _ifeval_instructions()
    n_pass = 0
    details: list[dict[str, Any]] = []
    for prompt_text, verifier, kind in instructions:
        chat_prompt = render_chat_prompt([Message("user", prompt_text)])
        prompt_ids = tokenizer.encode(chat_prompt)
        allowed = cfg.context_length - args.max_new_tokens - 1
        if allowed <= 0:
            details.append({"kind": kind, "skipped": "context too short"})
            continue
        if len(prompt_ids) > allowed:
            prompt_ids = prompt_ids[-allowed:]
        gen_ids = engine.generate(prompt_ids, opts, request_id=kind)
        completion = tokenizer.decode(gen_ids)
        if "<|eot|>" in completion:
            completion = completion.split("<|eot|>", 1)[0]
        passed = bool(verifier(completion))
        if passed:
            n_pass += 1
        details.append(
            {
                "kind": kind,
                "prompt": prompt_text,
                "response": completion[:80],
                "passed": passed,
            }
        )
    n_total = len(instructions)
    payload: dict[str, Any] = {
        "from_checkpoint": args.from_checkpoint,
        "n_pass": n_pass,
        "n_total": n_total,
        "strict_accuracy": round(n_pass / n_total if n_total else 0.0, 4),
    }
    if args.show_details:
        payload["details"] = details
    _emit(payload)


# ---------- 5) eval-dpo --------------------------------------------------


def _eval_dpo(args: argparse.Namespace) -> None:
    from baby_whale_v4.training.dpo import (
        _logp_response,
        dpo_examples_from_jsonl,
        make_reference,
    )

    tokenizer = _load_tokenizer(args.tokenizer_path)
    model, _ = _load_model_from_ckpt(args.from_checkpoint, tokenizer)
    model.eval()
    if args.ref_checkpoint is not None:
        ref_model, _ = _load_model_from_ckpt(args.ref_checkpoint, tokenizer)
        ref_model.eval()
    else:
        ref_model = make_reference(model)

    examples = dpo_examples_from_jsonl(
        args.input_jsonl,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("no DPO examples to evaluate")

    n_correct = 0
    margins: list[float] = []
    for ex in examples:
        prompt = ex.prompt[None, :]
        pi_c = float(_logp_response(model, prompt, ex.chosen[None, :])[0])
        pi_r = float(_logp_response(model, prompt, ex.rejected[None, :])[0])
        ref_c = float(_logp_response(ref_model, prompt, ex.chosen[None, :])[0])
        ref_r = float(_logp_response(ref_model, prompt, ex.rejected[None, :])[0])
        reward_c = args.beta * (pi_c - ref_c)
        reward_r = args.beta * (pi_r - ref_r)
        if reward_c > reward_r:
            n_correct += 1
        margins.append(reward_c - reward_r)
    n_total = len(examples)
    _emit(
        {
            "from_checkpoint": args.from_checkpoint,
            "ref_checkpoint": args.ref_checkpoint,
            "beta": args.beta,
            "n_total": n_total,
            "reward_accuracy": round(n_correct / n_total, 4),
            "mean_margin": round(sum(margins) / n_total, 4),
        }
    )


# ---------- 6) eval-rl-health -------------------------------------------


def _eval_rl_health(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for _, rec in _iter_jsonl(args.metrics_jsonl):
        rows.append(rec)
    if not rows:
        raise ValueError(f"no rows in {args.metrics_jsonl}")

    warnings_list: list[str] = []
    for row in rows:
        for k, v in row.items():
            if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                warnings_list.append(f"non-finite {k}={v} at step={row.get('step', '?')}")

    def _series(key: str) -> list[float]:
        return [float(r[key]) for r in rows if key in r and r[key] is not None]

    rewards = _series("reward_mean")
    if len(rewards) >= 4:
        q = max(1, len(rewards) // 4)
        first_q = sum(rewards[:q]) / q
        last_q = sum(rewards[-q:]) / q
        if last_q <= first_q + args.min_reward_delta:
            warnings_list.append(f"reward stagnation: first_q={first_q:.4f} -> last_q={last_q:.4f}")

    stds = _series("reward_std")
    if stds and stds[-1] < args.min_reward_std:
        warnings_list.append(f"reward_std collapse: last={stds[-1]:.4g} < {args.min_reward_std}")

    kls = _series("kl_mean")
    if kls and max(kls) > args.max_kl:
        warnings_list.append(f"kl ceiling exceeded: max={max(kls):.4f} > {args.max_kl}")

    entropies = _series("entropy_mean")
    if entropies and min(entropies) < args.min_entropy:
        warnings_list.append(f"entropy collapsed: min={min(entropies):.4f} < {args.min_entropy}")

    _emit(
        {
            "metrics_jsonl": args.metrics_jsonl,
            "n_steps": len(rows),
            "final_reward_mean": rewards[-1] if rewards else None,
            "final_reward_std": stds[-1] if stds else None,
            "max_kl_mean": max(kls) if kls else None,
            "min_entropy_mean": min(entropies) if entropies else None,
            "final_entropy_mean": entropies[-1] if entropies else None,
            "warnings": warnings_list,
            "healthy": not warnings_list,
        }
    )


# ---------- 8) watch-metrics ---------------------------------------------


def _watch_metrics_once(path: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _, rec in _iter_jsonl(path):
        rows.append(rec)
    if not rows:
        return {"metrics_jsonl": path, "n_steps": 0, "latest": None}

    def series(key: str) -> list[float]:
        return [float(r[key]) for r in rows if key in r and r[key] is not None]

    rewards = series("reward_mean")
    entropies = series("entropy_mean")
    kls = series("kl_mean")
    response_lens = series("response_len_mean")
    loss_keys = sorted({k for r in rows for k in r if k.endswith("_loss")})
    losses_by_kind: dict[str, list[float]] = {k: series(k) for k in loss_keys}

    def trajectory(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "first": round(values[0], 4),
            "last": round(values[-1], 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    return {
        "metrics_jsonl": path,
        "n_steps": len(rows),
        "latest": rows[-1],
        "reward_mean": trajectory(rewards),
        "entropy_mean": trajectory(entropies),
        "kl_mean": trajectory(kls),
        "response_len_mean": trajectory(response_lens),
        "losses": {k: trajectory(v) for k, v in losses_by_kind.items()} or None,
    }


def _watch_metrics(args: argparse.Namespace) -> None:
    import sys
    import time

    if not args.watch:
        summary = _watch_metrics_once(args.metrics_jsonl)
        _emit(summary)
        return

    print(f"watching {args.metrics_jsonl} (Ctrl-C to stop)", file=sys.stderr)
    try:
        while True:
            try:
                summary = _watch_metrics_once(args.metrics_jsonl)
            except FileNotFoundError:
                print(f"  {args.metrics_jsonl} not found yet…", file=sys.stderr, flush=True)
                time.sleep(args.interval)
                continue
            # ANSI clear-screen + cursor home for a stable redraw.
            sys.stdout.write("\033[2J\033[H")
            _emit(summary)
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)


# ---------- 7) eval-parity ----------------------------------------------


def _eval_parity(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from baby_whale_v4.inference.engine import Engine, GenerationOptions
    from baby_whale_v4.inference.prefix_cache import PrefixCache

    tokenizer = _load_tokenizer(args.tokenizer_path)
    model, cfg = _load_model_from_ckpt(args.from_checkpoint, tokenizer)
    model.eval()
    mx.random.seed(args.seed)

    engine = Engine(
        model=model,
        config=cfg,
        tokenizer_hash=tokenizer.hash_signature(),
        prefix_cache=PrefixCache(capacity=4) if args.with_prefix_cache else None,
    )
    opts = GenerationOptions(max_new_tokens=args.max_new_tokens, mode="greedy")
    prompt_ids = tokenizer.encode(args.prompt)
    if len(prompt_ids) + args.max_new_tokens > cfg.context_length:
        raise ValueError("prompt + max-new-tokens exceeds context_length")

    engine_out = engine.generate(prompt_ids, opts)

    seq = mx.array([prompt_ids], dtype=mx.int32)
    ref_tokens: list[int] = []
    for _ in range(args.max_new_tokens):
        logits = model(seq).logits[:, -1, :]
        nxt = mx.argmax(logits, axis=-1).reshape(1, 1)
        ref_tokens.append(int(nxt[0, 0]))
        seq = mx.concatenate([seq, nxt], axis=1)

    n_match = sum(1 for a, b in zip(engine_out, ref_tokens, strict=False) if a == b)
    _emit(
        {
            "from_checkpoint": args.from_checkpoint,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "with_prefix_cache": args.with_prefix_cache,
            "engine_tokens": engine_out,
            "ref_tokens": ref_tokens,
            "n_match": n_match,
            "n_total": args.max_new_tokens,
            "parity_ok": engine_out == ref_tokens,
        }
    )


# ---------- registration ------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "eval-tokenizer",
        help="Bytes-per-token and fertility for a tokenizer on a JSONL.",
    )
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--input-jsonl", type=str, required=True)
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_eval_tokenizer)

    p = sub.add_parser(
        "eval-bpb",
        help="Held-out perplexity and bits-per-byte for a pretrain/midtrain checkpoint.",
    )
    p.add_argument("--from-checkpoint", type=str, required=True)
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--input-jsonl", type=str, required=True)
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--limit-blocks", type=int, default=None)
    p.set_defaults(func=_eval_bpb)

    p = sub.add_parser(
        "eval-code",
        help="Pass@1 on a CodeProblem JSONL via the code-execution sandbox.",
    )
    p.add_argument("--from-checkpoint", type=str, required=True)
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--problems-jsonl", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--chat-template",
        action="store_true",
        help="Wrap prompts as <|user|>...<|eot|><|assistant|> (for SFT'd checkpoints).",
    )
    p.add_argument("--timeout-sec", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--show-details", action="store_true")
    p.set_defaults(func=_eval_code)

    p = sub.add_parser(
        "eval-ifeval",
        help="Minimal verifiable instruction-following accuracy (8 deterministic checks).",
    )
    p.add_argument("--from-checkpoint", type=str, required=True)
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--show-details", action="store_true")
    p.set_defaults(func=_eval_ifeval)

    p = sub.add_parser(
        "eval-dpo",
        help="DPO reward accuracy and mean margin on a preference JSONL.",
    )
    p.add_argument("--from-checkpoint", type=str, required=True)
    p.add_argument(
        "--ref-checkpoint",
        type=str,
        default=None,
        help="Reference policy. Defaults to a copy of --from-checkpoint (no-signal baseline).",
    )
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--input-jsonl", type=str, required=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-prompt-tokens", type=int, default=128)
    p.add_argument("--max-response-tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_eval_dpo)

    p = sub.add_parser(
        "eval-rl-health",
        help="Threshold check on a GRPO/PPO/RLOO metrics JSONL.",
    )
    p.add_argument("--metrics-jsonl", type=str, required=True)
    p.add_argument("--max-kl", type=float, default=20.0)
    p.add_argument("--min-entropy", type=float, default=0.1)
    p.add_argument("--min-reward-std", type=float, default=1e-4)
    p.add_argument("--min-reward-delta", type=float, default=0.0)
    p.set_defaults(func=_eval_rl_health)

    p = sub.add_parser(
        "watch-metrics",
        help="One-shot or live trajectory summary of a training metrics JSONL.",
    )
    p.add_argument("--metrics-jsonl", type=str, required=True)
    p.add_argument(
        "--watch",
        action="store_true",
        help="Poll the file forever and redraw. Default: read once and exit.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between redraws when --watch is set.",
    )
    p.set_defaults(func=_watch_metrics)

    p = sub.add_parser(
        "eval-parity",
        help="Engine.generate vs. naive greedy decode parity on a single prompt.",
    )
    p.add_argument("--from-checkpoint", type=str, required=True)
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--prompt", type=str, default="hello")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--with-prefix-cache", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_eval_parity)
