import argparse
import json
from pathlib import Path


def _prepare_hf(args: argparse.Namespace) -> None:
    from baby_whale_v4.data import HFSource, materialize_hf_source

    source = HFSource(
        dataset_id=args.dataset_id,
        subset=args.subset,
        split=args.split,
        kind=args.kind,
        text_field=args.text_field,
        messages_field=args.messages_field,
        prompt_field=args.prompt_field,
        chosen_field=args.chosen_field,
        rejected_field=args.rejected_field,
        tools_field=args.tools_field,
        limit=args.limit,
        seed=args.seed,
        license_note=args.license_note,
        streaming=not args.no_streaming,
    )
    materialized = materialize_hf_source(source, args.out_dir)
    print(
        json.dumps(
            {
                "path": str(materialized.path),
                "manifest_path": str(materialized.manifest_path),
                "rows": materialized.rows,
                "kind": source.kind,
                "dataset_id": source.dataset_id,
                "subset": source.subset,
                "split": source.split,
            },
            indent=2,
        )
    )


def _train_tokenizer(args: argparse.Namespace) -> None:
    from baby_whale_v4.data import read_normalized_texts, train_byte_bpe

    texts = read_normalized_texts(args.input_jsonl, limit=args.limit)
    tokenizer = train_byte_bpe(
        texts,
        vocab_size=args.vocab_size,
        min_pair_count=args.min_pair_count,
    )
    out = tokenizer.save(args.out)
    print(
        json.dumps(
            {
                "path": str(out),
                "kind": tokenizer.kind,
                "vocab_size": tokenizer.vocab_size,
                "merges": len(tokenizer.merges),
                "hash": tokenizer.hash_signature(),
            },
            indent=2,
        )
    )


def _prepare_code_tasks(args: argparse.Namespace) -> None:
    from baby_whale_v4.rl import (
        load_humaneval_from_hf,
        load_mbpp_from_hf,
        save_problems_to_jsonl,
    )

    if args.dataset == "mbpp":
        problems = load_mbpp_from_hf(split=args.split, limit=args.limit)
    elif args.dataset == "humaneval":
        problems = load_humaneval_from_hf(limit=args.limit)
    else:
        raise ValueError(f"unknown code dataset {args.dataset!r}")
    save_problems_to_jsonl(problems, args.out)
    print(
        json.dumps(
            {
                "path": str(args.out),
                "n_problems": len(problems),
                "dataset": args.dataset,
                "split": args.split if args.dataset == "mbpp" else "test",
            },
            indent=2,
        )
    )


def _prepare_code_prefs(args: argparse.Namespace) -> None:
    """Generate a coding preference JSONL from a CodeProblem JSONL.

    Each MBPP-style problem produces one preference row where the chosen
    response is the canonical solution and the rejected response is the
    canonical solution of a *different* problem (clearly wrong for this
    prompt). Both rolled into the chat template so the data is consistent
    with the SFT'd model.
    """
    import json
    import random

    from baby_whale_v4.rl import load_problems_from_jsonl

    problems = load_problems_from_jsonl(args.problems_jsonl)
    problems = [p for p in problems if p.canonical_solution]
    if len(problems) < 2:
        raise ValueError("need at least 2 problems with canonical_solution for negatives")

    rng = random.Random(args.seed)

    def _wrap_chosen(prompt: str, code: str) -> tuple[str, str]:
        user_text = prompt.rstrip()
        chat_prompt = f"<|user|>{user_text}<|eot|><|assistant|>"
        chat_response = f"```python\n{code.strip()}\n```<|eot|>"
        return chat_prompt, chat_response

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, problem in enumerate(problems):
            # Pick a negative from a different problem.
            j = rng.randrange(len(problems) - 1)
            if j >= i:
                j += 1
            negative = problems[j]
            # Both filtered for truthy canonical_solution above; reassert for ty.
            chosen_code = problem.canonical_solution
            rejected_code = negative.canonical_solution
            if chosen_code is None or rejected_code is None:
                raise AssertionError("canonical_solution unexpectedly None after filter")
            prompt_str, chosen_str = _wrap_chosen(problem.prompt, chosen_code)
            _, rejected_str = _wrap_chosen(problem.prompt, rejected_code)
            f.write(
                json.dumps(
                    {
                        "kind": "preference",
                        "prompt": prompt_str,
                        "chosen": chosen_str,
                        "rejected": rejected_str,
                    }
                )
                + "\n"
            )
            n_written += 1
            if args.limit is not None and n_written >= args.limit:
                break
    print(json.dumps({"out": str(out_path), "n_written": n_written}))


def _pack_jsonl(args: argparse.Namespace) -> None:
    from baby_whale_v4.data import (
        load_tokenizer,
        pack_normalized_jsonl,
        save_packed_token_file,
    )

    tokenizer = load_tokenizer(args.tokenizer_path)
    dataset = pack_normalized_jsonl(
        args.input_jsonl,
        tokenizer=tokenizer,
        block_size=args.block_size,
        limit=args.limit,
    )
    packed = save_packed_token_file(
        dataset,
        args.out,
        tokenizer_hash=tokenizer.hash_signature(),
        sources=[args.input_jsonl],
    )
    print(
        json.dumps(
            {
                "path": str(packed.path),
                "manifest_path": str(packed.manifest_path),
                "n_tokens": packed.n_tokens,
                "n_blocks": packed.n_blocks,
                "block_size": packed.block_size,
                "tokenizer_hash": packed.tokenizer_hash,
            },
            indent=2,
        )
    )


def register(sub: argparse._SubParsersAction) -> None:
    prep = sub.add_parser("prepare-hf", help="Materialize a small Hugging Face subset to JSONL.")
    prep.add_argument("--dataset-id", required=True)
    prep.add_argument("--subset", default=None)
    prep.add_argument("--split", default="train")
    prep.add_argument(
        "--kind",
        choices=["pretrain", "chat", "preference", "tool_trace"],
        required=True,
    )
    prep.add_argument("--out-dir", required=True)
    prep.add_argument("--limit", type=int, default=1000)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--license-note", default="check upstream dataset card")
    prep.add_argument("--text-field", default="text")
    prep.add_argument("--messages-field", default="messages")
    prep.add_argument("--prompt-field", default="prompt")
    prep.add_argument("--chosen-field", default="chosen")
    prep.add_argument("--rejected-field", default="rejected")
    prep.add_argument("--tools-field", default="tools")
    prep.add_argument("--no-streaming", action="store_true")
    prep.set_defaults(func=_prepare_hf)

    tok_p = sub.add_parser("train-tokenizer", help="Train an educational byte-BPE tokenizer.")
    tok_p.add_argument("--input-jsonl", required=True)
    tok_p.add_argument("--out", required=True)
    tok_p.add_argument("--vocab-size", type=int, default=512)
    tok_p.add_argument("--min-pair-count", type=int, default=2)
    tok_p.add_argument("--limit", type=int, default=None)
    tok_p.set_defaults(func=_train_tokenizer)

    pack_p = sub.add_parser("pack-jsonl", help="Pack normalized JSONL into token blocks.")
    pack_p.add_argument("--input-jsonl", required=True)
    pack_p.add_argument("--out", required=True)
    pack_p.add_argument("--tokenizer-path", default=None)
    pack_p.add_argument("--block-size", type=int, default=128)
    pack_p.add_argument("--limit", type=int, default=None)
    pack_p.set_defaults(func=_pack_jsonl)

    code_p = sub.add_parser(
        "prepare-code-tasks",
        help="Download MBPP or HumanEval problems and save to a CodeProblem JSONL.",
    )
    code_p.add_argument("--dataset", choices=["mbpp", "humaneval"], required=True)
    code_p.add_argument(
        "--split",
        default="test",
        help="MBPP split (train/validation/test/prompt). Ignored for HumanEval.",
    )
    code_p.add_argument("--limit", type=int, default=None)
    code_p.add_argument("--out", required=True)
    code_p.set_defaults(func=_prepare_code_tasks)

    pref_p = sub.add_parser(
        "prepare-code-prefs",
        help="Build a coding preference JSONL (chosen = canonical, rejected = canonical of a different problem).",
    )
    pref_p.add_argument(
        "--problems-jsonl",
        required=True,
        help="CodeProblem JSONL with canonical_solution fields populated.",
    )
    pref_p.add_argument("--out", required=True)
    pref_p.add_argument("--limit", type=int, default=None)
    pref_p.add_argument("--seed", type=int, default=0)
    pref_p.set_defaults(func=_prepare_code_prefs)
