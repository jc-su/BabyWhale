"""Cohort batched decode in the scheduler.

Same-length, same-sampling decoding requests are advanced by one shared batched
forward per tick (continuous batching). Greedy output must be token-identical to
per-request decode, and the scheduler must actually collapse the cohort's decode
forwards into one per step.
"""

from __future__ import annotations

import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import ByteTokenizer
from baby_whale_v4.inference.batched import decode_group_batched, decode_ragged_batched
from baby_whale_v4.inference.engine import Engine, GenerationOptions, RequestState
from baby_whale_v4.inference.scheduler import RequestScheduler
from baby_whale_v4.typing import RequestId


def _engine(*, ctx: int = 48, hybrid: bool = False) -> tuple[Engine, ByteTokenizer]:
    mx.random.seed(0)
    tok = ByteTokenizer()
    if hybrid:
        cfg = BabyWhaleV4Config.hybrid_tiny(vocab_size=tok.vocab_size, context_length=ctx)
    else:
        cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=ctx)
    model = BabyWhaleV4Model(cfg)
    model.eval()
    return Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature()), tok


def _prefilled(engine: Engine, prompts: list[list[int]], opts: GenerationOptions):
    states = []
    for i, prompt in enumerate(prompts):
        state = engine.new_request(RequestId(f"s{i}"), prompt, opts)
        while state.remaining_prefill > 0:
            engine.prefill_chunk(state, chunk_size=engine.config.context_length)
        states.append(state)
    return states


class TestBatchedScheduler(unittest.TestCase):
    def test_decode_group_batched_matches_per_request(self) -> None:
        engine, tok = _engine()
        opts = GenerationOptions(max_new_tokens=5, mode="greedy")
        prompts = [tok.encode("aaaaa"), tok.encode("bbbbb"), tok.encode("ccccc")]  # all len 5
        refs = [engine.generate(p, opts) for p in prompts]

        states = _prefilled(engine, prompts, opts)
        for _ in range(10):
            if all(s.finished for s in states):
                break
            decode_group_batched(engine.model, states)

        for state, ref in zip(states, refs, strict=True):
            self.assertEqual(list(state.generated), ref)

    def test_decode_group_batched_hybrid_attention(self) -> None:
        # HCA + CSA + sliding layers: uniform-length batching must still be
        # token-identical (positions/mask are shared across the batch).
        engine, tok = _engine(hybrid=True)
        opts = GenerationOptions(max_new_tokens=4, mode="greedy")
        prompts = [tok.encode("hello"), tok.encode("world")]  # len 5
        refs = [engine.generate(p, opts) for p in prompts]

        states = _prefilled(engine, prompts, opts)
        for _ in range(10):
            if all(s.finished for s in states):
                break
            decode_group_batched(engine.model, states)

        for state, ref in zip(states, refs, strict=True):
            self.assertEqual(list(state.generated), ref)

    def test_scheduler_cohort_matches_per_request(self) -> None:
        engine, tok = _engine()
        opts = GenerationOptions(max_new_tokens=6, mode="greedy")
        prompts = {
            "p1": tok.encode("hello"),
            "p2": tok.encode("world"),
            "p3": tok.encode("there"),
        }  # all len 5 -> one cohort
        refs = {k: engine.generate(p, opts) for k, p in prompts.items()}

        sched = RequestScheduler(engine, prefill_chunk=8)
        states = {k: sched.submit(k, p, opts) for k, p in prompts.items()}
        sched.run_until_done()
        for k, state in states.items():
            self.assertEqual(list(state.generated), refs[k])

    def test_scheduler_takes_batched_path_for_cohort(self) -> None:
        engine, tok = _engine()
        opts = GenerationOptions(max_new_tokens=5, mode="greedy")
        prompts = [tok.encode(w) for w in ("aaaaa", "bbbbb", "ccccc", "ddddd")]  # all len 5
        group_sizes: list[int] = []

        class _RecordingScheduler(RequestScheduler):
            def _advance(self, group: list[RequestState]) -> None:
                group_sizes.append(len(group))
                super()._advance(group)

        sched = _RecordingScheduler(engine, prefill_chunk=8)
        for i, prompt in enumerate(prompts):
            sched.submit(f"r{i}", prompt, opts)
        sched.run_until_done()

        # A same-length cohort of >= 2 formed -> the scheduler batched it (rather
        # than decoding each request separately).
        self.assertTrue(any(size >= 2 for size in group_sizes))

    def test_scheduler_mixed_lengths_still_correct(self) -> None:
        # Different-length prompts fall into different cohorts (or singletons)
        # and decode per-request; output must still be correct.
        engine, tok = _engine(ctx=64)
        opts = GenerationOptions(max_new_tokens=4, mode="greedy")
        prompts = {"short": tok.encode("hi"), "long": tok.encode("a longer prompt here")}
        refs = {k: engine.generate(p, opts) for k, p in prompts.items()}

        sched = RequestScheduler(engine, prefill_chunk=4)
        states = {k: sched.submit(k, p, opts) for k, p in prompts.items()}
        sched.run_until_done()
        for k, state in states.items():
            self.assertEqual(list(state.generated), refs[k])

    def test_ragged_batched_matches_per_request(self) -> None:
        # DIFFERENT-length prompts decoded in ONE batched forward (mixed lengths)
        # must be token-identical to per-request greedy decode.
        engine, tok = _engine(ctx=48)  # tiny config is all sliding_mqa
        opts = GenerationOptions(max_new_tokens=6, mode="greedy")
        prompts = [tok.encode("hi"), tok.encode("a longer prompt here"), tok.encode("mid one")]
        self.assertGreater(len({len(p) for p in prompts}), 1)  # genuinely ragged
        refs = [engine.generate(p, opts) for p in prompts]

        states = _prefilled(engine, prompts, opts)
        for _ in range(10):
            if all(s.finished for s in states):
                break
            decode_ragged_batched(engine.model, states)

        for state, ref in zip(states, refs, strict=True):
            self.assertEqual(list(state.generated), ref)

    def test_ragged_scheduler_batches_mixed_lengths(self) -> None:
        engine, tok = _engine(ctx=64)
        opts = GenerationOptions(max_new_tokens=5, mode="greedy")
        # Similar-but-different lengths (5/6/7 bytes): with prefill_chunk=4 all
        # finish prefill on the same tick, then decode together at DIFFERENT
        # lengths — exactly the case ragged batching is for.
        prompts = {
            "a": tok.encode("alpha"),
            "b": tok.encode("bravos"),
            "c": tok.encode("charlie"),
        }
        self.assertEqual(len({len(p) for p in prompts.values()}), 3)  # distinct lengths
        refs = {k: engine.generate(p, opts) for k, p in prompts.items()}

        group_max = [0]

        class _RaggedScheduler(RequestScheduler):
            def _advance(self, group: list[RequestState]) -> None:
                group_max[0] = max(group_max[0], len(group))
                super()._advance(group)

        sched = _RaggedScheduler(engine, prefill_chunk=4, ragged=True)
        states = {k: sched.submit(k, p, opts) for k, p in prompts.items()}
        sched.run_until_done()

        for k, state in states.items():
            self.assertEqual(list(state.generated), refs[k])
        # Mixed-length requests shared a forward (a cohort of >= 2 formed).
        self.assertGreaterEqual(group_max[0], 2)

    def test_ragged_rejects_non_sliding_model(self) -> None:
        engine, tok = _engine(hybrid=True)  # schedule includes hca / csa
        opts = GenerationOptions(max_new_tokens=2, mode="greedy")
        states = _prefilled(engine, [tok.encode("hi"), tok.encode("longer one")], opts)
        with self.assertRaisesRegex(ValueError, "sliding_mqa"):
            decode_ragged_batched(engine.model, states)


if __name__ == "__main__":
    unittest.main()
