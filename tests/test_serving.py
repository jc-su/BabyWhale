"""Isolation tests for the continuous-batching serving loop (no HTTP).

Validates the concurrency contract of ``BatchingServer`` directly: the model is
only ever touched by the loop thread, results are correct, and requests are
served concurrently (a short request is not blocked behind a long one).
"""

from __future__ import annotations

import threading
import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import ByteTokenizer
from baby_whale_v4.inference.engine import Engine, GenerationOptions
from baby_whale_v4.inference.serving import BatchingServer


def _engine(*, ctx: int = 32, mtp: int = 0) -> tuple[Engine, ByteTokenizer]:
    mx.random.seed(0)
    tok = ByteTokenizer()
    base = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=ctx).to_dict()
    cfg = BabyWhaleV4Config.from_dict({**base, "mtp_heads": mtp})
    model = BabyWhaleV4Model(cfg)
    model.eval()
    return Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature()), tok


class TestBatchingServer(unittest.TestCase):
    def test_served_matches_direct_greedy(self) -> None:
        engine, tok = _engine()
        prompts = [tok.encode(w) for w in ("alpha", "beta gamma", "d")]
        opts = GenerationOptions(max_new_tokens=6, mode="greedy")
        # Reference computed BEFORE start() — no loop thread, so no concurrent
        # model access. Chunked-prefill + interleaved decode must match.
        refs = [engine.generate(p, opts) for p in prompts]

        server = BatchingServer(engine, prefill_chunk=2)
        server.start()
        try:
            handles = [server.submit(p, opts) for p in prompts]
            outs = [list(h.result(timeout=10).generated) for h in handles]
        finally:
            server.stop()
        self.assertEqual(outs, refs)

    def test_concurrent_clients_all_correct(self) -> None:
        engine, tok = _engine()
        opts = GenerationOptions(max_new_tokens=5, mode="greedy")
        prompts = {w: tok.encode(w) for w in ("one", "two", "three", "four")}
        refs = {w: engine.generate(p, opts) for w, p in prompts.items()}

        server = BatchingServer(engine, prefill_chunk=2)
        server.start()
        results: dict[str, list[int]] = {}

        def worker(word: str) -> None:
            handle = server.submit(prompts[word], opts)
            results[word] = list(handle.result(timeout=15).generated)

        try:
            threads = [threading.Thread(target=worker, args=(w,)) for w in prompts]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=20)
        finally:
            server.stop()
        self.assertEqual(results, refs)

    def test_short_request_completes_before_long(self) -> None:
        # Continuous batching: submit a long request, then a short one. The short
        # one must finish first (round-robin decode), i.e. it is not blocked
        # behind the long request's decode. Order is decided by loop-tick order,
        # not wall-clock, so the ~19-token gap makes this deterministic.
        engine, tok = _engine(ctx=64)
        server = BatchingServer(engine, prefill_chunk=2)
        server.start()
        order: list[str] = []
        lock = threading.Lock()

        def wait_record(name: str, handle) -> None:
            handle.result(timeout=20)
            with lock:
                order.append(name)

        try:
            long_h = server.submit(tok.encode("long"), GenerationOptions(max_new_tokens=20))
            short_h = server.submit(tok.encode("short"), GenerationOptions(max_new_tokens=1))
            t_long = threading.Thread(target=wait_record, args=("long", long_h))
            t_short = threading.Thread(target=wait_record, args=("short", short_h))
            t_long.start()
            t_short.start()
            t_long.join(timeout=25)
            t_short.join(timeout=25)
        finally:
            server.stop()
        self.assertEqual(order[0], "short")
        self.assertEqual(len(order), 2)

    def test_streaming_tokens_match_final(self) -> None:
        engine, tok = _engine()
        server = BatchingServer(engine, prefill_chunk=2)
        server.start()
        try:
            handle = server.submit(tok.encode("hello"), GenerationOptions(max_new_tokens=4))
            streamed: list[int] = []
            while True:
                token = handle.next_token(timeout=10)
                if token is None:
                    break
                streamed.append(token)
            self.assertIsNone(handle.error)
            self.assertEqual(streamed, list(handle.result(timeout=1).generated))
        finally:
            server.stop()

    def test_speculative_through_loop(self) -> None:
        engine, tok = _engine(mtp=2)
        prompt = tok.encode("abc")
        ref = engine.generate(prompt, GenerationOptions(max_new_tokens=5, mode="speculative"))
        server = BatchingServer(engine, prefill_chunk=2)
        server.start()
        try:
            handle = server.submit(prompt, GenerationOptions(max_new_tokens=5, mode="speculative"))
            out = list(handle.result(timeout=10).generated)
        finally:
            server.stop()
        self.assertEqual(out, ref)

    def test_run_control_executes_on_loop(self) -> None:
        engine, _tok = _engine()
        server = BatchingServer(engine, prefill_chunk=2)
        server.start()
        try:
            value = server.run_control(lambda e: e.config.n_layer)
        finally:
            server.stop()
        self.assertEqual(value, engine.config.n_layer)

    def test_validation_error_surfaces(self) -> None:
        engine, _tok = _engine(ctx=16)
        server = BatchingServer(engine, prefill_chunk=2)
        server.start()
        try:
            handle = server.submit([1] * 40, GenerationOptions(max_new_tokens=2))
            with self.assertRaisesRegex(ValueError, "context_length"):
                handle.result(timeout=10)
        finally:
            server.stop()

    def test_cancel_stops_generation(self) -> None:
        engine, tok = _engine(ctx=128)
        server = BatchingServer(engine, prefill_chunk=2)
        server.start()
        try:
            handle = server.submit(
                tok.encode("hello"), GenerationOptions(max_new_tokens=50, mode="greedy")
            )
            self.assertIsNotNone(handle.next_token(timeout=10))  # a token or two flow
            handle.cancel()
            state = handle.result(timeout=10)
            self.assertTrue(state.cancelled)
            self.assertLess(state.total_emitted, 50)  # stopped well short of the cap
        finally:
            server.stop()

    def test_submit_before_start_raises(self) -> None:
        engine, tok = _engine()
        server = BatchingServer(engine, prefill_chunk=2)
        with self.assertRaisesRegex(RuntimeError, "not running"):
            server.submit(tok.encode("hi"), GenerationOptions(max_new_tokens=1))


if __name__ == "__main__":
    unittest.main()
