"""Tests for the new vLLM/SGLang-inspired inference optimizations.

Covers:
* RadixKVCache match/insert/split/eviction
* Engine.fork() + radix-cache integration (prompt prefill is shared)
* GenerationOptions extensions: top-p, min-p, validation
* RequestState.cancel() / scheduler honors cancellation
* decode_step_group API contract
* spec_decode acceptance-rate diagnostics
"""

from __future__ import annotations

import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import ByteTokenizer
from baby_whale_v4.inference.batched import (
    decode_step_batched,
    generate_batched,
    tile_cache,
)
from baby_whale_v4.inference.engine import Engine, GenerationOptions
from baby_whale_v4.inference.paged_kv import (
    PagedKVCache,
    PagedKVConfig,
    PagedKVPool,
)
from baby_whale_v4.inference.radix_cache import RadixKVCache
from baby_whale_v4.inference.scheduler import RequestScheduler
from baby_whale_v4.typing import RequestId


def _tiny_engine(*, ctx: int = 24, with_radix: bool = False) -> tuple[Engine, ByteTokenizer]:
    mx.random.seed(0)
    tok = ByteTokenizer()
    cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=ctx)
    model = BabyWhaleV4Model(cfg)
    model.eval()
    radix = (
        RadixKVCache(config=cfg, tokenizer_hash=tok.hash_signature(), capacity_nodes=32)
        if with_radix
        else None
    )
    engine = Engine(
        model=model,
        config=cfg,
        tokenizer_hash=tok.hash_signature(),
        radix_cache=radix,
    )
    return engine, tok


class TestRadixKVCache(unittest.TestCase):
    def test_miss_returns_none(self) -> None:
        engine, _tok = _tiny_engine(with_radix=True)
        assert engine.radix_cache is not None
        self.assertIsNone(engine.radix_cache.match([1, 2, 3]))
        self.assertEqual(engine.radix_cache.misses, 1)
        self.assertEqual(engine.radix_cache.hits, 0)

    def test_insert_and_full_match(self) -> None:
        engine, tok = _tiny_engine(with_radix=True)
        prompt = tok.encode("hello")
        engine.generate(prompt, GenerationOptions(max_new_tokens=2, mode="greedy"))
        # The first generate prefilled and committed the prompt KV to the
        # radix cache. A second call with the same prompt must hit it.
        assert engine.radix_cache is not None
        hits_before = engine.radix_cache.hits
        engine.generate(prompt, GenerationOptions(max_new_tokens=2, mode="greedy"))
        self.assertGreater(engine.radix_cache.hits, hits_before)

    def test_split_on_partial_insert(self) -> None:
        # Insert two prefixes that share an initial span. The radix tree
        # must split the original edge so the shared portion becomes an
        # internal node, and both leaves keep their own payload.
        engine, tok = _tiny_engine(ctx=32, with_radix=True)
        assert engine.radix_cache is not None
        prompt_a = tok.encode("shared-A")
        prompt_b = tok.encode("shared-B")
        engine.generate(prompt_a, GenerationOptions(max_new_tokens=1, mode="greedy"))
        engine.generate(prompt_b, GenerationOptions(max_new_tokens=1, mode="greedy"))
        # After both inserts, querying each prompt should hit its own leaf.
        hits_before = engine.radix_cache.hits
        engine.radix_cache.match(prompt_a)
        engine.radix_cache.match(prompt_b)
        self.assertEqual(engine.radix_cache.hits - hits_before, 2)
        # And the tree must have more than 2 nodes (root + split point + 2 leaves).
        self.assertGreaterEqual(engine.radix_cache.n_nodes, 4)

    def test_capacity_eviction(self) -> None:
        engine, tok = _tiny_engine(ctx=32, with_radix=True)
        # Cap is 32 nodes; pump distinct prompts through and verify the tree
        # never exceeds the cap.
        assert engine.radix_cache is not None
        for i in range(60):
            prompt = tok.encode(f"p{i}")
            engine.generate(prompt, GenerationOptions(max_new_tokens=1, mode="greedy"))
        self.assertLessEqual(engine.radix_cache.n_nodes, 32)

    def test_match_preserves_mla_latents(self) -> None:
        mx.random.seed(0)
        tok = ByteTokenizer()
        cfg = BabyWhaleV4Config.mla_tiny(vocab_size=tok.vocab_size, context_length=32)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        radix = RadixKVCache(config=cfg, tokenizer_hash=tok.hash_signature(), capacity_nodes=32)
        engine = Engine(
            model=model,
            config=cfg,
            tokenizer_hash=tok.hash_signature(),
            radix_cache=radix,
        )
        prompt = tok.encode("latent")
        engine.generate(prompt, GenerationOptions(max_new_tokens=1, mode="greedy"))
        hit = radix.match(prompt)
        self.assertIsNotNone(hit)
        if hit is None:
            raise AssertionError("expected radix cache hit")
        _n, restored, _logits = hit
        self.assertEqual(restored.max_sequence_length(), len(prompt))
        self.assertIsNotNone((restored.latents or [None])[0])


class TestEngineFork(unittest.TestCase):
    def test_fork_returns_n_states(self) -> None:
        engine, tok = _tiny_engine(ctx=32, with_radix=True)
        prompt = tok.encode("hi")
        branches = engine.fork(
            prompt, n=4, options=GenerationOptions(max_new_tokens=2, mode="greedy")
        )
        self.assertEqual(len(branches), 4)
        # Each branch is a fresh request that hit the radix cache populated
        # by the anchor prefill.
        for state in branches:
            self.assertTrue(state.used_prefix_cache)
            self.assertEqual(state.prefilled, len(prompt))

    def test_fork_decode_step_group_advances_all(self) -> None:
        engine, tok = _tiny_engine(ctx=32, with_radix=True)
        prompt = tok.encode("hi")
        branches = engine.fork(
            prompt, n=3, options=GenerationOptions(max_new_tokens=4, mode="greedy")
        )
        engine.decode_step_group(branches)
        for state in branches:
            self.assertEqual(state.total_emitted, 1)

    def test_fork_requires_positive_n(self) -> None:
        engine, tok = _tiny_engine(with_radix=True)
        with self.assertRaisesRegex(ValueError, "positive"):
            engine.fork(tok.encode("hi"), n=0, options=GenerationOptions(max_new_tokens=1))


class TestGenerationOptionsExtensions(unittest.TestCase):
    def test_top_p_validation(self) -> None:
        GenerationOptions(top_p=0.9)
        with self.assertRaisesRegex(ValueError, "top_p"):
            GenerationOptions(top_p=0.0)
        with self.assertRaisesRegex(ValueError, "top_p"):
            GenerationOptions(top_p=1.5)

    def test_min_p_validation(self) -> None:
        GenerationOptions(min_p=0.05)
        with self.assertRaisesRegex(ValueError, "min_p"):
            GenerationOptions(min_p=-0.1)
        with self.assertRaisesRegex(ValueError, "min_p"):
            GenerationOptions(min_p=1.0)

    def test_top_p_sample_runs(self) -> None:
        engine, tok = _tiny_engine(ctx=24)
        out = engine.generate(
            tok.encode("hi"),
            GenerationOptions(max_new_tokens=4, mode="sample", top_p=0.9),
        )
        self.assertEqual(len(out), 4)

    def test_min_p_sample_runs(self) -> None:
        engine, tok = _tiny_engine(ctx=24)
        out = engine.generate(
            tok.encode("hi"),
            GenerationOptions(max_new_tokens=4, mode="sample", min_p=0.05),
        )
        self.assertEqual(len(out), 4)


class TestRequestCancellation(unittest.TestCase):
    def test_cancel_stops_further_decode_in_scheduler(self) -> None:
        engine, tok = _tiny_engine(ctx=24)
        sched = RequestScheduler(engine, prefill_chunk=2)
        opts = GenerationOptions(max_new_tokens=10, mode="greedy")
        state = sched.submit("req-1", tok.encode("hello"), opts)
        # Run a couple of ticks, then cancel.
        sched.tick()
        sched.tick()
        before = state.total_emitted
        state.cancel()
        sched.run_until_done(max_ticks=50)
        # No new tokens were generated after cancel.
        self.assertEqual(state.total_emitted, before)
        self.assertTrue(state.finished)
        self.assertTrue(state.cancelled)


class TestSpecDecodeMetrics(unittest.TestCase):
    def test_acceptance_rate_is_bounded(self) -> None:
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.from_dict(
            {
                **BabyWhaleV4Config.tiny(vocab_size=32, context_length=24).to_dict(),
                "mtp_heads": 2,
                "name": "tiny-accept",
            }
        )
        model = BabyWhaleV4Model(cfg)
        model.eval()
        prefix = mx.random.randint(0, 32, (1, 4))
        result = model.spec_decode(prefix, max_new_tokens=5)
        self.assertGreaterEqual(result.acceptance_rate, 0.0)
        self.assertLessEqual(result.acceptance_rate, 1.0)
        # n_verify_calls is at least 1 when m > 0 and we emit at least 2 tokens.
        self.assertGreaterEqual(result.n_verify_calls, 1)


class TestBatchedDecode(unittest.TestCase):
    def test_fork_batched_returns_state_at_correct_shape(self) -> None:
        engine, tok = _tiny_engine(ctx=32, with_radix=True)
        prompt = tok.encode("hi")
        state = engine.fork_batched(
            prompt, n=3, options=GenerationOptions(max_new_tokens=4, mode="greedy")
        )
        self.assertEqual(state.n_branches, 3)
        self.assertEqual(state.last_logits.shape[0], 3)
        # Cache layers should all have batch dim = 3 now.
        for k in state.cache.keys:
            if k is not None:
                self.assertEqual(k.shape[0], 3)

    def test_batched_decode_matches_serial_decode_greedy(self) -> None:
        engine, tok = _tiny_engine(ctx=32, with_radix=True)
        prompt = tok.encode("hello")
        opts = GenerationOptions(max_new_tokens=5, mode="greedy")

        # Reference: 3 independent greedy generates (same prompt → same output).
        ref_tokens = engine.generate(prompt, opts)

        # Batched: prefill once, decode 3 branches together.
        state = engine.fork_batched(prompt, n=3, options=opts)
        generate_batched(engine.model, state)

        # Every branch should match the reference greedy decode token-for-token.
        for branch in state.generated:
            self.assertEqual(branch, ref_tokens)

    def test_batched_decode_step_advances_one_token_per_branch(self) -> None:
        engine, tok = _tiny_engine(ctx=32, with_radix=True)
        state = engine.fork_batched(
            tok.encode("hi"),
            n=4,
            options=GenerationOptions(max_new_tokens=3, mode="greedy"),
        )
        decode_step_batched(engine.model, state)
        for branch in state.generated:
            self.assertEqual(len(branch), 1)

    def test_tile_cache_preserves_b1_data(self) -> None:
        engine, tok = _tiny_engine(ctx=24)
        # Use plain generate to populate a B=1 cache via an anchor request.
        state = engine.new_request(
            request_id=RequestId("anchor"),
            prompt_ids=tok.encode("hi"),
            options=GenerationOptions(max_new_tokens=1, mode="greedy"),
        )
        while state.remaining_prefill > 0:
            engine.prefill_chunk(state, chunk_size=engine.config.context_length)
        from baby_whale_v4.cache import DynamicKVCache

        assert isinstance(state.cache, DynamicKVCache)
        tiled = tile_cache(state.cache, 3)
        for k_orig, k_tiled in zip(state.cache.keys, tiled.keys, strict=True):
            if k_orig is None:
                self.assertIsNone(k_tiled)
            else:
                assert k_tiled is not None
                self.assertEqual(k_tiled.shape[0], 3)
                # Row 0 of the tiled cache must equal the original B=1 row.
                self.assertTrue(bool(mx.allclose(k_tiled[0], k_orig[0])))

    def test_tile_cache_rejects_b_not_one(self) -> None:
        from baby_whale_v4.cache import DynamicKVCache

        bad = DynamicKVCache(keys=[mx.zeros((2, 1, 1, 4))], values=[mx.zeros((2, 1, 1, 4))])
        with self.assertRaisesRegex(ValueError, "B=1"):
            tile_cache(bad, 3)


class TestPagedKVPool(unittest.TestCase):
    def _pool(self) -> PagedKVPool:
        cfg = PagedKVConfig(
            n_layer=2,
            n_heads=2,
            head_dim=4,
            block_size=4,
            n_blocks=8,
        )
        return PagedKVPool(cfg)

    def test_allocate_and_free_cycle(self) -> None:
        pool = self._pool()
        self.assertEqual(pool.n_free, 8)
        idx = pool.allocate()
        self.assertEqual(pool.n_free, 7)
        self.assertEqual(pool.n_allocated, 1)
        pool.free(idx)
        self.assertEqual(pool.n_free, 8)
        self.assertEqual(pool.n_allocated, 0)

    def test_allocate_exhaustion_raises(self) -> None:
        pool = self._pool()
        for _ in range(8):
            pool.allocate()
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            pool.allocate()

    def test_append_grows_into_new_blocks(self) -> None:
        pool = self._pool()
        cache = PagedKVCache(pool=pool)
        # Append 6 tokens with block_size=4 → spans 2 blocks (4 + 2 leftover).
        H, D = 2, 4
        keys = mx.ones((1, H, 6, D))
        values = mx.ones((1, H, 6, D)) * 2
        cache.append(0, keys, values)
        self.assertEqual(cache.table.n_blocks, 2)
        self.assertEqual(cache.table.length, 6)
        # gather() must return the full [1, H, 6, D] contiguous K/V.
        gk, gv = cache.gather(0)
        self.assertEqual(tuple(gk.shape), (1, H, 6, D))
        self.assertTrue(bool(mx.allclose(gk, keys)))
        self.assertTrue(bool(mx.allclose(gv, values)))

    def test_free_returns_blocks_to_pool(self) -> None:
        pool = self._pool()
        cache = PagedKVCache(pool=pool)
        cache.append(0, mx.ones((1, 2, 6, 4)), mx.ones((1, 2, 6, 4)))
        self.assertEqual(pool.n_allocated, 2)
        cache.free()
        self.assertEqual(pool.n_allocated, 0)
        self.assertEqual(cache.table.length, 0)
        self.assertEqual(cache.table.n_blocks, 0)

    def test_two_caches_share_pool_independently(self) -> None:
        pool = self._pool()
        a = PagedKVCache(pool=pool)
        b = PagedKVCache(pool=pool)
        a.append(0, mx.ones((1, 2, 5, 4)) * 1, mx.ones((1, 2, 5, 4)) * 1)
        b.append(0, mx.ones((1, 2, 5, 4)) * 9, mx.ones((1, 2, 5, 4)) * 9)
        ka, _ = a.gather(0)
        kb, _ = b.gather(0)
        # Each request sees its own data despite sharing the pool storage.
        self.assertTrue(bool(mx.allclose(ka, mx.ones_like(ka) * 1)))
        self.assertTrue(bool(mx.allclose(kb, mx.ones_like(kb) * 9)))


if __name__ == "__main__":
    unittest.main()
