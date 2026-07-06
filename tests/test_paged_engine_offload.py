"""Engine integration for paged KV storage, KV offload, and speculative decode.

These exercise the wiring that turned three previously-orphaned inference
modules into real Engine paths:

* ``paged_pool=`` makes the Engine store KV in a shared :class:`PagedKVPool`;
  decode must be token-identical to the dense :class:`DynamicKVCache` path.
* ``Engine.offload_request`` / ``reload_request`` round-trip a request's KV to
  disk and resume decode exactly where it left off.
* ``Engine.generate(mode="speculative")`` routes to ``model.spec_decode`` and
  the per-step decode loop fails fast on speculative.
"""

from __future__ import annotations

import tempfile
import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import ByteTokenizer
from baby_whale_v4.inference.engine import Engine, GenerationOptions
from baby_whale_v4.inference.paged_kv import PagedKVConfig, PagedKVPool
from baby_whale_v4.inference.prefix_cache import PrefixCache
from baby_whale_v4.typing import RequestId, array_to_int_tuple


def _model(
    *, ctx: int = 32, mtp: int = 0
) -> tuple[BabyWhaleV4Config, BabyWhaleV4Model, ByteTokenizer]:
    mx.random.seed(0)
    tok = ByteTokenizer()
    base = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=ctx).to_dict()
    cfg = BabyWhaleV4Config.from_dict({**base, "mtp_heads": mtp})
    model = BabyWhaleV4Model(cfg)
    model.eval()
    return cfg, model, tok


class TestPagedEngineParity(unittest.TestCase):
    def test_paged_greedy_matches_dense_greedy(self) -> None:
        cfg, model, tok = _model(ctx=32)
        prompt = tok.encode("hello world")
        opts = GenerationOptions(max_new_tokens=8, mode="greedy")

        dense = Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature())
        dense_out = dense.generate(prompt, opts)

        pool = PagedKVPool(PagedKVConfig.from_model_config(cfg, block_size=4, n_blocks=64))
        paged = Engine(
            model=model, config=cfg, tokenizer_hash=tok.hash_signature(), paged_pool=pool
        )
        paged_out = paged.generate(prompt, opts)

        # Paged storage reconstructs the exact same K/V, so greedy decode is
        # token-for-token identical to the dense path.
        self.assertEqual(paged_out, dense_out)
        # generate() releases the request's blocks back to the pool on finish.
        self.assertEqual(pool.n_allocated, 0)

    def test_paged_multi_request_reuses_freed_blocks(self) -> None:
        cfg, model, tok = _model(ctx=32)
        # A pool too small to hold two live requests still serves them serially
        # because generate() frees each request's blocks on completion.
        pool = PagedKVPool(PagedKVConfig.from_model_config(cfg, block_size=4, n_blocks=8))
        engine = Engine(
            model=model, config=cfg, tokenizer_hash=tok.hash_signature(), paged_pool=pool
        )
        opts = GenerationOptions(max_new_tokens=4, mode="greedy")
        for word in ("alpha", "beta", "gamma"):
            engine.generate(tok.encode(word), opts)
            self.assertEqual(pool.n_allocated, 0)

    def test_mla_plus_paged_fails_fast(self) -> None:
        mx.random.seed(0)
        tok = ByteTokenizer()
        cfg = BabyWhaleV4Config.mla_tiny(vocab_size=tok.vocab_size, context_length=24)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        pool = PagedKVPool(PagedKVConfig.from_model_config(cfg, block_size=4, n_blocks=16))
        with self.assertRaisesRegex(ValueError, "mla"):
            Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature(), paged_pool=pool)

    def test_paged_pool_dim_mismatch_fails_fast(self) -> None:
        cfg, model, tok = _model(ctx=24)
        bad = PagedKVPool(
            PagedKVConfig(
                n_layer=cfg.n_layer,
                n_heads=cfg.n_kv_head + 1,
                head_dim=cfg.head_dim,
                block_size=4,
                n_blocks=16,
            )
        )
        with self.assertRaisesRegex(ValueError, "do not match"):
            Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature(), paged_pool=bad)

    def test_engine_rejects_two_cache_strategies(self) -> None:
        cfg, model, tok = _model(ctx=24)
        pool = PagedKVPool(PagedKVConfig.from_model_config(cfg, block_size=4, n_blocks=16))
        with self.assertRaisesRegex(ValueError, "at most one"):
            Engine(
                model=model,
                config=cfg,
                tokenizer_hash=tok.hash_signature(),
                prefix_cache=PrefixCache(capacity=4),
                paged_pool=pool,
            )


class TestSpeculativeThroughEngine(unittest.TestCase):
    def test_generate_speculative_matches_spec_decode(self) -> None:
        cfg, model, tok = _model(ctx=24, mtp=2)
        engine = Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature())
        prompt = tok.encode("abc")
        out = engine.generate(prompt, GenerationOptions(max_new_tokens=6, mode="speculative"))

        prefix = mx.array([prompt], dtype=mx.int32)
        ref = model.spec_decode(prefix, max_new_tokens=6)
        ref_tail = list(array_to_int_tuple(ref.tokens[0, len(prompt) :]))

        self.assertEqual(out, ref_tail)
        self.assertLessEqual(len(out), 6)

    def test_per_step_speculative_fails_fast(self) -> None:
        cfg, model, tok = _model(ctx=24, mtp=2)
        engine = Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature())
        state = engine.new_request(
            RequestId("s"),
            tok.encode("abc"),
            GenerationOptions(max_new_tokens=3, mode="speculative"),
        )
        while state.remaining_prefill > 0:
            engine.prefill_chunk(state, chunk_size=cfg.context_length)
        with self.assertRaisesRegex(ValueError, "per-step"):
            engine.decode_step(state)


class TestKVOffloadThroughEngine(unittest.TestCase):
    def test_offload_reload_resumes_identically(self) -> None:
        cfg, model, tok = _model(ctx=32)
        engine = Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature())
        prompt = tok.encode("resume me")
        opts = GenerationOptions(max_new_tokens=8, mode="greedy")

        ref = engine.generate(prompt, opts)

        # Prefill a fresh request, snapshot its KV, then reload into a new state
        # and decode the rest — must reproduce the one-shot generate exactly.
        state = engine.new_request(RequestId("off"), prompt, opts)
        while state.remaining_prefill > 0:
            engine.prefill_chunk(state, chunk_size=cfg.context_length)
        assert state.last_logits is not None
        with tempfile.TemporaryDirectory() as tmp:
            report = engine.offload_request(state, f"{tmp}/kv.npz")
            self.assertEqual(report.sequence_length, len(prompt))
            state2 = engine.reload_request(
                RequestId("off2"),
                prompt,
                opts,
                report.path,
                prefilled=state.prefilled,
                last_logits=state.last_logits,
            )
        while not state2.finished and state2.remaining_decode > 0:
            engine.decode_step(state2)
        self.assertEqual(list(state2.generated), ref)

    def test_offload_accepts_extensionless_path(self) -> None:
        cfg, model, tok = _model(ctx=24)
        engine = Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature())
        state = engine.new_request(
            RequestId("x"), tok.encode("hi"), GenerationOptions(max_new_tokens=1)
        )
        while state.remaining_prefill > 0:
            engine.prefill_chunk(state, chunk_size=cfg.context_length)
        with tempfile.TemporaryDirectory() as tmp:
            # No ".npz" — the normalization fix must keep sha/stat/manifest in sync.
            report = engine.offload_request(state, f"{tmp}/kv")
            self.assertTrue(str(report.path).endswith(".npz"))
            self.assertTrue(report.path.exists())

    def test_offload_rejects_paged_cache(self) -> None:
        cfg, model, tok = _model(ctx=24)
        pool = PagedKVPool(PagedKVConfig.from_model_config(cfg, block_size=4, n_blocks=32))
        engine = Engine(
            model=model, config=cfg, tokenizer_hash=tok.hash_signature(), paged_pool=pool
        )
        state = engine.new_request(
            RequestId("p"), tok.encode("hi"), GenerationOptions(max_new_tokens=2)
        )
        while state.remaining_prefill > 0:
            engine.prefill_chunk(state, chunk_size=cfg.context_length)
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(TypeError, "DynamicKVCache"),
        ):
            engine.offload_request(state, f"{tmp}/x.npz")


if __name__ == "__main__":
    unittest.main()
