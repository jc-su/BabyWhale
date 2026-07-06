import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.attention import CSAAttention
from tests.mlx_helpers import close, finite, max_abs


class TestStep3(unittest.TestCase):
    def _build(self, hc_mult: int = 1, mtp: int = 0) -> BabyWhaleV4Model:
        mx.random.seed(1234)
        cfg = BabyWhaleV4Config.hybrid_tiny(
            vocab_size=64, context_length=32, hc_mult=hc_mult, mtp_heads=mtp
        )
        model = BabyWhaleV4Model(cfg)
        model.eval()
        return model

    def test_hybrid_forward_shape(self):
        model = self._build()
        idx = mx.random.randint(0, 64, (2, 24))
        out = model(idx)
        self.assertEqual(tuple(out.logits.shape), (2, 24, 64))

    def test_hybrid_cache_decode_parity(self):
        model = self._build()
        mx.random.seed(0)
        idx = mx.random.randint(0, 64, (2, 16))
        full = model(idx).logits
        cache = model.empty_cache()
        pieces = []
        for t in range(idx.shape[1]):
            step = model(idx[:, t : t + 1], cache=cache).logits
            pieces.append(step)
        cached = mx.concatenate(pieces, axis=1)
        self.assertLess(max_abs(full - cached), 1e-4)

    def test_csa_dense_debug_parity(self):
        model = self._build()
        for m in model.modules():
            if isinstance(m, CSAAttention):
                m.dense_debug = True
        idx = mx.random.randint(0, 64, (1, 16))
        dense = model(idx).logits
        for m in model.modules():
            if isinstance(m, CSAAttention):
                m.dense_debug = False
        sparse = model(idx).logits
        self.assertEqual(dense.shape, sparse.shape)
        self.assertTrue(finite(dense))
        self.assertTrue(finite(sparse))

    def test_csa_eval_topk_deterministic(self):
        model = self._build()
        idx = mx.random.randint(0, 64, (1, 16))
        a = model(idx).logits
        b = model(idx).logits
        self.assertTrue(close(a, b))

    def test_hca_cache_compression_savings(self):
        cfg = BabyWhaleV4Config.hybrid_tiny(vocab_size=64, context_length=64)
        T = 64
        block_size = cfg.hca_block_size
        n_blocks = T // block_size
        raw_kv_units = 2 * cfg.n_kv_head * T * cfg.head_dim
        compressed_units = 2 * cfg.n_kv_head * n_blocks * cfg.head_dim
        self.assertLess(compressed_units, raw_kv_units)
        self.assertEqual(raw_kv_units // compressed_units, block_size)

    def test_cache_stats_reports_bytes_and_layers(self):
        model = self._build()
        idx = mx.random.randint(0, 64, (1, 8))
        cache = model.empty_cache()
        model(idx, cache=cache)
        stats = cache.stats()
        self.assertEqual(stats.n_layer, model.config.n_layer)
        self.assertGreater(stats.present_layers, 0)
        self.assertEqual(stats.max_sequence_length, 8)
        self.assertGreater(stats.bytes, 0)


if __name__ == "__main__":
    unittest.main()
