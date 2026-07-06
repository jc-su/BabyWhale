import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.attention import MLAAttention


class TestMLA(unittest.TestCase):
    def _model(self, kv_lora_rank: int = 16) -> BabyWhaleV4Model:
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.mla_tiny(
            vocab_size=64, context_length=32, kv_lora_rank=kv_lora_rank
        )
        model = BabyWhaleV4Model(cfg)
        model.eval()
        return model

    def test_attention_kind_includes_mla(self):
        cfg = BabyWhaleV4Config.mla_tiny(vocab_size=64, context_length=32)
        self.assertIn("mla", cfg.effective_layer_schedule)

    def test_forward_shape(self):
        model = self._model()
        idx = mx.random.randint(0, model.config.vocab_size, (2, 16))
        out = model(idx)
        self.assertEqual(tuple(out.logits.shape), (2, 16, model.config.vocab_size))

    def test_cache_decode_matches_full_forward(self):
        model = self._model()
        mx.random.seed(7)
        idx = mx.random.randint(0, model.config.vocab_size, (1, 12))
        full = model(idx).logits
        cache = model.empty_cache()
        pieces = []
        for t in range(idx.shape[1]):
            pieces.append(model(idx[:, t : t + 1], cache=cache).logits)
        cached = mx.concatenate(pieces, axis=1)
        max_diff = float(mx.max(mx.abs(full - cached)))
        self.assertLess(max_diff, 1e-4)

    def test_latent_cache_smaller_than_raw_kv(self):
        # MLA caches c_kv with shape [B, T, R]. A sliding-MQA cache stores
        # K and V each with shape [B, n_kv_head, T, head_dim]. Educational
        # comparison: per-token MLA storage = R, MQA per-token storage =
        # 2 * n_kv_head * head_dim. With R < 2*n_kv*head_dim the latent wins.
        cfg = BabyWhaleV4Config.mla_tiny(vocab_size=64, context_length=32, kv_lora_rank=8)
        per_token_mla = cfg.kv_lora_rank
        per_token_mqa = 2 * cfg.n_kv_head * cfg.head_dim
        self.assertLess(per_token_mla, per_token_mqa)

    def test_mla_layer_uses_latent_cache_slot(self):
        model = self._model()
        idx = mx.random.randint(0, model.config.vocab_size, (1, 8))
        cache = model.empty_cache()
        model(idx, cache=cache)
        # Walk blocks to find the MLA layer index.
        mla_layer_idx = None
        for i, block in enumerate(model.blocks.values()):
            if isinstance(block.attn, MLAAttention):
                mla_layer_idx = i
                break
        self.assertIsNotNone(mla_layer_idx)
        if mla_layer_idx is None:
            raise AssertionError("expected at least one MLA layer")
        # MLA layer's KV slot is empty; latent slot has length T.
        self.assertIsNone(cache.keys[mla_layer_idx])
        self.assertEqual(cache.latent_length(mla_layer_idx), 8)

    def test_config_rejects_kv_lora_rank_above_n_embd(self):
        with self.assertRaisesRegex(ValueError, "kv_lora_rank cannot exceed"):
            BabyWhaleV4Config.mla_tiny(vocab_size=64, context_length=32, kv_lora_rank=999)


if __name__ == "__main__":
    unittest.main()
