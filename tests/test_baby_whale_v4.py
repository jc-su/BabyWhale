import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from tests.mlx_helpers import close, finite


class TestBabyWhaleV4(unittest.TestCase):
    def test_config_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "unsupported config.precision"):
            BabyWhaleV4Config.from_dict({"precision": "fp4-native"})
        with self.assertRaisesRegex(ValueError, "unsupported config.backend"):
            BabyWhaleV4Config.from_dict({"backend": "torch"})
        with self.assertRaisesRegex(ValueError, "n_embd must be divisible"):
            BabyWhaleV4Config(n_embd=65, n_head=4)

        cfg = BabyWhaleV4Config(quant_mode="fp4-native")
        self.assertEqual(cfg.backend, "mlx")
        cfg = BabyWhaleV4Config(quant_mode="fp4-expert")
        self.assertEqual(cfg.backend, "mlx")
        with self.assertRaisesRegex(ValueError, "unsupported config.quant_mode"):
            BabyWhaleV4Config.from_dict({"quant_mode": "fp4-native-train"})
        with self.assertRaisesRegex(ValueError, "unsupported config.quant_mode"):
            BabyWhaleV4Config.from_dict({"quant_mode": "fp8-sim"})

    def test_config_from_dict_validates_payload_shape(self):
        config = BabyWhaleV4Config.from_dict({"name": "typed", "vocab_size": 64})
        self.assertEqual(config.name, "typed")
        self.assertEqual(config.vocab_size, 64)
        self.assertEqual(config.backend, "mlx")

        with self.assertRaisesRegex(TypeError, "config payload must be an object"):
            BabyWhaleV4Config.from_dict(["not", "a", "dict"])
        with self.assertRaisesRegex(ValueError, "unknown config keys"):
            BabyWhaleV4Config.from_dict({"vocab_size": 64, "extra": 1})
        with self.assertRaisesRegex(TypeError, "config.vocab_size must be an integer"):
            BabyWhaleV4Config.from_dict({"vocab_size": "64"})
        with self.assertRaisesRegex(ValueError, "unsupported config.layer_schedule entry"):
            BabyWhaleV4Config.from_dict({"layer_schedule": ["sliding_mqa", "bad"]})

    def test_forward_shape_and_loss(self):
        mx.random.seed(1337)
        config = BabyWhaleV4Config.tiny(vocab_size=64, context_length=16)
        model = BabyWhaleV4Model(config)
        idx = mx.random.randint(0, config.vocab_size, (2, 8))
        targets = mx.random.randint(0, config.vocab_size, (2, 8))

        out = model(idx, targets=targets)

        self.assertEqual(tuple(out.logits.shape), (2, 8, config.vocab_size))
        loss = out.loss
        self.assertIsNotNone(loss)
        if loss is None:
            raise AssertionError("expected training loss")
        self.assertTrue(finite(loss))

    def test_activation_checkpoint_backprops_and_rejects_cache(self):
        mx.random.seed(1337)
        base = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        config = BabyWhaleV4Config.from_dict({**base.to_dict(), "activation_checkpoint": True})
        model = BabyWhaleV4Model(config)
        idx = mx.random.randint(0, config.vocab_size, (1, 4))
        targets = mx.random.randint(0, config.vocab_size, (1, 4))

        def loss_fn(m: BabyWhaleV4Model) -> mx.array:
            out = m(idx, targets=targets)
            if out.loss is None:
                raise RuntimeError("expected training loss")
            return out.loss

        loss, grads = mx.value_and_grad(loss_fn)(model)
        mx.eval(loss, grads)
        self.assertTrue(finite(loss))
        with self.assertRaisesRegex(ValueError, "activation_checkpoint"):
            model(idx, cache=model.empty_cache())

    def test_precision_casts_model_parameters(self):
        config = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        config = BabyWhaleV4Config.from_dict({**config.to_dict(), "precision": "bf16"})
        model = BabyWhaleV4Model(config)
        self.assertEqual(model.tok_emb.weight.dtype, mx.bfloat16)
        self.assertEqual(model.lm_head.inner.weight.dtype, mx.bfloat16)
        self.assertIs(model.lm_head.inner.weight, model.tok_emb.weight)

    def test_fp4_expert_config_only_quantizes_moe_experts(self):
        base = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        config = BabyWhaleV4Config.from_dict({**base.to_dict(), "quant_mode": "fp4-expert"})
        model = BabyWhaleV4Model(config)
        block = next(iter(model.blocks.values()))

        self.assertEqual(block.attn.q_proj.quant_mode, "none")
        self.assertEqual(block.moe.router.quant_mode, "none")
        self.assertEqual(block.moe.shared_expert.w_gate.quant_mode, "fp4-native")
        self.assertEqual(block.moe.experts["expert_0"].w_up.quant_mode, "fp4-native")
        self.assertEqual(model.lm_head.quant_mode, "none")

        logits = model(mx.random.randint(0, 32, (1, 4))).logits
        self.assertTrue(finite(logits))

    def test_cache_decode_matches_full_forward(self):
        mx.random.seed(1337)
        config = BabyWhaleV4Config.tiny(vocab_size=64, context_length=16)
        model = BabyWhaleV4Model(config)
        model.eval()
        idx = mx.random.randint(0, config.vocab_size, (2, 8))

        full = model(idx).logits
        cache = model.empty_cache()
        pieces = []
        for t in range(idx.shape[1]):
            step = model(idx[:, t : t + 1], cache=cache).logits
            pieces.append(step)
        cached = mx.concatenate(pieces, axis=1)

        self.assertTrue(close(full, cached, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
