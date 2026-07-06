"""Vision model integration: connector-projected tiles prepend to the token stream."""

from __future__ import annotations

import dataclasses
import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model


def _model(enable_vision: bool, *, vision_dim: int = 16) -> BabyWhaleV4Model:
    mx.random.seed(0)
    cfg = dataclasses.replace(
        BabyWhaleV4Config.tiny(vocab_size=32, context_length=64),
        enable_vision=enable_vision,
        vision_dim=vision_dim,
    )
    model = BabyWhaleV4Model(cfg)
    model.eval()
    return model


class TestVisionIntegration(unittest.TestCase):
    def test_no_connector_when_vision_disabled(self) -> None:
        model = _model(False)
        self.assertIsNone(model.vision_connector)
        out = model(mx.array([[1, 2, 3, 4]]))
        self.assertEqual(tuple(out.logits.shape), (1, 4, 32))

    def test_image_features_prepend_tokens(self) -> None:
        model = _model(True, vision_dim=16)
        ids = mx.array([[1, 2, 3, 4]])
        feats = mx.random.normal((1, 5, 16))  # 5 image tokens
        out = model(ids, image_features=feats)
        self.assertEqual(tuple(out.logits.shape), (1, 5 + 4, 32))  # image tokens prepended

    def test_none_image_features_is_noop(self) -> None:
        model = _model(True, vision_dim=16)
        ids = mx.array([[1, 2, 3, 4]])
        a = model(ids).logits
        b = model(ids, image_features=None).logits
        self.assertTrue(bool(mx.allclose(a, b)))
        self.assertEqual(tuple(a.shape), (1, 4, 32))  # no vision tokens

    def test_features_without_enable_vision_raises(self) -> None:
        model = _model(False)
        with self.assertRaisesRegex(ValueError, "enable_vision"):
            model(mx.array([[1, 2, 3]]), image_features=mx.random.normal((1, 2, 16)))

    def test_vision_with_targets_raises(self) -> None:
        model = _model(True, vision_dim=16)
        ids = mx.array([[1, 2, 3, 4]])
        with self.assertRaisesRegex(ValueError, "inference-only"):
            model(ids, targets=ids, image_features=mx.random.normal((1, 2, 16)))

    def test_wrong_feature_dim_raises(self) -> None:
        model = _model(True, vision_dim=16)
        with self.assertRaisesRegex(ValueError, "vision_dim|n_image_tokens"):
            model(mx.array([[1, 2, 3]]), image_features=mx.random.normal((1, 2, 99)))


if __name__ == "__main__":
    unittest.main()
