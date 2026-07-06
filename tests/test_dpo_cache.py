"""DPO reference-log-ratio caching is numerically identical to recomputing it."""

from __future__ import annotations

import tempfile
import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.training.dpo import (
    DPOConfig,
    DPOExample,
    _dpo_pair_loss,
    _precompute_ref_logratios,
    dpo,
    dpo_loss,
    make_reference,
)


def _tiny():
    mx.random.seed(0)
    cfg = BabyWhaleV4Config.tiny(vocab_size=64, context_length=32)
    model = BabyWhaleV4Model(cfg)
    model.eval()
    return model


class TestDPOCache(unittest.TestCase):
    def test_cached_loss_matches_recompute(self) -> None:
        model = _tiny()
        ref = make_reference(model)
        ex = DPOExample(
            prompt=mx.array([1, 2, 3], dtype=mx.int32),
            chosen=mx.array([4, 5], dtype=mx.int32),
            rejected=mx.array([6, 7], dtype=mx.int32),
        )
        direct = dpo_loss(
            model, ref, ex.prompt[None, :], ex.chosen[None, :], ex.rejected[None, :], beta=0.1
        )
        ratios = _precompute_ref_logratios(ref, [ex])
        cached = _dpo_pair_loss(
            model,
            ex.prompt[None, :],
            ex.chosen[None, :],
            ex.rejected[None, :],
            beta=0.1,
            ref_logratio=ratios[0],
        )
        self.assertTrue(bool(mx.allclose(direct, cached)))

    def test_dpo_still_trains(self) -> None:
        model = _tiny()
        examples = [
            DPOExample(
                prompt=mx.array([1, 2, 3], dtype=mx.int32),
                chosen=mx.array([4, 5], dtype=mx.int32),
                rejected=mx.array([6, 7], dtype=mx.int32),
            ),
            DPOExample(
                prompt=mx.array([1, 2], dtype=mx.int32),
                chosen=mx.array([3, 4], dtype=mx.int32),
                rejected=mx.array([5, 6], dtype=mx.int32),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = dpo(
                model=model,
                examples=examples,
                dpo_config=DPOConfig(max_steps=5, batch_size=2, lr=1e-3),
                out_dir=tmp,
            )
        self.assertEqual(out.config.vocab_size, 64)


if __name__ == "__main__":
    unittest.main()
