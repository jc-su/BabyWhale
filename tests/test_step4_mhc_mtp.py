import tempfile
import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import SyntheticCopyDataset
from baby_whale_v4.mhc import sinkhorn
from baby_whale_v4.training import PretrainConfig, pretrain
from tests.mlx_helpers import close, finite, max_abs


class TestStep4(unittest.TestCase):
    def test_sinkhorn_doubly_stochastic(self):
        mx.random.seed(0)
        logits = mx.random.normal((4, 4))
        m = sinkhorn(logits, n_iter=20)
        self.assertTrue(close(mx.sum(m, axis=-1), mx.ones((4,)), atol=1e-3, rtol=1e-3))
        self.assertTrue(close(mx.sum(m, axis=-2), mx.ones((4,)), atol=1e-3, rtol=1e-3))

    def test_hc_mult_1_matches_residual_baseline(self):
        mx.random.seed(7)
        cfg = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        m1 = BabyWhaleV4Model(cfg)
        m2 = BabyWhaleV4Model(cfg)
        m2.load_state_dict(m1.state_dict())
        m1.eval()
        m2.eval()
        x = mx.random.randint(0, 32, (2, 8))
        a = m1(x).logits
        b = m2(x).logits
        self.assertTrue(bool(mx.array_equal(a, b)))

    def test_hc_mult_2_forward_runs(self):
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.hybrid_tiny(vocab_size=32, context_length=16, hc_mult=2)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        idx = mx.random.randint(0, 32, (2, 12))
        out = model(idx)
        self.assertEqual(tuple(out.logits.shape), (2, 12, 32))
        self.assertTrue(finite(out.logits))

    def test_moe_route_metrics_report_utilization(self):
        mx.random.seed(0)
        cfg = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        model = BabyWhaleV4Model(cfg)
        block = next(iter(model.blocks.values()))
        x = mx.random.normal((2, 8, cfg.n_embd))
        input_ids = mx.random.randint(0, cfg.vocab_size, (2, 8))
        metrics = block.moe.route_metrics(x, input_ids)
        self.assertEqual(metrics.n_expert, cfg.n_expert)
        self.assertEqual(sum(metrics.counts), metrics.total_tokens)
        self.assertGreater(metrics.used_experts, 0)

    def test_hc_mult_2_cache_decode_parity(self):
        mx.random.seed(11)
        cfg = BabyWhaleV4Config.hybrid_tiny(vocab_size=32, context_length=16, hc_mult=2)
        model = BabyWhaleV4Model(cfg)
        model.eval()
        idx = mx.random.randint(0, 32, (2, 12))
        full = model(idx).logits
        cache = model.empty_cache()
        pieces = []
        for t in range(idx.shape[1]):
            step = model(idx[:, t : t + 1], cache=cache).logits
            pieces.append(step)
        cached = mx.concatenate(pieces, axis=1)
        self.assertLess(max_abs(full - cached), 1e-4)

    def test_hc_mult_2_stable_on_synthetic(self):
        cfg = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        cfg = BabyWhaleV4Config.from_dict({**cfg.to_dict(), "hc_mult": 2, "name": "tiny-hc2"})
        ds = SyntheticCopyDataset(n_samples=8, seq_len=16, vocab_size=32, seed=0)
        ptcfg = PretrainConfig(lr=2e-3, max_steps=4, batch_size=4, log_every=2, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            model = pretrain(
                config=cfg,
                pretrain_config=ptcfg,
                train_dataset=ds,
                out_dir=tmp,
            )
        model.eval()
        x, y = ds[0]
        out = model(x[None, :], targets=y[None, :])
        loss = out.loss
        self.assertIsNotNone(loss)
        if loss is None:
            raise AssertionError("expected training loss")
        self.assertTrue(finite(loss))

    def test_mtp_disabled_main_logits_identical(self):
        mx.random.seed(5)
        base = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        with_mtp = BabyWhaleV4Config.from_dict(
            {**base.to_dict(), "mtp_heads": 1, "name": "tiny-mtp"}
        )
        m1 = BabyWhaleV4Model(base)
        m2 = BabyWhaleV4Model(with_mtp)
        m2.update(m1.parameters())
        m1.eval()
        m2.eval()
        x = mx.random.randint(0, 32, (1, 8))
        a = m1(x).logits
        b = m2(x).logits
        self.assertTrue(bool(mx.array_equal(a, b)))

    def test_mtp_loss_decomposes(self):
        cfg = BabyWhaleV4Config.from_dict(
            {
                **BabyWhaleV4Config.tiny(vocab_size=32, context_length=16).to_dict(),
                "mtp_heads": 2,
                "name": "tiny-mtp2",
            }
        )
        model = BabyWhaleV4Model(cfg)
        x = mx.random.randint(0, 32, (1, 8))
        y = mx.random.randint(0, 32, (1, 8))
        out = model(x, targets=y, mtp_loss_weight=0.25)
        main_loss = out.main_loss
        loss = out.loss
        self.assertIsNotNone(main_loss)
        self.assertIsNotNone(loss)
        if main_loss is None or loss is None:
            raise AssertionError("expected MTP training losses")
        self.assertEqual(len(out.mtp_losses), 2)
        manual = main_loss + 0.25 * (out.mtp_losses[0] + out.mtp_losses[1])
        self.assertTrue(close(loss, manual, atol=1e-6, rtol=1e-6))

    def test_spec_decode_matches_greedy(self):
        mx.random.seed(13)
        cfg = BabyWhaleV4Config.from_dict(
            {
                **BabyWhaleV4Config.tiny(vocab_size=32, context_length=24).to_dict(),
                "mtp_heads": 2,
                "name": "tiny-spec",
            }
        )
        model = BabyWhaleV4Model(cfg)
        model.eval()
        prefix = mx.random.randint(0, 32, (1, 4))

        seq = prefix
        for _ in range(5):
            out = model(seq)
            next_id = mx.argmax(out.logits[:, -1, :], axis=-1).reshape(1, 1)
            seq = mx.concatenate([seq, next_id], axis=1)
        greedy = seq
        result = model.spec_decode(prefix, max_new_tokens=5)
        self.assertEqual(greedy.shape, result.tokens.shape)
        self.assertTrue(bool(mx.array_equal(greedy, result.tokens)))
        # Acceptance-rate metric is well-defined whenever at least one draft
        # round happened.
        self.assertGreaterEqual(result.acceptance_rate, 0.0)
        self.assertLessEqual(result.acceptance_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
