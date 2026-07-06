import unittest

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.moe import SparseMoE


def _learned_layer(model: BabyWhaleV4Model) -> SparseMoE:
    for block in model.blocks.values():
        if not block.moe.hash_routing:
            return block.moe
    raise AssertionError("expected at least one learned-routing MoE layer")


class TestAuxFreeBalancing(unittest.TestCase):
    def _model(self, *, rate: float, seed: int = 0) -> BabyWhaleV4Model:
        mx.random.seed(seed)
        base = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16).to_dict()
        cfg = BabyWhaleV4Config.from_dict({**base, "aux_free_bias_rate": rate, "n_hash_layers": 0})
        return BabyWhaleV4Model(cfg)

    def test_disabled_by_default_in_tiny_config(self):
        cfg = BabyWhaleV4Config.tiny(vocab_size=32, context_length=16)
        self.assertEqual(cfg.aux_free_bias_rate, 0.0)

    def test_zero_rate_keeps_bias_at_zero_after_training_step(self):
        model = self._model(rate=0.0, seed=1)
        model.train()
        moe = _learned_layer(model)
        idx = mx.random.randint(0, 32, (2, 8))
        targets = mx.random.randint(0, 32, (2, 8))
        out = model(idx, targets=targets)
        if out.loss is None:
            raise AssertionError("expected training loss")
        mx.eval(out.loss)
        self.assertEqual(moe.router_bias, tuple([0.0] * moe.n_expert))

    def test_positive_rate_shifts_bias_in_training_mode(self):
        model = self._model(rate=0.05, seed=2)
        model.train()
        moe = _learned_layer(model)
        idx = mx.random.randint(0, 32, (2, 8))
        targets = mx.random.randint(0, 32, (2, 8))
        out = model(idx, targets=targets)
        mx.eval(out.logits)
        bias_after = moe.router_bias
        # At least one expert was over- or under-used, so at least one bias entry
        # must have moved off zero.
        self.assertTrue(any(b != 0.0 for b in bias_after))
        # Each non-zero bias must be a multiple of the update rate.
        for b in bias_after:
            self.assertAlmostEqual(b % 0.05 if b >= 0 else (-b) % 0.05, 0.0, places=8)

    def test_eval_mode_does_not_update_bias(self):
        model = self._model(rate=0.05, seed=3)
        model.eval()
        moe = _learned_layer(model)
        idx = mx.random.randint(0, 32, (2, 8))
        out = model(idx)
        mx.eval(out.logits)
        self.assertEqual(moe.router_bias, tuple([0.0] * moe.n_expert))

    def test_bias_is_not_in_trainable_parameters(self):
        model = self._model(rate=0.05, seed=4)
        model.train()
        idx = mx.random.randint(0, 32, (1, 8))
        targets = mx.random.randint(0, 32, (1, 8))
        # Trigger one training pass so bias actually moves off zero.
        model(idx, targets=targets)
        moe = _learned_layer(model)
        self.assertTrue(any(b != 0.0 for b in moe.router_bias))
        # value_and_grad should not raise; the balancer state is not in the
        # parameter tree, so MLX never tries to take gradients of it.

        def loss_fn(m: BabyWhaleV4Model) -> mx.array:
            out = m(idx, targets=targets)
            if out.loss is None:
                raise AssertionError("expected loss")
            return out.loss

        loss, _grads = mx.value_and_grad(loss_fn)(model)
        mx.eval(loss)

    def test_bias_pushes_underused_expert_up_overused_down(self):
        # Drive the bias update directly with a handcrafted assignment tensor so
        # the directionality assertion does not depend on stochastic router output.
        model = self._model(rate=0.1, seed=5)
        model.train()
        moe = _learned_layer(model)
        # 6 tokens, all routed to expert 0; expert 0 is overused, all others
        # underused.
        indices = mx.array([[0, 0, 0, 0, 0, 0]], dtype=mx.int32)
        moe._maybe_update_bias(indices)
        bias = moe.router_bias
        self.assertAlmostEqual(bias[0], -0.1, places=8)
        for i in range(1, moe.n_expert):
            self.assertAlmostEqual(bias[i], 0.1, places=8)

    def test_balanced_assignments_leave_bias_unchanged(self):
        model = self._model(rate=0.1, seed=6)
        model.train()
        moe = _learned_layer(model)
        # Equal counts across all experts (assuming n_expert=4, tiny default).
        even = list(range(moe.n_expert)) * 3
        indices = mx.array([even], dtype=mx.int32)
        moe._maybe_update_bias(indices)
        self.assertEqual(moe.router_bias, tuple([0.0] * moe.n_expert))


if __name__ == "__main__":
    unittest.main()
