"""Long-context needle-retrieval eval: scoring, range, and sensitivity to training."""

from __future__ import annotations

import unittest

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.eval import evaluate_needle_retrieval, needle_accuracy_from_logits
from baby_whale_v4.eval.needle import _build_needle_batch
from baby_whale_v4.training.mlx_optim import AdamW


def _tiny(vocab: int, ctx: int) -> BabyWhaleV4Model:
    mx.random.seed(0)
    model = BabyWhaleV4Model(BabyWhaleV4Config.tiny(vocab_size=vocab, context_length=ctx))
    model.eval()
    return model


def _train_needle(
    model: BabyWhaleV4Model, *, vocab: int, seq_len: int, seed: int, steps: int
) -> None:
    x, answers = _build_needle_batch(24, seq_len, vocab, marker_id=1, seed=seed)
    optimizer = AdamW(learning_rate=3e-3)

    def loss_fn(m: BabyWhaleV4Model) -> mx.array:
        logits = m(x).logits[:, -1, :]
        return mx.mean(nn.losses.cross_entropy(logits, answers))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        _loss, grads = loss_and_grad(model)
        model.update(optimizer.step(model.parameters(), grads))
        mx.eval(model.parameters())


class TestNeedleEval(unittest.TestCase):
    def test_scoring_is_correct(self) -> None:
        answers = mx.array([2, 5, 7])
        cols = mx.arange(10)[None, :]
        perfect = mx.where(cols == answers[:, None], 10.0, -10.0)
        self.assertEqual(needle_accuracy_from_logits(perfect, answers), 1.0)
        wrong = mx.where(cols == (answers + 1)[:, None], 10.0, -10.0)
        self.assertEqual(needle_accuracy_from_logits(wrong, answers), 0.0)

    def test_returns_fraction_on_tiny_model(self) -> None:
        model = _tiny(vocab=32, ctx=64)
        result = evaluate_needle_retrieval(model, vocab_size=32, n_samples=16, seq_len=32)
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)
        self.assertEqual(result.n_samples, 16)

    def test_training_improves_retrieval(self) -> None:
        # The eval must be sensitive to model quality: a model trained on the task
        # retrieves far better than the untrained baseline (which is ~chance).
        model = _tiny(vocab=24, ctx=32)
        before = evaluate_needle_retrieval(
            model, vocab_size=24, n_samples=24, seq_len=24, seed=1
        ).accuracy
        _train_needle(model, vocab=24, seq_len=24, seed=1, steps=200)
        after = evaluate_needle_retrieval(
            model, vocab_size=24, n_samples=24, seq_len=24, seed=1
        ).accuracy
        self.assertGreater(after, before)
        self.assertGreater(after, 0.5)


if __name__ == "__main__":
    unittest.main()
