"""Milestones — prove the accumulated system works on a REAL task (TinyTorch's idea).

TinyTorch validates learning with historical breakthroughs (solve XOR, classify
CIFAR), not isolated unit tests: if the real task works, every piece underneath
composed correctly. These are our version — each milestone actually trains/runs
the model and checks a task metric you can't fake by pattern-matching a reference.

The fast ones (``it_learns``, ``it_remembers``) are green-gated in the test suite.
The heavier ones (follows / reasons / serves) are documented in ``MILESTONES.md``
with the CLI command that proves them, since they need real SFT / RL / serving runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.eval.needle import _build_needle_batch, evaluate_needle_retrieval
from baby_whale_v4.training.mlx_optim import AdamW
from course.journey import run_journey


@dataclass(frozen=True)
class MilestoneResult:
    name: str
    passed: bool
    evidence: str


def it_learns(steps: int = 60) -> MilestoneResult:
    """After the backbone + pre-training legs: the model actually reduces loss."""
    result = run_journey(steps=steps)
    first, last = float(result["loss_first"]), float(result["loss_last"])
    return MilestoneResult(
        "it-learns (backbone + pre-training)",
        passed=last < first - 1.0,
        evidence=f"toy-corpus loss {first:.2f} -> {last:.2f}",
    )


def it_remembers(steps: int = 200, seed: int = 1) -> MilestoneResult:
    """After the mid-training leg: the model can retrieve a fact from far back."""
    mx.random.seed(seed)
    model = BabyWhaleV4Model(BabyWhaleV4Config.tiny(vocab_size=24, context_length=32))
    x, answers = _build_needle_batch(24, 24, 24, marker_id=1, seed=seed)

    def loss_fn(m: BabyWhaleV4Model) -> mx.array:
        return mx.mean(nn.losses.cross_entropy(m(x).logits[:, -1, :], answers))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = AdamW(learning_rate=3e-3)
    for _ in range(steps):
        _loss, grads = loss_and_grad(model)
        model.update(optimizer.step(model.parameters(), grads))
        mx.eval(model.parameters())

    model.eval()
    acc = evaluate_needle_retrieval(
        model, vocab_size=24, n_samples=24, seq_len=24, seed=seed
    ).accuracy
    return MilestoneResult(
        "it-remembers (long-context retrieval)",
        passed=acc > 0.5,
        evidence=f"needle retrieval accuracy = {acc:.2f} (chance ~0.05)",
    )


FAST_MILESTONES = (it_learns, it_remembers)
