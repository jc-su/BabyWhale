"""The 5-minute journey, in-process: build a tiny model, teach it to read, generate.

This is the whole lifecycle in miniature and in one function, so Module 00 can
show the arc before any detail. At scale you'd instead run the CLI
(``baby-whale-v4 pretrain ...`` then ``... generate ...``); this does the same
thing small enough to finish in seconds and be tested.
"""

from __future__ import annotations

from typing import TypedDict

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.data import ByteTokenizer
from baby_whale_v4.inference.engine import Engine, GenerationOptions
from baby_whale_v4.training.mlx_optim import AdamW


class JourneyResult(TypedDict):
    loss_first: float
    loss_last: float
    sample: str


CORPUS = "the baby whale learns to read the deep blue sea. " * 200


def _windows(ids: list[int], ctx: int, n: int) -> tuple[mx.array, mx.array]:
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    span = max(1, len(ids) - ctx - 1)
    for i in range(n):
        start = (i * (ctx // 2)) % span
        chunk = ids[start : start + ctx + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return mx.array(xs), mx.array(ys)


def run_journey(steps: int = 120, seed: int = 0) -> JourneyResult:
    """Train a tiny model on a toy corpus and generate. Returns before/after loss
    and a sample so callers (and the test suite) can see it actually learned."""
    mx.random.seed(seed)
    tok = ByteTokenizer()
    cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=48)
    model = BabyWhaleV4Model(cfg)

    x, y = _windows(tok.encode(CORPUS), cfg.context_length, n=16)

    def loss_fn(m: BabyWhaleV4Model) -> mx.array:
        loss = m(x, targets=y).loss
        assert loss is not None  # targets are always given here, so loss is present
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = AdamW(learning_rate=3e-3)
    first = last = 0.0
    for step in range(steps):
        loss, grads = loss_and_grad(model)
        model.update(optimizer.step(model.parameters(), grads))
        mx.eval(model.parameters())
        last = float(loss)
        if step == 0:
            first = last

    model.eval()
    engine = Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature())
    out = engine.generate(tok.encode("the baby whale"), GenerationOptions(max_new_tokens=24))
    return {"loss_first": first, "loss_last": last, "sample": tok.decode(out)}
