"""Synthetic long-context needle-retrieval eval.

Places a ``marker`` token followed by a *per-sample random* ``answer`` at a
random position in a sequence of filler tokens, then repeats the ``marker`` at
the end as a query. The model must predict the answer at the final position — it
can only do so by attending back to the marked pair, so accuracy measures
long-range retrieval.

Unlike :class:`SyntheticNeedleDataset` (which uses a *fixed* answer id and so
only tests bigram learning), the answer varies per sample here, so a model
cannot memorize a single mapping — it must actually retrieve. This is the probe
for whether HCA/CSA reach beyond a sliding window: sliding-only attention fails
once the needle sits farther back than its window; the compressed/global paths
still reach it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from baby_whale_v4.model import BabyWhaleV4Model


@dataclass(frozen=True)
class NeedleResult:
    accuracy: float
    n_samples: int
    seq_len: int


def _build_needle_batch(
    n_samples: int, seq_len: int, vocab_size: int, *, marker_id: int, seed: int
) -> tuple[mx.array, mx.array]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if seq_len < 8:
        raise ValueError("seq_len must be >= 8")
    if marker_id < 0:
        raise ValueError("marker_id must be non-negative")
    low = marker_id + 2  # keep fillers/answers clear of the marker id
    if vocab_size < low + 2:
        raise ValueError("vocab_size too small for the needle eval")
    rng = np.random.default_rng(seed)
    x = rng.integers(low, vocab_size, size=(n_samples, seq_len), dtype=np.int32)
    answers = rng.integers(low, vocab_size, size=(n_samples,), dtype=np.int32)
    for i in range(n_samples):
        pos = int(rng.integers(1, seq_len - 3))
        x[i, pos] = marker_id
        x[i, pos + 1] = int(answers[i])
        x[i, seq_len - 1] = marker_id  # query marker at the end
    return mx.array(x), mx.array(answers)


def needle_accuracy_from_logits(final_logits: mx.array, answers: mx.array) -> float:
    """Fraction of rows whose argmax prediction equals the expected answer."""
    if final_logits.ndim != 2:
        raise ValueError("final_logits must be [N, vocab]")
    preds = mx.argmax(final_logits, axis=-1)
    return float(mx.mean(mx.equal(preds, answers).astype(mx.float32)))


def evaluate_needle_retrieval(
    model: BabyWhaleV4Model,
    *,
    vocab_size: int,
    n_samples: int = 64,
    seq_len: int = 64,
    marker_id: int = 1,
    seed: int = 0,
) -> NeedleResult:
    """Measure retrieval accuracy of ``model`` on the synthetic needle task."""
    x, answers = _build_needle_batch(n_samples, seq_len, vocab_size, marker_id=marker_id, seed=seed)
    out = model(x)
    accuracy = needle_accuracy_from_logits(out.logits[:, -1, :], answers)
    return NeedleResult(accuracy=accuracy, n_samples=n_samples, seq_len=seq_len)
