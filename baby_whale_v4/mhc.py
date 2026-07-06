from typing import Final

import mlx.core as mx
import mlx.nn as nn


def sinkhorn(logits: mx.array, n_iter: int = 3, eps: float = 1e-9) -> mx.array:
    """Iterative row+column normalization to a close-to doubly-stochastic matrix."""

    if logits.ndim < 2:
        raise ValueError("sinkhorn requires at least 2 dims (matrix on last two)")
    m = mx.exp(logits - mx.max(logits, axis=-1, keepdims=True))
    for _ in range(n_iter):
        m = m / mx.maximum(mx.sum(m, axis=-1, keepdims=True), eps)
        m = m / mx.maximum(mx.sum(m, axis=-2, keepdims=True), eps)
    return m


SUBLAYERS_PER_BLOCK: Final[int] = 2


class HyperConnect(nn.Module):
    """mHC mix for parallel residual streams."""

    def __init__(self, hc_mult: int, n_layer: int, n_sublayer: int = SUBLAYERS_PER_BLOCK):
        super().__init__()
        if hc_mult < 1:
            raise ValueError("hc_mult must be >= 1")
        self.hc_mult = hc_mult
        if hc_mult > 1:
            self.input_logits = mx.zeros((n_layer, n_sublayer, hc_mult), dtype=mx.float32)
            self.mix_logits = (
                0.02 * mx.random.normal((n_layer, n_sublayer, hc_mult, hc_mult))
            ).astype(mx.float32)
            self.write_logits = (0.02 * mx.random.normal((n_layer, n_sublayer, hc_mult))).astype(
                mx.float32
            )

    def expand(self, x: mx.array) -> mx.array:
        if self.hc_mult == 1:
            return x[:, :, None, :]
        return mx.broadcast_to(x[:, :, None, :], (*x.shape[:2], self.hc_mult, x.shape[-1]))

    def reduce(self, h: mx.array) -> mx.array:
        if self.hc_mult == 1:
            return h[:, :, 0, :]
        return mx.mean(h, axis=2)

    def consume(self, h: mx.array, layer_idx: int, sublayer_idx: int) -> mx.array:
        if self.hc_mult == 1:
            return h[:, :, 0, :]
        weights = mx.softmax(self.input_logits[layer_idx, sublayer_idx], axis=-1)
        return mx.einsum("btkd,k->btd", h, weights)

    def produce(
        self,
        h: mx.array,
        delta: mx.array,
        layer_idx: int,
        sublayer_idx: int,
    ) -> mx.array:
        if self.hc_mult == 1:
            return h + delta[:, :, None, :]
        mix = sinkhorn(self.mix_logits[layer_idx, sublayer_idx])
        h_mixed = mx.einsum("btkd,jk->btjd", h, mix)
        write = mx.softmax(self.write_logits[layer_idx, sublayer_idx], axis=-1)
        return h_mixed + delta[:, :, None, :] * write.reshape(1, 1, -1, 1)
