import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4.layers import WhaleLinear
from baby_whale_v4.typing import QuantMode


class MTPHead(nn.Module):
    """One MTP head predicts a future-shifted token from the same hidden state."""

    def __init__(self, n_embd: int, vocab_size: int, quant_mode: QuantMode = "none"):
        super().__init__()
        self.transform = WhaleLinear(
            n_embd, n_embd, bias=False, quant_mode=quant_mode, placement="mtp"
        )
        self.head = WhaleLinear(
            n_embd, vocab_size, bias=False, quant_mode=quant_mode, placement="mtp"
        )

    def __call__(self, h: mx.array) -> mx.array:
        return self.head(nn.silu(self.transform(h)))
