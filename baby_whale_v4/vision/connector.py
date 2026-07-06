"""Vision MLP connector (DeepSeek-VL2 recipe, Step 8).

Projects encoder tile features ``[.., vision_dim]`` into the LLM embedding space
``[.., n_embd]`` with a 2-layer MLP. Built on :class:`WhaleLinear` so the
project's existing quantization placement policies apply to it too.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4.layers import WhaleLinear


class VisionMLPConnector(nn.Module):
    def __init__(self, vision_dim: int, n_embd: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if vision_dim <= 0 or n_embd <= 0:
            raise ValueError("vision_dim and n_embd must be positive")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        self.vision_dim = vision_dim
        self.n_embd = n_embd
        self.fc1 = WhaleLinear(vision_dim, n_embd, bias=True)
        self.fc2 = WhaleLinear(n_embd, n_embd, bias=True)
        self.drop = nn.Dropout(dropout)

    def __call__(self, features: mx.array) -> mx.array:
        if features.shape[-1] != self.vision_dim:
            raise ValueError(
                f"connector expected last dim {self.vision_dim}, got {features.shape[-1]}"
            )
        hidden = nn.gelu(self.fc1(features))
        hidden = self.drop(hidden)
        return self.fc2(hidden)
