from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4.mlx_fp4 import (
    MLXFP4Mode,
    MLXFP4Weight,
    linear_mlx_fp4,
    quantize_weight_mlx_fp4,
)
from baby_whale_v4.quantization.policy import quant_mode_for_placement
from baby_whale_v4.typing import LinearPlacement, QuantMode, ResolvedQuantMode, assert_never

_AFFINE_GROUP_SIZE = 64
_AFFINE_BITS = {"int8-weight": 8, "int4-weight": 4}


@dataclass(frozen=True)
class _AffinePackedWeight:
    packed: mx.array
    scales: mx.array
    biases: mx.array
    bits: int
    group_size: int


class WhaleLinear(nn.Module):
    """MLX linear layer with real quantized-matmul paths for affine int4/int8 and FP4."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        *,
        bias: bool = True,
        quant_mode: QuantMode = "none",
        fp4_mode: MLXFP4Mode = "nvfp4",
        placement: LinearPlacement = "general",
    ):
        super().__init__()
        if input_dims <= 0 or output_dims <= 0:
            raise ValueError("WhaleLinear dimensions must be positive")
        if bias is not True and bias is not False:
            raise TypeError("WhaleLinear bias flag must be a boolean")
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.inner = nn.Linear(input_dims, output_dims, bias=bias)
        self.placement = placement
        self.quant_mode: ResolvedQuantMode
        self.quant_mode = quant_mode_for_placement(quant_mode, placement)
        self.fp4_mode = fp4_mode
        self._fp4_cached_weight: MLXFP4Weight | None = None
        self._fp4_cached_key: tuple[int, tuple[int, ...], str, MLXFP4Mode] | None = None
        self._affine_cached_weight: _AffinePackedWeight | None = None
        self._affine_cached_key: tuple[int, tuple[int, ...], str, int] | None = None
        self.lora_a: mx.array | None = None
        self.lora_b: mx.array | None = None
        self.lora_alpha: float = 1.0

    @property
    def weight(self) -> mx.array:
        return self.inner.weight

    @property
    def bias(self) -> mx.array | None:
        return self.inner.bias if hasattr(self.inner, "bias") else None

    def __call__(self, x: mx.array) -> mx.array:
        match self.quant_mode:
            case "none":
                y = self.inner(x)
            case "int8-weight" | "int4-weight":
                y = self._affine_matmul(x, _AFFINE_BITS[self.quant_mode])
            case "fp4-native":
                packed = self._packed_fp4_weight()
                y = linear_mlx_fp4(x, packed, self.bias, dtype=x.dtype)
            case _:
                assert_never(self.quant_mode)
        if self.lora_a is None or self.lora_b is None:
            return y
        delta = (x @ self.lora_a.T) @ self.lora_b.T
        return y + delta.astype(y.dtype) * (self.lora_alpha / self.lora_a.shape[0])

    def enable_lora(self, *, rank: int, alpha: float = 1.0, scale: float = 0.01) -> None:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if rank > min(self.input_dims, self.output_dims):
            raise ValueError("LoRA rank cannot exceed the smaller linear dimension")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if scale <= 0:
            raise ValueError("LoRA init scale must be positive")
        self.lora_a = mx.random.normal((rank, self.input_dims), dtype=self.weight.dtype) * scale
        self.lora_b = mx.zeros((self.output_dims, rank), dtype=self.weight.dtype)
        self.lora_alpha = alpha

    def clear_quant_cache(self) -> None:
        self._fp4_cached_weight = None
        self._fp4_cached_key = None
        self._affine_cached_weight = None
        self._affine_cached_key = None

    def _packed_fp4_weight(self) -> MLXFP4Weight:
        weight = self.inner.weight
        key = (id(weight), tuple(int(v) for v in weight.shape), str(weight.dtype), self.fp4_mode)
        if self._fp4_cached_key == key and self._fp4_cached_weight is not None:
            return self._fp4_cached_weight
        packed = quantize_weight_mlx_fp4(weight, self.fp4_mode)
        self._fp4_cached_key = key
        self._fp4_cached_weight = packed
        return packed

    def _packed_affine_weight(self, bits: int) -> _AffinePackedWeight:
        weight = self.inner.weight
        if weight.shape[-1] % _AFFINE_GROUP_SIZE != 0:
            raise ValueError(
                f"WhaleLinear with quant_mode={self.quant_mode!r} requires input_dims "
                f"divisible by {_AFFINE_GROUP_SIZE}; got input_dims={weight.shape[-1]}"
            )
        key = (id(weight), tuple(int(v) for v in weight.shape), str(weight.dtype), bits)
        if self._affine_cached_key == key and self._affine_cached_weight is not None:
            return self._affine_cached_weight
        packed, scales, biases = mx.quantize(
            weight, group_size=_AFFINE_GROUP_SIZE, bits=bits, mode="affine"
        )
        cached = _AffinePackedWeight(
            packed=packed,
            scales=scales,
            biases=biases,
            bits=bits,
            group_size=_AFFINE_GROUP_SIZE,
        )
        self._affine_cached_key = key
        self._affine_cached_weight = cached
        return cached

    def _affine_matmul(self, x: mx.array, bits: int) -> mx.array:
        packed = self._packed_affine_weight(bits)
        original_shape = tuple(int(v) for v in x.shape)
        x2 = x.reshape(-1, original_shape[-1])
        y = mx.quantized_matmul(
            x2,
            packed.packed,
            packed.scales,
            packed.biases,
            transpose=True,
            group_size=packed.group_size,
            bits=bits,
            mode="affine",
        ).astype(x.dtype)
        y = y.reshape(*original_shape[:-1], self.output_dims)
        b = self.bias
        if b is not None:
            y = y + b.astype(y.dtype)
        return y


class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((n_embd,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        scale = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return self.weight * x * scale


def rotate_half(x: mx.array) -> mx.array:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return mx.stack([-x_odd, x_even], axis=-1).reshape(*x.shape[:-1], x.shape[-1])


class PartialRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rotary_dim: int, base: float = 1000000.0):
        super().__init__()
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be positive, even, and <= head_dim")
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        power = mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim
        self.inv_freq = 1.0 / mx.power(base, power)

    def _cos_sin(self, positions: mx.array) -> tuple[mx.array, mx.array]:
        # 1D [T] positions -> [1, 1, T, D] (shared across batch, the training and
        # normal-decode path). 2D [B, T] per-row positions -> [B, 1, T, D], used by
        # ragged batched decode where each row's new token sits at its own position.
        if positions.ndim == 1:
            freqs = mx.einsum("t,d->td", positions.astype(mx.float32), self.inv_freq)
            emb = mx.concatenate([freqs, freqs], axis=-1)
            return mx.cos(emb)[None, None, :, :], mx.sin(emb)[None, None, :, :]
        if positions.ndim == 2:
            freqs = mx.einsum("bt,d->btd", positions.astype(mx.float32), self.inv_freq)
            emb = mx.concatenate([freqs, freqs], axis=-1)
            return mx.cos(emb)[:, None, :, :], mx.sin(emb)[:, None, :, :]
        raise ValueError("positions must be 1D [T] or 2D [B, T]")

    def __call__(self, q: mx.array, k: mx.array, positions: mx.array) -> tuple[mx.array, mx.array]:
        cos, sin = self._cos_sin(positions)
        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
        q = mx.concatenate([q_rot * cos + rotate_half(q_rot) * sin, q_pass], axis=-1)
        k = mx.concatenate([k_rot * cos + rotate_half(k_rot) * sin, k_pass], axis=-1)
        return q, k

    def rotate_one(self, x: mx.array, positions: mx.array) -> mx.array:
        """Rotate a single tensor at the given positions. Used by MLA where
        K is reconstructed from the latent cache and rotated separately from Q.
        """

        cos, sin = self._cos_sin(positions)
        x_rot, x_pass = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        x_rot = x_rot * cos + rotate_half(x_rot) * sin
        return mx.concatenate([x_rot, x_pass], axis=-1)


class SwiGLUExpert(nn.Module):
    def __init__(
        self,
        n_embd: int,
        hidden_size: int,
        clamp: float = 30.0,
        quant_mode: QuantMode = "none",
        placement: LinearPlacement = "moe_expert",
    ):
        super().__init__()
        self.w_gate = WhaleLinear(
            n_embd, hidden_size, bias=False, quant_mode=quant_mode, placement=placement
        )
        self.w_up = WhaleLinear(
            n_embd, hidden_size, bias=False, quant_mode=quant_mode, placement=placement
        )
        self.w_down = WhaleLinear(
            hidden_size, n_embd, bias=False, quant_mode=quant_mode, placement=placement
        )
        self.clamp = clamp

    def __call__(self, x: mx.array) -> mx.array:
        gate = mx.clip(self.w_gate(x), -self.clamp, self.clamp)
        up = mx.clip(self.w_up(x), -self.clamp, self.clamp)
        return self.w_down(nn.silu(gate) * up)
