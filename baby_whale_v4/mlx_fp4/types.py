from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

type MLXFP4Mode = Literal["mxfp4", "nvfp4"]
type MLXFP4WeightGrad = Literal["mlx", "metal"]
type MLXFP4CachePolicy = Literal["reuse", "recompute"]

_GROUP_SIZE: dict[MLXFP4Mode, int] = {
    "mxfp4": 32,
    "nvfp4": 16,
}


def group_size_for(mode: MLXFP4Mode) -> int:
    if mode not in _GROUP_SIZE:
        raise ValueError(f"unsupported MLX FP4 mode: {mode!r}")
    return _GROUP_SIZE[mode]


def expect_pair(value: object, source: str) -> tuple[mx.array, mx.array]:
    if not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(f"{source} must return a pair for FP4 modes")
    packed, scales = value[0], value[1]
    if not isinstance(packed, mx.array) or not isinstance(scales, mx.array):
        raise TypeError(f"{source} must return MLX arrays")
    return packed, scales


@dataclass(frozen=True)
class MLXFP4Weight:
    """Opaque MLX FP4-packed linear weight plus validated reconstruction metadata."""

    packed: mx.array
    scales: mx.array
    shape: tuple[int, int]
    mode: MLXFP4Mode
    group_size: int
    source_dtype: str

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError(f"MLX FP4 weight shape must be positive rank-2, got {self.shape}")
        expected_group_size = group_size_for(self.mode)
        if self.group_size != expected_group_size:
            raise ValueError(
                f"{self.mode} requires group_size={expected_group_size}, got {self.group_size}"
            )
        if self.shape[1] % self.group_size != 0:
            raise ValueError(
                f"weight input dim {self.shape[1]} must be divisible by group_size={self.group_size}"
            )
        if not self.source_dtype:
            raise ValueError("source_dtype must be non-empty")

    def dequantize(self, *, dtype: mx.Dtype = mx.float32) -> mx.array:
        out = mx.dequantize(
            self.packed,
            self.scales,
            group_size=self.group_size,
            bits=4,
            mode=self.mode,
            dtype=dtype,
        )
        mx.eval(out)
        return out
