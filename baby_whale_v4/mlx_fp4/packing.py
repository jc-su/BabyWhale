import mlx.core as mx

from baby_whale_v4.mlx_fp4.types import (
    MLXFP4Mode,
    MLXFP4Weight,
    expect_pair,
    group_size_for,
)


def quantize_weight_mlx_fp4(weight: mx.array, mode: MLXFP4Mode = "mxfp4") -> MLXFP4Weight:
    """Pack an MLX linear weight with MLX FP4 quantization."""

    if not isinstance(weight, mx.array):
        raise TypeError("MLX FP4 weight quantization expects an mlx.core.array")
    if weight.ndim != 2:
        raise ValueError("MLX FP4 weight quantization expects a 2D linear weight [out, in]")
    group_size = group_size_for(mode)
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"{mode} requires weight.shape[-1] divisible by group_size={group_size}; "
            f"got {weight.shape[-1]}"
        )
    if not bool(mx.all(mx.isfinite(weight))):
        raise ValueError("MLX FP4 weight must be finite before quantization")

    packed, scales = expect_pair(
        mx.quantize(weight, group_size=group_size, bits=4, mode=mode),
        "mlx.core.quantize",
    )
    mx.eval(packed, scales)
    return MLXFP4Weight(
        packed=packed,
        scales=scales,
        shape=(int(weight.shape[0]), int(weight.shape[1])),
        mode=mode,
        group_size=group_size,
        source_dtype=str(weight.dtype),
    )


def dequantize_weight_mlx_fp4(weight: mx.array, mode: MLXFP4Mode = "mxfp4") -> mx.array:
    """Quantize then dequantize an MLX weight using the requested FP4 format."""

    return quantize_weight_mlx_fp4(weight, mode).dequantize(dtype=weight.dtype)


def quantize_weight_unchecked(
    weight: mx.array,
    mode: MLXFP4Mode,
) -> tuple[mx.array, mx.array, int]:
    """Fast path used inside training VJPs; skips finiteness checks."""

    group_size = group_size_for(mode)
    packed, scales = expect_pair(
        mx.quantize(weight, group_size=group_size, bits=4, mode=mode),
        "mlx.core.quantize",
    )
    return packed, scales, group_size
