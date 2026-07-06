import mlx.core as mx

from baby_whale_v4.mlx_fp4.types import MLXFP4Weight


def linear_mlx_fp4(
    x: mx.array,
    weight: MLXFP4Weight,
    bias: mx.array | None = None,
    *,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Run ``x @ weight.T + bias`` through MLX ``quantized_matmul``."""

    if not isinstance(x, mx.array):
        raise TypeError("MLX FP4 linear input must be an mlx.core.array")
    if x.ndim == 0:
        raise ValueError("MLX FP4 linear input must have at least one dimension")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"input last dim {x.shape[-1]} does not match weight dim {weight.shape[1]}"
        )
    if bias is not None and tuple(bias.shape) != (weight.shape[0],):
        raise ValueError(f"bias shape must be {(weight.shape[0],)}, got {tuple(bias.shape)}")
    if not bool(mx.all(mx.isfinite(x))):
        raise ValueError("MLX FP4 linear input must be finite")
    if bias is not None and not bool(mx.all(mx.isfinite(bias))):
        raise ValueError("MLX FP4 linear bias must be finite")

    original_shape = tuple(int(v) for v in x.shape)
    x2 = x.reshape(-1, original_shape[-1])
    y = mx.quantized_matmul(
        x2,
        weight.packed,
        weight.scales,
        None,
        transpose=True,
        group_size=weight.group_size,
        bits=4,
        mode=weight.mode,
    ).astype(dtype)
    y = y.reshape(*original_shape[:-1], weight.shape[0])
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return y
