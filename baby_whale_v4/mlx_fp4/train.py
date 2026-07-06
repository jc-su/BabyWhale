"""Native MLX FP4 forward + custom-VJP training paths.

The 16 (mode x weight_grad x cache_policy x bias) combinations are emitted by
four factory functions. Each factory closes over the variant's mode/grad/bias
shape to satisfy the ``mx.custom_function`` static-signature requirement, so we
keep four template bodies instead of sixteen near-duplicate ones.
"""

from typing import cast

import mlx.core as mx

from baby_whale_v4.mlx_fp4.packing import quantize_weight_unchecked
from baby_whale_v4.mlx_fp4.types import (
    MLXFP4CachePolicy,
    MLXFP4Mode,
    MLXFP4WeightGrad,
    group_size_for,
)


def _bias_grad(cotangent: mx.array) -> mx.array:
    axes = tuple(range(cotangent.ndim - 1))
    return mx.sum(cotangent, axis=axes)


def _validate_linear_train_inputs(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
) -> None:
    if not isinstance(x, mx.array):
        raise TypeError("MLX FP4 training linear input must be an mlx.core.array")
    if not isinstance(weight, mx.array):
        raise TypeError("MLX FP4 training linear weight must be an mlx.core.array")
    if weight.ndim != 2:
        raise ValueError("MLX FP4 training linear weight must be 2D [out, in]")
    if x.ndim == 0:
        raise ValueError("MLX FP4 training linear input must have at least one dimension")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"input last dim {x.shape[-1]} does not match weight dim {weight.shape[1]}"
        )
    if bias is not None and tuple(bias.shape) != (weight.shape[0],):
        raise ValueError(f"bias shape must be {(weight.shape[0],)}, got {tuple(bias.shape)}")


def _forward_with_pack(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    mode: MLXFP4Mode,
) -> tuple[mx.array, mx.array, mx.array]:
    packed, scales, group_size = quantize_weight_unchecked(weight, mode)
    original_shape = tuple(int(v) for v in x.shape)
    x2 = x.reshape(-1, original_shape[-1])
    y = mx.quantized_matmul(
        x2,
        packed,
        scales,
        None,
        transpose=True,
        group_size=group_size,
        bits=4,
        mode=mode,
    ).astype(x.dtype)
    y = y.reshape(*original_shape[:-1], weight.shape[0])
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return y, packed, scales


def _forward_only(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    mode: MLXFP4Mode,
) -> mx.array:
    return _forward_with_pack(x, weight, bias, mode)[0]


def _backward_no_bias(
    x: mx.array,
    weight: mx.array,
    cotangent: mx.array,
    mode: MLXFP4Mode,
    packed: mx.array,
    scales: mx.array,
    *,
    weight_grad: MLXFP4WeightGrad,
) -> tuple[mx.array, mx.array]:
    original_x_shape = tuple(int(v) for v in x.shape)
    x2 = x.reshape(-1, original_x_shape[-1])
    dy2 = cotangent.reshape(-1, cotangent.shape[-1])
    group_size = group_size_for(mode)
    dx2 = mx.quantized_matmul(
        dy2,
        packed,
        scales,
        None,
        transpose=False,
        group_size=group_size,
        bits=4,
        mode=mode,
    )
    match weight_grad:
        case "mlx":
            dw = dy2.T @ x2
        case "metal":
            from baby_whale_v4.kernels.fp4_training import linear_weight_grad_metal

            dw = linear_weight_grad_metal(dy2.astype(mx.float32), x2.astype(mx.float32))
        case _:
            raise ValueError(f"unsupported MLX FP4 weight_grad path: {weight_grad!r}")
    return dx2.reshape(*original_x_shape).astype(x.dtype), dw.astype(weight.dtype)


def _backward_recompute_no_bias(
    x: mx.array,
    weight: mx.array,
    cotangent: mx.array,
    mode: MLXFP4Mode,
    *,
    weight_grad: MLXFP4WeightGrad,
) -> tuple[mx.array, mx.array]:
    packed, scales, _gs = quantize_weight_unchecked(weight, mode)
    return _backward_no_bias(x, weight, cotangent, mode, packed, scales, weight_grad=weight_grad)


def _make_reuse_no_bias(mode: MLXFP4Mode, weight_grad: MLXFP4WeightGrad):
    @mx.custom_function
    def fn(x: mx.array, weight: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        return _forward_with_pack(x, weight, None, mode)

    @fn.vjp
    def fn_vjp(
        primals: tuple[mx.array, mx.array],
        cotangent: tuple[mx.array, mx.array, mx.array],
        output: tuple[mx.array, mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        x, weight = primals
        _y, packed, scales = output
        return _backward_no_bias(
            x, weight, cotangent[0], mode, packed, scales, weight_grad=weight_grad
        )

    return fn


def _make_reuse_bias(mode: MLXFP4Mode, weight_grad: MLXFP4WeightGrad):
    @mx.custom_function
    def fn(x: mx.array, weight: mx.array, bias: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        return _forward_with_pack(x, weight, bias, mode)

    @fn.vjp
    def fn_vjp(
        primals: tuple[mx.array, mx.array, mx.array],
        cotangent: tuple[mx.array, mx.array, mx.array],
        output: tuple[mx.array, mx.array, mx.array],
    ) -> tuple[mx.array, mx.array, mx.array]:
        x, weight, _bias = primals
        _y, packed, scales = output
        dy = cotangent[0]
        dx, dw = _backward_no_bias(x, weight, dy, mode, packed, scales, weight_grad=weight_grad)
        return dx, dw, _bias_grad(dy)

    return fn


def _make_recompute_no_bias(mode: MLXFP4Mode, weight_grad: MLXFP4WeightGrad):
    @mx.custom_function
    def fn(x: mx.array, weight: mx.array) -> mx.array:
        return _forward_only(x, weight, None, mode)

    @fn.vjp
    def fn_vjp(
        primals: tuple[mx.array, mx.array],
        cotangent: mx.array,
        output: mx.array,
    ) -> tuple[mx.array, mx.array]:
        del output
        x, weight = primals
        return _backward_recompute_no_bias(x, weight, cotangent, mode, weight_grad=weight_grad)

    return fn


def _make_recompute_bias(mode: MLXFP4Mode, weight_grad: MLXFP4WeightGrad):
    @mx.custom_function
    def fn(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
        return _forward_only(x, weight, bias, mode)

    @fn.vjp
    def fn_vjp(
        primals: tuple[mx.array, mx.array, mx.array],
        cotangent: mx.array,
        output: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        del output
        x, weight, _bias = primals
        dx, dw = _backward_recompute_no_bias(x, weight, cotangent, mode, weight_grad=weight_grad)
        return dx, dw, _bias_grad(cotangent)

    return fn


# Pre-build the 16 dispatch entries at import time so each (mx.custom_function)
# is statically registered with MLX exactly once.
_REUSE_NO_BIAS = {
    (mode, grad): _make_reuse_no_bias(mode, grad)
    for mode in ("mxfp4", "nvfp4")
    for grad in ("mlx", "metal")
}
_REUSE_BIAS = {
    (mode, grad): _make_reuse_bias(mode, grad)
    for mode in ("mxfp4", "nvfp4")
    for grad in ("mlx", "metal")
}
_RECOMPUTE_NO_BIAS = {
    (mode, grad): _make_recompute_no_bias(mode, grad)
    for mode in ("mxfp4", "nvfp4")
    for grad in ("mlx", "metal")
}
_RECOMPUTE_BIAS = {
    (mode, grad): _make_recompute_bias(mode, grad)
    for mode in ("mxfp4", "nvfp4")
    for grad in ("mlx", "metal")
}


def linear_mlx_fp4_train(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    mode: MLXFP4Mode = "nvfp4",
    weight_grad: MLXFP4WeightGrad = "mlx",
    cache_policy: MLXFP4CachePolicy = "reuse",
) -> mx.array:
    """Native MLX FP4 forward with a Python/MLX custom VJP for training.

    The forward path uses ``mlx.core.quantized_matmul`` with packed FP4 weights.
    The backward path is defined explicitly with MLX ops because MLX does not
    currently provide a VJP for quantized weights.
    """

    _validate_linear_train_inputs(x, weight, bias)
    if mode not in ("mxfp4", "nvfp4"):
        raise ValueError(f"unsupported MLX FP4 mode: {mode!r}")
    if weight_grad not in ("mlx", "metal"):
        raise ValueError(f"unsupported MLX FP4 weight_grad path: {weight_grad!r}")
    if cache_policy not in ("reuse", "recompute"):
        raise ValueError(f"unsupported MLX FP4 cache_policy: {cache_policy!r}")

    key = (mode, weight_grad)
    match (cache_policy, bias is None):
        case ("reuse", True):
            out = _REUSE_NO_BIAS[key](x, weight)
            return cast(tuple[mx.array, mx.array, mx.array], out)[0]
        case ("reuse", False):
            assert bias is not None
            out = _REUSE_BIAS[key](x, weight, bias)
            return cast(tuple[mx.array, mx.array, mx.array], out)[0]
        case ("recompute", True):
            return cast(mx.array, _RECOMPUTE_NO_BIAS[key](x, weight))
        case ("recompute", False):
            assert bias is not None
            return cast(mx.array, _RECOMPUTE_BIAS[key](x, weight, bias))
        case _:
            raise AssertionError("unreachable MLX FP4 training mode")
