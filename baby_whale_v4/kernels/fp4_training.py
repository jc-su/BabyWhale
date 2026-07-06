from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from time import perf_counter
from typing import cast

import mlx.core as mx

from baby_whale_v4.kernels.metal import (
    MetalKernel,
    _call_metal_kernel,
    require_metal_kernel_runtime,
)


@dataclass(frozen=True)
class MetalWeightGradBenchmark:
    dense_ms: float
    metal_ms: float
    ratio: float
    max_ratio: float

    def __post_init__(self) -> None:
        if self.dense_ms <= 0 or self.metal_ms <= 0:
            raise ValueError("benchmark timings must be positive")
        if self.ratio <= 0 or self.max_ratio <= 0:
            raise ValueError("benchmark ratios must be positive")

    @property
    def passed(self) -> bool:
        return self.ratio <= self.max_ratio


def linear_weight_grad_metal(dy: mx.array, x: mx.array) -> mx.array:
    """Compute ``dy.T @ x`` with a dedicated MLX custom Metal kernel.

    This is the weight-gradient leg used by the explicit FP4 training VJP. It
    compiles only this kernel through MLX's runtime Metal hook; it does not
    require rebuilding MLX. The kernel uses 8x8 ``simdgroup_matrix`` tiles, so
    it is a real Metal matrix kernel rather than a scalar thread-per-output
    loop.
    """

    _validate_weight_grad_inputs(dy, x)
    require_metal_kernel_runtime()
    batch, output_dims = (int(v) for v in dy.shape)
    input_dims = int(x.shape[1])
    tile = 8
    input_tiles = (input_dims + tile - 1) // tile
    output_tiles = (output_dims + tile - 1) // tile
    threadgroup = 32
    return _call_metal_kernel(
        _weight_grad_kernel(),
        inputs=[dy, x],
        template=[
            ("BATCH", batch),
            ("IN_DIMS", input_dims),
            ("OUT_DIMS", output_dims),
            ("IN_TILES", input_tiles),
        ],
        grid=(input_tiles * output_tiles * threadgroup, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(output_dims, input_dims)],
        output_dtypes=[mx.float32],
    )[0]


@cache
def benchmark_metal_weight_grad(
    *,
    batch: int = 16,
    input_dims: int = 256,
    output_dims: int = 256,
    warmup_steps: int = 2,
    timed_steps: int = 10,
    max_ratio: float = 1.0,
) -> MetalWeightGradBenchmark:
    if batch <= 0 or input_dims <= 0 or output_dims <= 0:
        raise ValueError("benchmark dimensions must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if timed_steps <= 0:
        raise ValueError("timed_steps must be positive")
    if max_ratio <= 0:
        raise ValueError("max_ratio must be positive")

    require_metal_kernel_runtime()
    mx.random.seed(0)
    x = mx.random.normal((batch, input_dims), dtype=mx.float32)
    dy = mx.random.normal((batch, output_dims), dtype=mx.float32)

    for _ in range(warmup_steps):
        _eval_dense_weight_grad(dy, x)
        _eval_metal_weight_grad(dy, x)

    dense_ms = _time_weight_grad(lambda: _eval_dense_weight_grad(dy, x), timed_steps)
    metal_ms = _time_weight_grad(lambda: _eval_metal_weight_grad(dy, x), timed_steps)
    return MetalWeightGradBenchmark(
        dense_ms=dense_ms,
        metal_ms=metal_ms,
        ratio=metal_ms / dense_ms,
        max_ratio=max_ratio,
    )


def require_metal_weight_grad_performance(max_ratio: float = 1.0) -> None:
    bench = benchmark_metal_weight_grad(max_ratio=max_ratio)
    if not bench.passed:
        raise RuntimeError(
            "custom Metal FP4 weight-gradient kernel is slower than MLX matmul: "
            f"metal={bench.metal_ms:.4f}ms dense={bench.dense_ms:.4f}ms "
            f"ratio={bench.ratio:.2f} max_ratio={bench.max_ratio:.2f}"
        )


@cache
def _weight_grad_kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="bwv4_fp4_weight_grad",
            input_names=["dy", "x"],
            output_names=["dw"],
            header="""
                #include <metal_simdgroup>
                #include <metal_simdgroup_matrix>
                using namespace metal;

                METAL_FUNC short2 bwv4_frag_coord(ushort lane) {
                    const short qid = lane / 4;
                    const short fm = (qid & 4) + ((lane / 2) % 4);
                    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
                    return short2(fn, fm);
                }
            """,
            source="""
                typedef metal::vec<float, 2> frag_t;

                const uint group = threadgroup_position_in_grid.x;
                const uint tile_i_id = group % IN_TILES;
                const uint tile_o_id = group / IN_TILES;
                const uint lane = thread_index_in_simdgroup;
                const short2 coord = bwv4_frag_coord(ushort(lane));

                simdgroup_matrix<float, 8, 8> dy_t;
                simdgroup_matrix<float, 8, 8> x_tile;
                simdgroup_matrix<float, 8, 8> acc;

                thread frag_t& dy_frag =
                    reinterpret_cast<thread frag_t&>(dy_t.thread_elements());
                thread frag_t& x_frag =
                    reinterpret_cast<thread frag_t&>(x_tile.thread_elements());
                thread frag_t& acc_frag =
                    reinterpret_cast<thread frag_t&>(acc.thread_elements());
                acc_frag[0] = 0.0f;
                acc_frag[1] = 0.0f;

                for (uint b0 = 0; b0 < BATCH; b0 += 8) {
                    const uint dy_row = tile_o_id * 8 + coord.y;
                    const uint dy_col0 = b0 + coord.x;
                    const uint dy_col1 = b0 + coord.x + 1;
                    dy_frag[0] = (dy_row < OUT_DIMS && dy_col0 < BATCH)
                        ? float(dy[dy_col0 * OUT_DIMS + dy_row])
                        : 0.0f;
                    dy_frag[1] = (dy_row < OUT_DIMS && dy_col1 < BATCH)
                        ? float(dy[dy_col1 * OUT_DIMS + dy_row])
                        : 0.0f;

                    const uint x_row = b0 + coord.y;
                    const uint x_col0 = tile_i_id * 8 + coord.x;
                    const uint x_col1 = tile_i_id * 8 + coord.x + 1;
                    x_frag[0] = (x_row < BATCH && x_col0 < IN_DIMS)
                        ? float(x[x_row * IN_DIMS + x_col0])
                        : 0.0f;
                    x_frag[1] = (x_row < BATCH && x_col1 < IN_DIMS)
                        ? float(x[x_row * IN_DIMS + x_col1])
                        : 0.0f;

                    simdgroup_multiply_accumulate(acc, dy_t, x_tile, acc);
                }

                const uint o = tile_o_id * 8 + coord.y;
                const uint i0 = tile_i_id * 8 + coord.x;
                const uint i1 = tile_i_id * 8 + coord.x + 1;
                if (o < OUT_DIMS && i0 < IN_DIMS) {
                    dw[o * IN_DIMS + i0] = acc_frag[0];
                }
                if (o < OUT_DIMS && i1 < IN_DIMS) {
                    dw[o * IN_DIMS + i1] = acc_frag[1];
                }
            """,
        ),
    )


def _validate_weight_grad_inputs(dy: mx.array, x: mx.array) -> None:
    if not isinstance(dy, mx.array):
        raise TypeError("dy must be an mlx.core.array")
    if not isinstance(x, mx.array):
        raise TypeError("x must be an mlx.core.array")
    if dy.ndim != 2:
        raise ValueError(f"dy must be rank-2 [batch, output_dims], got {tuple(dy.shape)}")
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [batch, input_dims], got {tuple(x.shape)}")
    if dy.shape[0] != x.shape[0]:
        raise ValueError(f"dy batch {dy.shape[0]} does not match x batch {x.shape[0]}")
    if dy.dtype != mx.float32:
        raise TypeError(f"dy must be float32 for the Metal kernel, got {dy.dtype}")
    if x.dtype != mx.float32:
        raise TypeError(f"x must be float32 for the Metal kernel, got {x.dtype}")


def _eval_dense_weight_grad(dy: mx.array, x: mx.array) -> mx.array:
    out = dy.T @ x
    mx.eval(out)
    return out


def _eval_metal_weight_grad(dy: mx.array, x: mx.array) -> mx.array:
    out = linear_weight_grad_metal(dy, x)
    mx.eval(out)
    return out


def _time_weight_grad(fn: Callable[[], mx.array], steps: int) -> float:
    start = perf_counter()
    for _ in range(steps):
        fn()
    elapsed = perf_counter() - start
    return (elapsed / steps) * 1000.0
