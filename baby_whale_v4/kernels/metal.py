from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, cast

import mlx.core as mx

type MetalKernel = Callable[..., list[mx.array]]


@dataclass(frozen=True)
class MetalKernelStatus:
    supported: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self.supported) is not bool:
            raise TypeError("MetalKernelStatus.supported must be a boolean")
        if not self.reason:
            raise ValueError("MetalKernelStatus.reason must be non-empty")


@cache
def _add_one_kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="bwv4_add_one_probe",
            input_names=["inp"],
            output_names=["out"],
            source="""
                uint elem = thread_position_in_grid.x;
                out[elem] = inp[elem] + T(1);
            """,
        ),
    )


def _call_metal_kernel(kernel: MetalKernel, **kwargs: Any) -> list[mx.array]:
    return kernel(**kwargs)


@cache
def probe_metal_kernel_runtime() -> MetalKernelStatus:
    """Verify that MLX can compile and execute a Python-defined Metal kernel.

    Returns a non-supported status (rather than raising) when the active MLX
    wheel doesn't expose ``mx.metal`` at all. Callers that strictly require Apple Metal should use
    ``require_metal_kernel_runtime()`` which raises.
    """

    try:
        metal = getattr(mx, "metal", None)
        if metal is None or not callable(getattr(metal, "is_available", None)):
            return MetalKernelStatus(
                supported=False,
                reason="MLX runtime does not expose mx.metal",
            )
        if not metal.is_available():
            return MetalKernelStatus(supported=False, reason="MLX Metal backend is unavailable")
        x = mx.arange(8, dtype=mx.float32)
        y = _call_metal_kernel(
            _add_one_kernel(),
            inputs=[x],
            template=[("T", mx.float32)],
            grid=(x.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]
        mx.eval(y)
    except Exception as exc:
        return MetalKernelStatus(supported=False, reason=f"{type(exc).__name__}: {exc}")
    if not bool(mx.allclose(y, x + 1)):
        return MetalKernelStatus(supported=False, reason="Metal probe kernel returned wrong values")
    return MetalKernelStatus(supported=True, reason="MLX custom Metal kernels are available")


def require_metal_kernel_runtime() -> None:
    status = probe_metal_kernel_runtime()
    if not status.supported:
        raise RuntimeError(f"MLX custom Metal kernels are not available: {status.reason}")
