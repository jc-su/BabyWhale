"""Fail-fast MLX runtime checks.

Baby Whale v4 uses MLX as the framework backend. The runtime can be Apple
Metal (`mlx-metal`) or NVIDIA CUDA through MLX (`mlx-cuda`). CPU execution is
not a project runtime target.
"""

from __future__ import annotations

from typing import cast, get_args

import mlx.core as mx

from baby_whale_v4.typing import Backend, MLXRuntime

_RUNTIMES: tuple[MLXRuntime, ...] = get_args(MLXRuntime)


def _module_is_available(name: str) -> bool:
    module = getattr(mx, name, None)
    if module is None:
        return False
    is_available = getattr(module, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())


def is_metal_runtime() -> bool:
    """True iff the active MLX wheel exposes a working Metal runtime."""

    return _module_is_available("metal")


def is_cuda_runtime() -> bool:
    """True iff the active MLX wheel exposes a working CUDA runtime."""

    return _module_is_available("cuda")


def available_runtimes() -> tuple[MLXRuntime, ...]:
    runtimes: list[MLXRuntime] = []
    if is_metal_runtime():
        runtimes.append("mlx-metal")
    if is_cuda_runtime():
        runtimes.append("mlx-cuda")
    return tuple(runtimes)


def active_runtime() -> MLXRuntime:
    """Return the only available MLX runtime, or fail if selection is ambiguous."""

    runtimes = available_runtimes()
    if len(runtimes) == 1:
        return runtimes[0]
    if not runtimes:
        raise RuntimeError("Baby Whale v4 requires a working MLX Metal or MLX CUDA runtime")
    raise RuntimeError(
        f"multiple MLX runtimes are available {list(runtimes)}; choose runtime explicitly"
    )


def ensure_runtime_matches(
    backend: Backend,
    runtime: MLXRuntime | object | None = None,
) -> MLXRuntime:
    """Validate the requested MLX framework backend and concrete runtime."""

    if backend != "mlx":
        raise ValueError(f"unsupported backend {backend!r}; supported: ['mlx']")
    requested = active_runtime() if runtime is None else runtime
    if requested not in _RUNTIMES:
        raise ValueError(f"unsupported runtime {requested!r}; supported: {list(_RUNTIMES)}")
    runtime_t = cast(MLXRuntime, requested)
    if runtime_t not in available_runtimes():
        raise RuntimeError(
            f"requested runtime {runtime_t!r} is unavailable; available: {list(available_runtimes())}"
        )
    return runtime_t
