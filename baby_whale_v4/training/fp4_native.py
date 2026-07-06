from dataclasses import dataclass
from functools import cache
from time import perf_counter
from typing import Literal

import mlx.core as mx

from baby_whale_v4.mlx_fp4 import (
    MLXFP4CachePolicy,
    MLXFP4WeightGrad,
    linear_mlx_fp4_train,
    quantize_weight_mlx_fp4,
)
from baby_whale_v4.training.mlx_optim import Adafactor, AdamW, Muon

type FP4MemoryOptimizer = Literal["none", "adamw", "adafactor", "muon"]
type FP4MasterDtype = Literal["fp32", "bf16"]


@dataclass(frozen=True)
class NativeFP4TrainingStatus:
    supported: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self.supported) is not bool:
            raise TypeError("supported must be a boolean")
        if not self.reason:
            raise ValueError("reason must be non-empty")


@dataclass(frozen=True)
class FP4TrainingBenchmark:
    dense_ms: float
    fp4_ms: float
    ratio: float
    max_ratio: float

    def __post_init__(self) -> None:
        if self.dense_ms <= 0 or self.fp4_ms <= 0:
            raise ValueError("benchmark timings must be positive")
        if self.ratio <= 0 or self.max_ratio <= 0:
            raise ValueError("benchmark ratios must be positive")

    @property
    def passed(self) -> bool:
        return self.ratio <= self.max_ratio


@dataclass(frozen=True)
class FP4TrainingMemoryBenchmark:
    dense_fp32_peak_bytes: int
    dense_fp32_active_bytes: int
    dense_bf16_peak_bytes: int
    dense_bf16_active_bytes: int
    fp4_peak_bytes: int
    fp4_active_bytes: int
    baseline: Literal["fp32", "bf16"]
    max_peak_ratio: float
    weight_grad: MLXFP4WeightGrad
    cache_policy: MLXFP4CachePolicy
    optimizer: FP4MemoryOptimizer
    fp4_master_dtype: FP4MasterDtype

    def __post_init__(self) -> None:
        byte_values = (
            self.dense_fp32_peak_bytes,
            self.dense_fp32_active_bytes,
            self.dense_bf16_peak_bytes,
            self.dense_bf16_active_bytes,
            self.fp4_peak_bytes,
            self.fp4_active_bytes,
        )
        if any(value <= 0 for value in byte_values):
            raise ValueError("memory benchmark byte counts must be positive")
        if self.max_peak_ratio <= 0:
            raise ValueError("max_peak_ratio must be positive")

    @property
    def baseline_peak_bytes(self) -> int:
        match self.baseline:
            case "fp32":
                return self.dense_fp32_peak_bytes
            case "bf16":
                return self.dense_bf16_peak_bytes
            case _:
                raise ValueError(f"unsupported memory baseline {self.baseline!r}")

    @property
    def peak_ratio(self) -> float:
        return self.fp4_peak_bytes / self.baseline_peak_bytes

    @property
    def baseline_active_bytes(self) -> int:
        match self.baseline:
            case "fp32":
                return self.dense_fp32_active_bytes
            case "bf16":
                return self.dense_bf16_active_bytes
            case _:
                raise ValueError(f"unsupported memory baseline {self.baseline!r}")

    @property
    def active_ratio(self) -> float:
        return self.fp4_active_bytes / self.baseline_active_bytes

    @property
    def passed(self) -> bool:
        return self.peak_ratio <= self.max_peak_ratio


@cache
def probe_native_fp4_training() -> NativeFP4TrainingStatus:
    """Probe whether MLX can train through native FP4 quantized matmul.

    Real native FP4 training requires gradients through the native FP4 matmul
    path. This intentionally does not use approximate gradients, dequantized
    matmul, or any non-native emulation.
    """

    mx.random.seed(0)
    weight = mx.random.normal((4, 16), dtype=mx.float32)
    x = mx.random.normal((3, 16), dtype=mx.float32)
    y = mx.random.normal((3, 4), dtype=mx.float32)

    def loss_fn(w: mx.array) -> mx.array:
        packed_weight = quantize_weight_mlx_fp4(w, "nvfp4")
        out = mx.quantized_matmul(
            x,
            packed_weight.packed,
            packed_weight.scales,
            None,
            transpose=True,
            group_size=16,
            bits=4,
            mode="nvfp4",
        )
        return mx.mean(mx.square(out - y))

    try:
        loss, grad = mx.value_and_grad(loss_fn)(weight)
        mx.eval(loss, grad)
    except Exception as exc:
        return NativeFP4TrainingStatus(supported=False, reason=f"{type(exc).__name__}: {exc}")
    if not bool(mx.all(mx.isfinite(grad))):
        return NativeFP4TrainingStatus(supported=False, reason="native FP4 gradient is non-finite")
    if float(mx.max(mx.abs(grad))) == 0.0:
        return NativeFP4TrainingStatus(supported=False, reason="native FP4 gradient is zero")
    return NativeFP4TrainingStatus(supported=True, reason="native FP4 gradient path is available")


@cache
def probe_custom_vjp_fp4_training() -> NativeFP4TrainingStatus:
    """Probe the no-recompile custom-VJP FP4 training path."""

    mx.random.seed(0)
    weight = mx.random.normal((4, 16), dtype=mx.float32)
    x = mx.random.normal((3, 16), dtype=mx.float32)
    y = mx.random.normal((3, 4), dtype=mx.float32)

    def loss_fn(w: mx.array) -> mx.array:
        out = linear_mlx_fp4_train(x, w, mode="nvfp4")
        return mx.mean(mx.square(out - y))

    try:
        loss, grad = mx.value_and_grad(loss_fn)(weight)
        mx.eval(loss, grad)
    except Exception as exc:
        return NativeFP4TrainingStatus(supported=False, reason=f"{type(exc).__name__}: {exc}")
    if not bool(mx.all(mx.isfinite(grad))):
        return NativeFP4TrainingStatus(
            supported=False, reason="custom VJP FP4 gradient is non-finite"
        )
    if float(mx.max(mx.abs(grad))) == 0.0:
        return NativeFP4TrainingStatus(supported=False, reason="custom VJP FP4 gradient is zero")
    return NativeFP4TrainingStatus(
        supported=True, reason="custom VJP FP4 gradient path is available"
    )


@cache
def probe_metal_vjp_fp4_training() -> NativeFP4TrainingStatus:
    """Probe the custom-VJP path that uses a Python-defined Metal gradient kernel."""

    mx.random.seed(0)
    weight = mx.random.normal((4, 16), dtype=mx.float32)
    x = mx.random.normal((3, 16), dtype=mx.float32)
    y = mx.random.normal((3, 4), dtype=mx.float32)

    def loss_fn(w: mx.array) -> mx.array:
        out = linear_mlx_fp4_train(x, w, mode="nvfp4", weight_grad="metal")
        return mx.mean(mx.square(out - y))

    try:
        loss, grad = mx.value_and_grad(loss_fn)(weight)
        mx.eval(loss, grad)
    except Exception as exc:
        return NativeFP4TrainingStatus(supported=False, reason=f"{type(exc).__name__}: {exc}")
    if not bool(mx.all(mx.isfinite(grad))):
        return NativeFP4TrainingStatus(
            supported=False, reason="Metal custom VJP FP4 gradient is non-finite"
        )
    if float(mx.max(mx.abs(grad))) == 0.0:
        return NativeFP4TrainingStatus(
            supported=False, reason="Metal custom VJP FP4 gradient is zero"
        )
    return NativeFP4TrainingStatus(
        supported=True, reason="Metal custom VJP FP4 gradient path is available"
    )


def require_native_fp4_training() -> None:
    status = probe_native_fp4_training()
    if not status.supported:
        raise RuntimeError(f"native MLX FP4 training is not supported: {status.reason}")


@cache
def benchmark_custom_vjp_fp4_training(
    *,
    batch: int = 8,
    input_dims: int = 128,
    output_dims: int = 128,
    warmup_steps: int = 8,
    timed_steps: int = 20,
    max_ratio: float = 1.5,
    weight_grad: MLXFP4WeightGrad = "mlx",
    cache_policy: MLXFP4CachePolicy = "reuse",
) -> FP4TrainingBenchmark:
    if batch <= 0 or input_dims <= 0 or output_dims <= 0:
        raise ValueError("benchmark dimensions must be positive")
    if input_dims % 16 != 0:
        raise ValueError("input_dims must be divisible by 16 for nvfp4")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if timed_steps <= 0:
        raise ValueError("timed_steps must be positive")
    if max_ratio <= 0:
        raise ValueError("max_ratio must be positive")

    dense_ms = _time_training_step(
        kind="dense",
        batch=batch,
        input_dims=input_dims,
        output_dims=output_dims,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
    )
    fp4_ms = _time_training_step(
        kind="fp4",
        batch=batch,
        input_dims=input_dims,
        output_dims=output_dims,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        weight_grad=weight_grad,
        cache_policy=cache_policy,
    )
    return FP4TrainingBenchmark(
        dense_ms=dense_ms,
        fp4_ms=fp4_ms,
        ratio=fp4_ms / dense_ms,
        max_ratio=max_ratio,
    )


@cache
def benchmark_fp4_training_memory(
    *,
    batch: int = 32,
    input_dims: int = 1024,
    output_dims: int = 1024,
    baseline: Literal["fp32", "bf16"] = "bf16",
    max_peak_ratio: float = 1.0,
    weight_grad: MLXFP4WeightGrad = "mlx",
    cache_policy: MLXFP4CachePolicy = "reuse",
    optimizer: FP4MemoryOptimizer = "none",
    fp4_master_dtype: FP4MasterDtype = "bf16",
) -> FP4TrainingMemoryBenchmark:
    if batch <= 0 or input_dims <= 0 or output_dims <= 0:
        raise ValueError("benchmark dimensions must be positive")
    if input_dims % 16 != 0:
        raise ValueError("input_dims must be divisible by 16 for nvfp4")
    if baseline not in ("fp32", "bf16"):
        raise ValueError(f"unsupported memory baseline {baseline!r}")
    if max_peak_ratio <= 0:
        raise ValueError("max_peak_ratio must be positive")
    fp4_dtype = _dtype_for_fp4_master(fp4_master_dtype)

    dense_fp32_peak, dense_fp32_active = _measure_training_step_memory(
        kind="dense",
        batch=batch,
        input_dims=input_dims,
        output_dims=output_dims,
        dtype=mx.float32,
        weight_grad=weight_grad,
        cache_policy=cache_policy,
        optimizer=optimizer,
    )
    dense_bf16_peak, dense_bf16_active = _measure_training_step_memory(
        kind="dense",
        batch=batch,
        input_dims=input_dims,
        output_dims=output_dims,
        dtype=mx.bfloat16,
        weight_grad=weight_grad,
        cache_policy=cache_policy,
        optimizer=optimizer,
    )
    fp4_peak, fp4_active = _measure_training_step_memory(
        kind="fp4",
        batch=batch,
        input_dims=input_dims,
        output_dims=output_dims,
        dtype=fp4_dtype,
        weight_grad=weight_grad,
        cache_policy=cache_policy,
        optimizer=optimizer,
    )
    return FP4TrainingMemoryBenchmark(
        dense_fp32_peak_bytes=dense_fp32_peak,
        dense_fp32_active_bytes=dense_fp32_active,
        dense_bf16_peak_bytes=dense_bf16_peak,
        dense_bf16_active_bytes=dense_bf16_active,
        fp4_peak_bytes=fp4_peak,
        fp4_active_bytes=fp4_active,
        baseline=baseline,
        max_peak_ratio=max_peak_ratio,
        weight_grad=weight_grad,
        cache_policy=cache_policy,
        optimizer=optimizer,
        fp4_master_dtype=fp4_master_dtype,
    )


def require_fp4_training_memory_efficiency(
    *,
    baseline: Literal["fp32", "bf16"] = "bf16",
    max_peak_ratio: float = 1.0,
    weight_grad: MLXFP4WeightGrad = "mlx",
    cache_policy: MLXFP4CachePolicy = "reuse",
    optimizer: FP4MemoryOptimizer = "none",
    fp4_master_dtype: FP4MasterDtype = "bf16",
) -> None:
    bench = benchmark_fp4_training_memory(
        baseline=baseline,
        max_peak_ratio=max_peak_ratio,
        weight_grad=weight_grad,
        cache_policy=cache_policy,
        optimizer=optimizer,
        fp4_master_dtype=fp4_master_dtype,
    )
    if not bench.passed:
        raise RuntimeError(
            "FP4 training peak memory is not lower than the requested baseline: "
            f"fp4_peak={bench.fp4_peak_bytes} baseline={bench.baseline_peak_bytes} "
            f"baseline_kind={bench.baseline} ratio={bench.peak_ratio:.2f} "
            f"max_ratio={bench.max_peak_ratio:.2f} cache_policy={cache_policy} "
            f"optimizer={optimizer} fp4_master_dtype={fp4_master_dtype}"
        )


def _time_training_step(
    *,
    kind: str,
    batch: int,
    input_dims: int,
    output_dims: int,
    warmup_steps: int,
    timed_steps: int,
    weight_grad: MLXFP4WeightGrad = "mlx",
    cache_policy: MLXFP4CachePolicy = "reuse",
) -> float:
    mx.random.seed(0)
    x = mx.random.normal((batch, input_dims), dtype=mx.float32)
    weight = mx.random.normal((output_dims, input_dims), dtype=mx.float32)
    target = mx.random.normal((batch, output_dims), dtype=mx.float32)

    match kind:
        case "dense":

            def loss_fn(w: mx.array) -> mx.array:
                return mx.mean(mx.square((x @ w.T) - target))
        case "fp4":

            def loss_fn(w: mx.array) -> mx.array:
                return mx.mean(
                    mx.square(
                        linear_mlx_fp4_train(
                            x,
                            w,
                            mode="nvfp4",
                            weight_grad=weight_grad,
                            cache_policy=cache_policy,
                        )
                        - target
                    )
                )
        case _:
            raise ValueError(f"unsupported benchmark kind {kind!r}")

    loss_and_grad = mx.value_and_grad(loss_fn)
    for _ in range(warmup_steps):
        loss, grad = loss_and_grad(weight)
        mx.eval(loss, grad)

    start = perf_counter()
    for _ in range(timed_steps):
        loss, grad = loss_and_grad(weight)
        mx.eval(loss, grad)
    elapsed = perf_counter() - start
    return (elapsed / timed_steps) * 1000.0


def _measure_training_step_memory(
    *,
    kind: Literal["dense", "fp4"],
    batch: int,
    input_dims: int,
    output_dims: int,
    dtype: mx.Dtype,
    weight_grad: MLXFP4WeightGrad,
    cache_policy: MLXFP4CachePolicy,
    optimizer: FP4MemoryOptimizer,
) -> tuple[int, int]:
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.random.seed(0)
    x = mx.random.normal((batch, input_dims), dtype=dtype)
    weight = mx.random.normal((output_dims, input_dims), dtype=dtype)
    target = mx.random.normal((batch, output_dims), dtype=dtype)

    match kind:
        case "dense":

            def loss_fn(w: mx.array) -> mx.array:
                return mx.mean(mx.square((x @ w.T) - target))
        case "fp4":

            def loss_fn(w: mx.array) -> mx.array:
                return mx.mean(
                    mx.square(
                        linear_mlx_fp4_train(
                            x,
                            w,
                            mode="nvfp4",
                            weight_grad=weight_grad,
                            cache_policy=cache_policy,
                        )
                        - target
                    )
                )
        case _:
            raise ValueError(f"unsupported memory benchmark kind {kind!r}")

    loss, grad = mx.value_and_grad(loss_fn)(weight)
    opt = _build_memory_optimizer(optimizer)
    if opt is None:
        mx.eval(loss, grad)
    else:
        params: dict[str, object] = {"weight": weight}
        grads: dict[str, object] = {"weight": grad}
        updates = opt.step(params, grads)
        mx.eval(loss, updates, opt.state_dict())
    return int(mx.get_peak_memory()), int(mx.get_active_memory())


def _build_memory_optimizer(optimizer: FP4MemoryOptimizer) -> AdamW | Adafactor | Muon | None:
    match optimizer:
        case "none":
            return None
        case "adamw":
            return AdamW(learning_rate=1e-3, weight_decay=0.0)
        case "adafactor":
            return Adafactor(learning_rate=1e-3, weight_decay=0.0)
        case "muon":
            return Muon(learning_rate=1e-3, weight_decay=0.0)
        case _:
            raise ValueError(f"unsupported memory optimizer {optimizer!r}")


def _dtype_for_fp4_master(dtype: FP4MasterDtype) -> mx.Dtype:
    match dtype:
        case "fp32":
            return mx.float32
        case "bf16":
            return mx.bfloat16
        case _:
            raise ValueError(f"unsupported FP4 master dtype {dtype!r}")
