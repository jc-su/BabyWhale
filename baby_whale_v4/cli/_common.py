"""Shared CLI argument choices."""

QUANT_CHOICES: tuple[str, ...] = (
    "none",
    "int8-weight",
    "int4-weight",
    "fp4-expert",
    "fp4-native",
)

PRECISION_CHOICES: tuple[str, ...] = ("fp32", "fp16", "bf16")

OPTIMIZER_CHOICES: tuple[str, ...] = ("adamw", "adafactor", "muon")

RUNTIME_CHOICES: tuple[str, ...] = ("mlx-metal", "mlx-cuda")

FP4_GRAD_CHOICES: tuple[str, ...] = ("mlx", "metal")
FP4_CACHE_CHOICES: tuple[str, ...] = ("reuse", "recompute")
FP4_MASTER_DTYPE_CHOICES: tuple[str, ...] = ("fp32", "bf16")
