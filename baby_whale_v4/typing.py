from dataclasses import dataclass
from typing import Literal, Never, NewType

import mlx.core as mx

# Branded ID types. Each is structurally a `str` at runtime but distinct to
# the type checker — so a `ConfigHash` cannot be silently passed where a
# `TokenizerHash` is expected, even though both are strings. Construct one
# explicitly at the boundary (e.g. ``ConfigHash(cfg.config_hash_raw())``) and
# the rest of the code flows the brand through inference.
ProblemId = NewType("ProblemId", str)
RequestId = NewType("RequestId", str)
ConfigHash = NewType("ConfigHash", str)
TokenizerHash = NewType("TokenizerHash", str)

Backend = Literal["mlx"]
MLXRuntime = Literal["mlx-metal", "mlx-cuda"]
Precision = Literal["fp32", "fp16", "bf16"]
AttentionKind = Literal["sliding_mqa", "hca", "csa", "mla"]
RouteKind = Literal["hash", "learned"]
OptimizerKind = Literal["adamw", "adafactor", "muon"]
SchedulerKind = Literal["constant", "cosine"]
LinearPlacement = Literal["general", "attention", "moe_expert", "router", "lm_head", "mtp"]
QuantMode = Literal[
    "none",
    "int8-weight",
    "int4-weight",
    "fp4-expert",
    "fp4-native",
]
ResolvedQuantMode = Literal[
    "none",
    "int8-weight",
    "int4-weight",
    "fp4-native",
]
ChatRole = Literal["system", "user", "assistant", "tool"]
ThinkMode = Literal["none", "thinking"]
TokenizerKind = Literal["byte", "byte_bpe"]
VisionEncoderKind = Literal["siglip"]
GenerationMode = Literal["greedy", "sample", "speculative"]
RewardKind = Literal["arithmetic", "format_json", "needle"]


@dataclass(frozen=True)
class Ok[T]:
    value: T

    def __post_init__(self) -> None:
        if isinstance(self.value, Err):
            raise TypeError("Ok value must not be an Err")

    @property
    def is_ok(self) -> bool:
        return True

    def unwrap(self) -> T:
        return self.value


@dataclass(frozen=True)
class Err[E]:
    error: E

    def __post_init__(self) -> None:
        if isinstance(self.error, Ok):
            raise TypeError("Err error must not be an Ok")
        if self.error is None:
            raise ValueError("Err error must be non-null")

    @property
    def is_ok(self) -> bool:
        return False

    def unwrap(self) -> Never:
        raise ValueError(f"Result was Err({self.error!r})")


type Result[T, E] = Ok[T] | Err[E]


def assert_never(value: Never) -> Never:
    raise AssertionError(f"unhandled variant: {value!r}")


def ensure_tensor(name: str, x: mx.array, *, ndim: int, dtype: object | None = None) -> None:
    if not isinstance(x, mx.array):
        raise TypeError(f"{name} must be an mlx.core.array, got {type(x).__name__}")
    if x.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got {x.ndim} (shape={tuple(x.shape)})")
    if dtype is not None and x.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {x.dtype}")


def ensure_in[T](name: str, value: T, choices: tuple[T, ...]) -> None:
    if value not in choices:
        raise ValueError(f"unsupported {name} {value!r}; supported: {list(choices)}")


def array_to_int_tuple(arr: mx.array) -> tuple[int, ...]:
    """Materialize a 1-D MLX int array as a ``tuple[int, ...]`` with a runtime check.

    ``mlx.core.array.tolist()`` is statically typed as ``int | float | list[...]``
    because it dispatches on rank. Most call sites here pass a 1-D prompt and
    expect a list; this helper narrows the rank once instead of sprinkling the
    same comprehension and cast across every RL trainer.
    """
    if arr.ndim != 1:
        raise ValueError(f"array_to_int_tuple expects ndim=1, got shape={tuple(arr.shape)}")
    values = arr.tolist()
    if not isinstance(values, list):
        raise TypeError(
            f"array_to_int_tuple: tolist() returned {type(values).__name__}, expected list"
        )
    return tuple(int(v) for v in values)
