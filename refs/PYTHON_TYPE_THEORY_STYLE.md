# Python Type-Theory Style For Baby Whale v4

Status: project guide, researched 2026-05-08  
Target Python: 3.14+  
Goal: write Python that feels closer to Rust/ML/Haskell in its invariants while staying pragmatic for MLX/Apple ML code.

## 1. Core Position

Python's type annotations are not enforced by the runtime. The standard library documentation is explicit about this: annotations are consumed by type checkers, IDEs, linters, and similar tools. Therefore, this project uses two layers:

1. Static types describe the design.
2. Runtime validators enforce boundaries.

For `baby_whale_v4`, this means:

- `Literal`, unions, `Protocol`, `NewType`, generic dataclasses, `type` aliases, and `assert_never` describe legal program states.
- `__post_init__`, `ensure_in`, `ensure_tensor`, device checks, dtype checks, and config hash checks enforce the states at runtime.
- Unit tests prove semantic invariants that Python's type system cannot express, especially tensor shape, device, cache length, and numerical behavior.

The rule is: use the type checker to reject bad code before execution, but never trust type hints at IO, config, tensor, checkpoint, cache, or API boundaries.

## 1.1. Project Policy: Fail Fast

This project is educational and API-breaking changes are allowed. Prefer a smaller, stricter API over compatibility layers that hide invalid state.

Do:

- raise explicit exceptions when config, device, dtype, cache state, checkpoint identity, or backend support is invalid.
- reject unknown config keys instead of ignoring them.
- require callers to select a real backend/device/precision instead of silently substituting another one.
- make invalid state impossible with types when Python can express it.
- make remaining invalid state fail at construction or public boundaries.

Do not:

- introduce alternate execution paths for unsupported features.
- catch broad exceptions and continue with a guessed behavior.
- preserve old names or argument shapes only for compatibility.
- use `Any`, `Optional`, or raw `dict` to defer a design decision.
- hide technical debt behind adapter layers.

## 2. Type Theory Translation Table

| Type-theory concept | Python pattern | Baby Whale usage |
| --- | --- | --- |
| Sum type / tagged union | `Literal[...]`, `A | B`, frozen dataclass variants | `Backend`, `Precision`, `QuantMode`, `Ok[T] | Err[E]` |
| Product type | `@dataclass`, preferably `frozen=True` for value objects | `PrefixCacheKey`, `Message`, `DPOExample` |
| Bottom type | `Never` / `NoReturn`, `assert_never(x)` | exhaustive backend/attention/quant branches |
| New nominal type | `NewType("TokenId", int)` | use when two ints must not mix semantically |
| Interface / typeclass | `Protocol` | `Tokenizer` |
| Refinement type | `__post_init__`, `ensure_tensor`, constructor validation | config, tensors, cache keys |
| Existential/plugin boundary | `Protocol` plus runtime validation | future reward/tokenizer/backend plugins |
| Type narrowing | `TypeGuard` / `TypeIs` from stdlib `typing` | parsing external payloads |
| Correlated input/output types | `@overload` | loaders, tokenizer decode modes, config constructors |
| Immutable value semantics | `@dataclass(frozen=True)` | cache keys, examples, messages |

## 3. Project Rules

### Rule 1: Use Python 3.14 Native Annotations

Do not add:

```python
from __future__ import annotations
```

Python 3.14 implements deferred annotation evaluation by default. Keeping the future import would preserve the older stringized-annotation behavior and block the project from using the current runtime annotation model.

Use native 3.14 syntax:

```python
class Ok[T]:
    value: T

type Result[T, E] = Ok[T] | Err[E]

def ensure_in[T](name: str, value: T, choices: tuple[T, ...]) -> None:
    ...
```

### Rule 2: No Raw Strings For Closed Sets

Use `Literal` aliases in `baby_whale_v4/typing.py`.

```python
Backend = Literal["mlx"]
MLXRuntime = Literal["mlx-metal", "mlx-cuda"]
Precision = Literal["fp32", "fp16", "bf16"]
AttentionKind = Literal["sliding_mqa", "hca", "csa"]
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
```

`fp4-expert` is a requested boundary mode, not a layer runtime mode. Resolve it with
`QuantizedLinearPolicy` into `ResolvedQuantMode` before dispatching in `WhaleLinear`.
Every active `ResolvedQuantMode` runs through real MLX kernels: `int8-weight`/`int4-weight`
use `mx.quantized_matmul(mode="affine")` (group_size 64), `fp4-native` uses
`mode="mxfp4"`/`"nvfp4"`. Do not add `fp4-sim` or `fp8-sim` back to `QuantMode`: a
quantize/dequantize round-trip on Apple Silicon costs accuracy without matching the
hardware throughput of the real `mx.quantized_matmul` path.
Full-FP4 training experiments must not be added back to `QuantMode`; keep them as explicit primitive benchmark functions until they become a real supported path.

At boundaries, validate:

```python
ensure_in("precision", value, get_args(Precision))
```

Inside branches, exhaust:

```python
def select_backend(name: Backend) -> BackendRuntime:
    match name:
        case "mlx":
            return MLXRuntime()
        case _:
            assert_never(name)
```

When adding a new variant, the type checker and tests should force every branch to be updated.

### Rule 2.1: Use `match` For Closed-Set Dispatch

Python `match` is structural pattern matching. In type-theory style, use it for destructuring variants and dispatching over closed sets. This is the closest Python gets to ML/Rust-style pattern matching.

Good uses:

- `Literal` mode dispatch.
- `Ok(...) | Err(...)` result handling.
- dataclass variant matching.
- tuple/list shape parsing after the outer boundary has validated the input.

Closed-set dispatch:

```python
def sample(logits: mx.array, mode: GenerationMode) -> mx.array:
    match mode:
        case "greedy":
            return mx.argmax(logits, axis=-1)
        case "sample":
            return sample_from_distribution(logits)
        case "speculative":
            return mx.argmax(logits, axis=-1)
        case _:
            assert_never(mode)
```

Result handling:

```python
def unwrap_or_raise(result: Result[T, str]) -> T:
    match result:
        case Ok(value=value):
            return value
        case Err(error=error):
            raise ValueError(error)
        case _:
            assert_never(result)
```

Dataclass variant matching:

```python
@dataclass(frozen=True)
class AdamWOpt:
    lr: float

@dataclass(frozen=True)
class LionOpt:
    lr: float

OptimizerSpec = AdamWOpt | LionOpt

def build_optimizer(spec: OptimizerSpec) -> Optimizer:
    match spec:
        case AdamWOpt(lr=lr):
            return build_adamw(lr)
        case LionOpt(lr=lr):
            return build_lion(lr)
        case _:
            assert_never(spec)
```

Important rules:

- use `case _:` only to call `assert_never` for typed closed sets.
- do not put alternate execution behavior in the wildcard branch.
- do not use `match` just because it looks modern; normal `if` is better for numeric checks, tensor shape checks, and guard-heavy validation.
- keep guard clauses explicit before `match` when invalid inputs should fail with precise errors.

`match` does not replace runtime validation. It makes already-validated variants easier to read and easier for strict checkers to prove exhaustive.

### Rule 3: Public Dataclasses Validate Themselves

Any public dataclass must define `__post_init__`, unless it is a pure output record with no invariants.

Good:

```python
@dataclass
class GenerationOptions:
    max_new_tokens: int
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
```

For value semantics, freeze:

```python
@dataclass(frozen=True)
class PrefixCacheKey:
    prefix_hash: str
    config_hash: str
    tokenizer_hash: str
```

Use frozen dataclasses for:

- cache keys.
- dataset examples.
- chat messages.
- config identity fragments.
- result variants.

Use mutable dataclasses only for state machines:

- request state.
- mutable cache entries.
- metric writers.

### Rule 4: Tensor Types Need Runtime Refinement

Python typing cannot express most tensor invariants. This is not optional for ML code.

Use helpers:

```python
ensure_tensor("input_ids", input_ids, ndim=2, dtype=mx.int32)
```

Also check:

- device matches model device.
- shape matches semantic contract.
- total cached context length stays within `context_length`.
- dtype matches precision mode.
- finite loss/logits where numerical instability is possible.

Prefer boundary checks at:

- `model.forward`.
- inference request creation.
- cache append/clone.
- checkpoint load.
- dataset construction.
- public training entry points.

Do not scatter shape assertions deep inside every inner kernel unless it improves error locality.

### Rule 5: Treat Config As A Type Boundary

Config is untrusted input, even when it comes from local TOML or a checkpoint.

Required config behavior:

- unknown keys fail.
- unsupported backend/precision/cache/quant modes fail.
- cross-field invariants fail in `__post_init__`.
- `config_hash()` is stable and included in checkpoints and prefix caches.

Pattern:

```python
@classmethod
def from_dict(cls, data: dict[str, object]) -> Self:
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return cls(**data)
```

On Python 3.14, import `Self` from `typing` when it makes constructors or fluent APIs clearer.

### Rule 6: Model Runtime State Is Not A Config

Do not encode runtime state using optional fields on config. Use separate product types.

Good:

```python
@dataclass
class RequestState:
    prompt_ids: list[int]
    cache: DynamicKVCache
    prefilled: int = 0
```

Bad:

```python
config.prefilled = 3
config.cache = cache
```

Config is identity. State is execution.

### Rule 7: Prefer Result For Recoverable Internal Outcomes

Use `Result[T, E] = Ok[T] | Err[E]` when failure is expected and should be handled locally.

Use exceptions when:

- external input is invalid.
- config is impossible.
- checkpoint/cache is corrupt.
- a requested backend or precision is unsupported.
- continuing would produce misleading results.

Example:

```python
def parse_tool_call(text: str) -> Result[ToolCall, ParseError]:
    ...
```

Avoid returning `None` for errors unless absence is a normal state. `Optional[T]` is for "maybe present", not "operation failed".

### Rule 8: Protocols Define Plugins, Not Base Classes

Use `Protocol` for structural interfaces like tokenizers and reward functions.

```python
class Tokenizer(Protocol):
    kind: TokenizerKind
    vocab_size: int
    bos_id: int
    eos_id: int
    pad_id: int

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
```

Do not use inheritance unless shared implementation is the point. Protocols keep plugin code decoupled from framework code.

Only use `@runtime_checkable` when you truly need `isinstance(x, ProtocolType)`. Runtime protocol checks only verify attribute presence, not full type signatures, and can be slower than normal checks.

### Rule 9: Use NewType For Semantic Integers

Use `NewType` when two values have the same runtime representation but must not be mixed.

Good candidates:

```python
TokenId = NewType("TokenId", int)
LayerIdx = NewType("LayerIdx", int)
RequestId = NewType("RequestId", str)
ConfigHash = NewType("ConfigHash", str)
```

Use this sparingly. If the value needs runtime validation, prefer a frozen dataclass:

```python
@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid token span")
```

### Rule 10: Use Overloads For Correlated Return Types

If an argument changes the return type, use `@overload`.

```python
@overload
def load_checkpoint(path: Path, *, with_optimizer: Literal[True]) -> FullCheckpoint: ...

@overload
def load_checkpoint(path: Path, *, with_optimizer: Literal[False]) -> ModelCheckpoint: ...

def load_checkpoint(path: Path, *, with_optimizer: bool) -> FullCheckpoint | ModelCheckpoint:
    ...
```

Do not use overloads for basic branching when a simple `Union` is clear.

## 4. Recommended Tooling

Use the Astral stack as the default local workflow:

- `uv` manages the environment, dependency groups, and `uv.lock`.
- `ruff` formats code and catches style, import, modernization, and common bug patterns.
- `ty` is the fast default type checker for project code.

Commands:

```bash
conda run -n base uv sync --dev
conda run -n base uv run ruff format baby_whale_v4 tests
conda run -n base uv run ruff check baby_whale_v4 tests
conda run -n base uv run ty check baby_whale_v4
conda run -n base uv run python -m unittest discover -s tests
```

`ty` is scoped to `baby_whale_v4` until the tests are typed. Do not expand it to `tests` and then suppress errors broadly. Fix test typing explicitly when that becomes the goal.

Current `pyproject.toml` has these tool settings:

```toml
[tool.ruff]
target-version = "py314"
line-length = 100
src = ["baby_whale_v4", "tests"]

[dependency-groups]
dev = [
    "ruff",
    "ty",
]
```

For MLX-heavy modules, strict type checkers may still need local, documented ignores because array shapes and dtypes are runtime refinements. Rules:

- `# type: ignore[...]` must include the error code.
- Every ignore needs a short comment explaining why the checker cannot express the invariant.
- Never use file-wide ignores for project code.

Do not silently skip type checks in CI.

## 5. Boundary Validation Checklist

Use this for every public function/class.

- Inputs have precise type annotations.
- Closed-set strings use `Literal`, not plain `str`.
- Public dataclasses validate in `__post_init__`.
- Tensors are checked for rank, dtype, device, and semantic length at boundaries.
- Config objects reject unknown keys.
- Checkpoint and cache keys include every field that can affect semantics.
- `Optional` fields are justified as "not yet available", not hidden errors.
- `assert_never` closes every exhaustive branch.
- No public function returns `Any`.
- Any type ignore is local, coded, and explained.

## 6. ML-Specific Type Invariants

The checker cannot prove these, so tests and runtime checks must:

- `input_ids`: `mx.int32`, shape `[B, T]`, device equals the active MLX device.
- `targets`: same shape/device as `input_ids`, `-1` only for ignored positions.
- KV cache: per-layer key/value shapes match; the cache stores raw float K/V — there is no KV quantization mode because a Python round-trip on every append slows decode without a real hardware win.
- Generation: `prompt_len + max_new_tokens <= context_length`.
- Prefix cache key: includes config hash, tokenizer hash, backend, precision, weight quantization, layer schedule, and prefix content.
- Quantization: every active mode (`int8-weight`, `int4-weight`, `fp4-native`) routes through `mx.quantized_matmul`. No mode is allowed to alias to a Python quantize/dequantize shim.
- Training: gradient accumulation scales by examples/tokens, not by accidental microbatch count.
- Inference scheduler: decode cannot run before prefill, and completed requests cannot re-enter queues.

## 7. Patterns To Avoid

Avoid:

- `dict[str, Any]` flowing past a parser.
- plain `str` for backend, precision, attention, generation, or quantization mode.
- mutable config.
- unvalidated dataclass constructors.
- `Optional` chains where a state-specific dataclass would be clearer.
- `assert` for user/config validation. Use explicit exceptions because `assert` can be optimized out.
- using type hints instead of runtime checks for external input.
- catching broad exceptions and continuing with substituted behavior.
- defaulting to CPU or another backend when a requested backend fails.
- shape comments without executable checks or tests.

## 8. Code Review Questions

Ask these before merging:

1. What invalid state does this type make impossible?
2. What invalid state is still possible and where is it checked?
3. Which external boundary is this code crossing?
4. Does a new `Literal` variant force all dispatch sites to update?
5. Does checkpoint/cache identity include every semantic input?
6. Do tests cover a failing boundary, not only the happy path?
7. Are tensor shape/device/dtype assumptions executable?

## 9. Current Project Direction

The current code already uses the right core ideas:

- `Literal` aliases for backend/precision/attention/quant/generation modes.
- `assert_never` for exhaustive dispatch.
- `ensure_in` and `ensure_tensor` boundary helpers.
- `Ok[T]`, `Err[E]`, and `type Result[T, E] = Ok[T] | Err[E]`.
- frozen value objects such as `PrefixCacheKey`, `Message`, and `DPOExample`.
- `Protocol` for tokenizer behavior.
- dataclass `__post_init__` validation on major configs and request options.
- `TypedDict`, `ReadOnly`, and `TypeIs` for HTTP JSON payload parsing.
- `ConfigPayload` plus explicit field parsers for config payloads.
- `CheckpointPayload` plus explicit validation for pickle checkpoint payloads and MLX model-state arrays.
- `MLXFP4Mode = Literal["mxfp4", "nvfp4"]` plus frozen `MLXFP4Weight` validation for mandatory native MLX FP4 primitives.
- JSON-scalar metric records instead of `dict[str, Any]`.
- uv, Ruff, and ty configuration in `pyproject.toml`.

Next improvements:

- Introduce `NewType` for semantic IDs and hashes.
- Convert more state variants from `Optional` flags into explicit product types where state transitions get complex.
- Keep adding regression tests for every runtime validation rule.

## 10. Latest Python Typing Features

Researched on 2026-05-08. The project now runs on Python 3.14 through uv. Use stable Python 3.14 typing features directly from stdlib `typing`; keep Python 3.15 features out of the baseline until 3.15 is stable and local MLX, Ruff, ty, and unittest gates are clean.

| Feature | Python | What it gives us | Project decision |
| --- | --- | --- | --- |
| `type` alias statement | 3.12 | Native alias syntax: `type Result[T, E] = Ok[T] | Err[E]` | Use now. |
| Type parameter syntax | 3.12 | `class Box[T]: ...`, `def f[T](x: T) -> T: ...` | Use now for small generic helpers. |
| `@override` | 3.12 | Checker-visible override intent | Low priority because this project prefers protocols and little inheritance. |
| `ReadOnly` for `TypedDict` | 3.13 | Marks parsed JSON/config fields as read-only to checkers | Use now at IO boundaries. |
| `TypeIs` | 3.13 | Stronger narrowing than `TypeGuard` when validating unknown objects | Use now for JSON/config/checkpoint parsing. |
| Type parameter defaults | 3.13 | Defaults for generic types | Low priority. Useful only if project generics grow. |
| `warnings.deprecated` | 3.13 | Type-checker-visible deprecation | Low priority because the project allows API breaks instead of compatibility debt. |
| Deferred annotation evaluation | 3.14 | PEP 649/749 runtime annotation behavior and `annotationlib` | Use by removing `from __future__ import annotations`; only use `annotationlib` if introspection becomes real. |
| TypedDict `closed` / `extra_items` | 3.15 / future | Type-level unknown-key policy for dict payloads | Very aligned with fail-fast JSON/config parsing. Adopt when available through tooling. |
| `TypeForm` | 3.15 / future | Type expressions as first-class values for APIs that accept types | Not needed now. Could help future schema/plugin code. |
| `@disjoint_base` | 3.15 / future | Helps checkers reason that classes cannot overlap | Not needed now. Useful only if class variant hierarchies become complex. |

Immediate project use:

- Keep Python `>=3.14`.
- Use stdlib `typing.TypeIs`, `typing.ReadOnly`, and `typing.TypedDict`.
- Use `TypedDict` + `TypeIs` at external boundaries: HTTP JSON, checkpoint metadata, config dictionaries.
- Use Python 3.12+ syntax where it reduces boilerplate: `type` aliases, `class Box[T]`, and `def f[T](...)`.
- Do not make Python 3.15 the default yet. uv can see/download 3.15 pre-releases, but this project should stay on the latest stable Python that passes the full local MLX suite.
- Do not use new typing features as decoration. Each one must either reject bad input earlier, improve exhaustive dispatch, or remove a real `Any`.

## 11. Sources

- Python `typing` docs: https://docs.python.org/3/library/typing.html
- Python typing specification: https://typing.python.org/en/latest/spec/index.html
- Python `match` statement docs: https://docs.python.org/3/reference/compound_stmts.html#the-match-statement
- PEP 484, Type Hints: https://peps.python.org/pep-0484/
- PEP 634, Structural Pattern Matching: https://peps.python.org/pep-0634/
- PEP 649, Deferred Evaluation Of Annotations: https://peps.python.org/pep-0649/
- PEP 695, Type Parameter Syntax: https://peps.python.org/pep-0695/
- PEP 696, Type Defaults for Type Parameters: https://peps.python.org/pep-0696/
- PEP 702, Marking deprecations using the type system: https://peps.python.org/pep-0702/
- PEP 705, Read-only items in TypedDict: https://peps.python.org/pep-0705/
- PEP 728, TypedDict with Typed Extra Items: https://peps.python.org/pep-0728/
- PEP 742, Narrowing types with TypeIs: https://peps.python.org/pep-0742/
- PEP 747, TypeForm: https://peps.python.org/pep-0747/
- PEP 800, `disjoint_base`: https://peps.python.org/pep-0800/
- Python `dataclasses` docs: https://docs.python.org/3/library/dataclasses.html
- uv docs: https://docs.astral.sh/uv/
- Ruff docs: https://docs.astral.sh/ruff/
- ty docs: https://docs.astral.sh/ty/
