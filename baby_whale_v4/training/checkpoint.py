import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, ReadOnly, TypedDict, TypeIs, cast

import mlx.core as mx

from baby_whale_v4.config import BabyWhaleV4Config, ConfigJsonValue
from baby_whale_v4.typing import ConfigHash

type ModelState = dict[str, object]
type OptimizerState = dict[str, object]
type SchedulerState = dict[str, object]
type RngState = dict[str, int]
type CheckpointExtraValue = str | int | float | bool | None
type CheckpointExtra = dict[str, CheckpointExtraValue]


class StateDictProvider(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...


class CheckpointPayload(TypedDict):
    config: ReadOnly[Mapping[str, ConfigJsonValue]]
    config_hash: ReadOnly[str]
    model_state: ReadOnly[ModelState]
    optimizer_state: ReadOnly[OptimizerState | None]
    scheduler_state: ReadOnly[SchedulerState | None]
    step: ReadOnly[int]
    rng_state: ReadOnly[RngState]
    extra: ReadOnly[CheckpointExtra]
    format_version: ReadOnly[Literal[2]]


_CHECKPOINT_KEYS = frozenset(
    {
        "config",
        "config_hash",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "step",
        "rng_state",
        "extra",
        "format_version",
    }
)


@dataclass
class Checkpoint:
    config: BabyWhaleV4Config
    model_state: ModelState
    optimizer_state: OptimizerState | None
    scheduler_state: SchedulerState | None
    step: int
    rng_state: RngState
    config_hash: ConfigHash
    extra: CheckpointExtra

    def __post_init__(self) -> None:
        if not isinstance(self.config, BabyWhaleV4Config):
            raise TypeError("checkpoint.config must be a BabyWhaleV4Config")
        if self.step < 0:
            raise ValueError("checkpoint step must be non-negative")
        if not self.config_hash:
            raise ValueError("checkpoint config_hash must be non-empty")
        if self.config_hash != self.config.config_hash():
            raise ValueError("checkpoint config_hash does not match config")


def _is_str_key_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def _copy_object_state(value: object, name: str) -> dict[str, object]:
    if not _is_str_key_mapping(value):
        raise TypeError(f"checkpoint.{name} must be an object with string keys")
    return {
        key: _copy_state_value(item, f"checkpoint.{name}[{key!r}]", allow_scalar=True)
        for key, item in value.items()
    }


def _copy_optional_object_state(value: object, name: str) -> dict[str, object] | None:
    if value is None:
        return None
    return _copy_object_state(value, name)


def _copy_state_value(value: object, name: str, *, allow_scalar: bool) -> object:
    if isinstance(value, mx.array):
        return value
    if _is_str_key_mapping(value):
        return {
            key: _copy_state_value(item, f"{name}.{key}", allow_scalar=allow_scalar)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _copy_state_value(item, f"{name}[{i}]", allow_scalar=allow_scalar)
            for i, item in enumerate(value)
        ]
    if allow_scalar and (value is None or type(value) in {str, int, float, bool}):
        return value
    raise TypeError(f"{name} must be an MLX array or nested array object")


def _copy_model_state(value: object) -> ModelState:
    if not _is_str_key_mapping(value):
        raise TypeError("checkpoint.model_state must be an object with string keys")
    out: ModelState = {}
    for key, item in value.items():
        out[key] = _copy_state_value(item, f"checkpoint.model_state[{key!r}]", allow_scalar=False)
    return out


def _copy_rng_state(value: object) -> RngState:
    if not _is_str_key_mapping(value):
        raise TypeError("checkpoint.rng_state must be an object with string keys")
    out: RngState = {}
    for key, item in value.items():
        if type(item) is not int:
            raise TypeError(f"checkpoint.rng_state[{key!r}] must be an integer")
        out[key] = item
    if "mlx_seed" not in out:
        raise ValueError("checkpoint.rng_state must contain an 'mlx_seed' entry")
    return out


def _copy_extra(value: object) -> CheckpointExtra:
    if value is None:
        return {}
    if not _is_str_key_mapping(value):
        raise TypeError("checkpoint.extra must be an object with string keys")
    out: CheckpointExtra = {}
    for key, item in value.items():
        if item is not None and type(item) not in {str, int, float, bool}:
            raise TypeError(f"checkpoint.extra[{key!r}] must be JSON-scalar")
        out[key] = cast(CheckpointExtraValue, item)
    return out


def _parse_checkpoint_payload(raw: object) -> CheckpointPayload:
    if not _is_str_key_mapping(raw):
        raise TypeError("checkpoint payload must be an object with string keys")
    unknown = set(raw) - _CHECKPOINT_KEYS
    if unknown:
        raise ValueError(f"unknown checkpoint keys: {sorted(unknown)}")
    missing = _CHECKPOINT_KEYS - set(raw)
    if missing:
        raise ValueError(f"missing checkpoint keys: {sorted(missing)}")

    format_version = raw["format_version"]
    if format_version != 2:
        raise ValueError(f"unsupported checkpoint format_version {format_version!r}")
    config = raw["config"]
    if not _is_str_key_mapping(config):
        raise TypeError("checkpoint.config must be an object with string keys")
    config_hash = raw["config_hash"]
    if not isinstance(config_hash, str):
        raise TypeError("checkpoint.config_hash must be a string")
    step = raw["step"]
    if type(step) is not int:
        raise TypeError("checkpoint.step must be an integer")

    return {
        "config": cast(Mapping[str, ConfigJsonValue], config),
        "config_hash": config_hash,
        "model_state": _copy_model_state(raw["model_state"]),
        "optimizer_state": _copy_optional_object_state(raw["optimizer_state"], "optimizer_state"),
        "scheduler_state": _copy_optional_object_state(raw["scheduler_state"], "scheduler_state"),
        "step": step,
        "rng_state": _copy_rng_state(raw["rng_state"]),
        "extra": _copy_extra(raw["extra"]),
        "format_version": 2,
    }


def save_checkpoint(
    path: Path | str,
    *,
    config: BabyWhaleV4Config,
    model: StateDictProvider,
    optimizer: object | None,
    scheduler: StateDictProvider | None,
    step: int,
    seed: int = 0,
    extra: Mapping[str, CheckpointExtraValue] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    optimizer_state = None
    if optimizer is not None:
        state_dict = getattr(optimizer, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("optimizer must expose state_dict()")
        optimizer_state = state_dict()
    payload: CheckpointPayload = {
        "config": cast(Mapping[str, ConfigJsonValue], config.to_dict()),
        "config_hash": config.config_hash(),
        "model_state": _copy_model_state(model.state_dict()),
        "optimizer_state": _copy_optional_object_state(optimizer_state, "optimizer_state"),
        "scheduler_state": _copy_optional_object_state(
            scheduler.state_dict() if scheduler is not None else None,
            "scheduler_state",
        ),
        "step": int(step),
        "rng_state": {"mlx_seed": int(seed)},
        "extra": _copy_extra(extra),
        "format_version": 2,
    }
    out.write_bytes(pickle.dumps(payload))
    return out


def load_checkpoint(
    path: Path | str,
    *,
    expected_config_hash: ConfigHash | None = None,
) -> Checkpoint:
    raw_payload = pickle.loads(Path(path).read_bytes())
    payload = _parse_checkpoint_payload(raw_payload)
    config = BabyWhaleV4Config.from_dict(payload["config"])
    config_hash = ConfigHash(payload["config_hash"])
    if expected_config_hash is not None and expected_config_hash != config_hash:
        raise ValueError(
            f"checkpoint config hash mismatch: got {config_hash}, expected {expected_config_hash}"
        )
    if config.config_hash() != config_hash:
        raise ValueError("checkpoint config_hash does not match recomputed hash; corruption?")
    return Checkpoint(
        config=config,
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
        step=payload["step"],
        rng_state=payload["rng_state"],
        config_hash=config_hash,
        extra=payload["extra"],
    )
