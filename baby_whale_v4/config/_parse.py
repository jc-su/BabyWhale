from collections.abc import Mapping, Sequence
from typing import ReadOnly, TypedDict, TypeIs, cast, get_args

from baby_whale_v4.typing import (
    AttentionKind,
    Backend,
    Precision,
    QuantMode,
    VisionEncoderKind,
)

_BACKENDS: tuple[Backend, ...] = get_args(Backend)
_PRECISIONS: tuple[Precision, ...] = get_args(Precision)
_ATTENTION: tuple[AttentionKind, ...] = get_args(AttentionKind)
_QUANT: tuple[QuantMode, ...] = get_args(QuantMode)
_VISION_ENCODERS: tuple[VisionEncoderKind, ...] = get_args(VisionEncoderKind)

_CONFIG_KEYS = frozenset(
    {
        "name",
        "backend",
        "precision",
        "attention_impl",
        "vocab_size",
        "context_length",
        "n_layer",
        "n_embd",
        "n_head",
        "n_kv_head",
        "sliding_window",
        "rope_fraction",
        "resid_pdrop",
        "attn_pdrop",
        "embd_pdrop",
        "rms_norm_eps",
        "tie_weights",
        "hc_mult",
        "mtp_heads",
        "n_expert",
        "n_shared_expert",
        "experts_per_token",
        "n_hash_layers",
        "moe_intermediate_size",
        "swiglu_clamp",
        "aux_free_bias_rate",
        "layer_schedule",
        "hca_compress_rate",
        "hca_block_size",
        "csa_compress_rate",
        "csa_block_size",
        "csa_block_stride",
        "csa_index_topk",
        "kv_lora_rank",
        "quant_mode",
        "activation_checkpoint",
        "enable_vision",
        "vision_encoder_kind",
        "vision_tile_size",
        "vision_max_tiles",
        "vision_dim",
        "vision_dropout",
    }
)

type ConfigJsonValue = str | int | float | bool | tuple[AttentionKind, ...]


class ConfigPayload(TypedDict, total=False):
    name: ReadOnly[str]
    backend: ReadOnly[Backend]
    precision: ReadOnly[Precision]
    attention_impl: ReadOnly[AttentionKind]
    vocab_size: ReadOnly[int]
    context_length: ReadOnly[int]
    n_layer: ReadOnly[int]
    n_embd: ReadOnly[int]
    n_head: ReadOnly[int]
    n_kv_head: ReadOnly[int]
    sliding_window: ReadOnly[int]
    rope_fraction: ReadOnly[float]
    resid_pdrop: ReadOnly[float]
    attn_pdrop: ReadOnly[float]
    embd_pdrop: ReadOnly[float]
    rms_norm_eps: ReadOnly[float]
    tie_weights: ReadOnly[bool]
    hc_mult: ReadOnly[int]
    mtp_heads: ReadOnly[int]
    n_expert: ReadOnly[int]
    n_shared_expert: ReadOnly[int]
    experts_per_token: ReadOnly[int]
    n_hash_layers: ReadOnly[int]
    moe_intermediate_size: ReadOnly[int]
    swiglu_clamp: ReadOnly[float]
    aux_free_bias_rate: ReadOnly[float]
    layer_schedule: ReadOnly[tuple[AttentionKind, ...]]
    hca_compress_rate: ReadOnly[int]
    hca_block_size: ReadOnly[int]
    csa_compress_rate: ReadOnly[int]
    csa_block_size: ReadOnly[int]
    csa_block_stride: ReadOnly[int]
    csa_index_topk: ReadOnly[int]
    kv_lora_rank: ReadOnly[int]
    quant_mode: ReadOnly[QuantMode]
    activation_checkpoint: ReadOnly[bool]
    enable_vision: ReadOnly[bool]
    vision_encoder_kind: ReadOnly[VisionEncoderKind]
    vision_tile_size: ReadOnly[int]
    vision_max_tiles: ReadOnly[int]
    vision_dim: ReadOnly[int]
    vision_dropout: ReadOnly[float]


def default_layer_schedule() -> tuple[AttentionKind, ...]:
    return ("sliding_mqa",) * 8


def _is_str_key_mapping(value: object) -> TypeIs[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def _read_str(data: Mapping[str, object], key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"config.{key} must be a string")
    return value


def _read_int(data: Mapping[str, object], key: str, default: int) -> int:
    if key not in data:
        return default
    value = data[key]
    if type(value) is not int:
        raise TypeError(f"config.{key} must be an integer")
    return value


def _read_float(data: Mapping[str, object], key: str, default: float) -> float:
    if key not in data:
        return default
    value = data[key]
    if type(value) is int:
        return float(value)
    if type(value) is not float:
        raise TypeError(f"config.{key} must be a number")
    return value


def _read_bool(data: Mapping[str, object], key: str, default: bool) -> bool:
    if key not in data:
        return default
    value = data[key]
    if type(value) is not bool:
        raise TypeError(f"config.{key} must be a boolean")
    return value


def _read_choice[T](
    data: Mapping[str, object],
    key: str,
    default: T,
    choices: tuple[T, ...],
) -> T:
    if key not in data:
        return default
    value = data[key]
    if value not in choices:
        raise ValueError(f"unsupported config.{key} {value!r}; supported: {list(choices)}")
    return cast(T, value)


def _read_layer_schedule(data: Mapping[str, object]) -> tuple[AttentionKind, ...]:
    if "layer_schedule" not in data:
        return default_layer_schedule()
    value = data["layer_schedule"]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError("config.layer_schedule must be a sequence of attention kinds")
    schedule: list[AttentionKind] = []
    for item in value:
        if item not in _ATTENTION:
            raise ValueError(
                f"unsupported config.layer_schedule entry {item!r}; supported: {list(_ATTENTION)}"
            )
        schedule.append(cast(AttentionKind, item))
    return tuple(schedule)


def parse_config_payload(data: object) -> ConfigPayload:
    if not _is_str_key_mapping(data):
        raise TypeError("config payload must be an object with string keys")
    unknown = set(data) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return {
        "name": _read_str(data, "name", "baby-whale-v4-30m"),
        "backend": _read_choice(data, "backend", "mlx", _BACKENDS),
        "precision": _read_choice(data, "precision", "fp32", _PRECISIONS),
        "attention_impl": _read_choice(data, "attention_impl", "sliding_mqa", _ATTENTION),
        "vocab_size": _read_int(data, "vocab_size", 32768),
        "context_length": _read_int(data, "context_length", 1024),
        "n_layer": _read_int(data, "n_layer", 8),
        "n_embd": _read_int(data, "n_embd", 384),
        "n_head": _read_int(data, "n_head", 6),
        "n_kv_head": _read_int(data, "n_kv_head", 1),
        "sliding_window": _read_int(data, "sliding_window", 256),
        "rope_fraction": _read_float(data, "rope_fraction", 0.25),
        "resid_pdrop": _read_float(data, "resid_pdrop", 0.0),
        "attn_pdrop": _read_float(data, "attn_pdrop", 0.0),
        "embd_pdrop": _read_float(data, "embd_pdrop", 0.0),
        "rms_norm_eps": _read_float(data, "rms_norm_eps", 1e-5),
        "tie_weights": _read_bool(data, "tie_weights", True),
        "hc_mult": _read_int(data, "hc_mult", 1),
        "mtp_heads": _read_int(data, "mtp_heads", 0),
        "n_expert": _read_int(data, "n_expert", 8),
        "n_shared_expert": _read_int(data, "n_shared_expert", 1),
        "experts_per_token": _read_int(data, "experts_per_token", 2),
        "n_hash_layers": _read_int(data, "n_hash_layers", 1),
        "moe_intermediate_size": _read_int(data, "moe_intermediate_size", 512),
        "swiglu_clamp": _read_float(data, "swiglu_clamp", 30.0),
        "aux_free_bias_rate": _read_float(data, "aux_free_bias_rate", 0.0),
        "layer_schedule": _read_layer_schedule(data),
        "hca_compress_rate": _read_int(data, "hca_compress_rate", 16),
        "hca_block_size": _read_int(data, "hca_block_size", 16),
        "csa_compress_rate": _read_int(data, "csa_compress_rate", 4),
        "csa_block_size": _read_int(data, "csa_block_size", 4),
        "csa_block_stride": _read_int(data, "csa_block_stride", 2),
        "csa_index_topk": _read_int(data, "csa_index_topk", 8),
        "kv_lora_rank": _read_int(data, "kv_lora_rank", 64),
        "quant_mode": _read_choice(data, "quant_mode", "none", _QUANT),
        "activation_checkpoint": _read_bool(data, "activation_checkpoint", False),
        "enable_vision": _read_bool(data, "enable_vision", False),
        "vision_encoder_kind": _read_choice(
            data, "vision_encoder_kind", "siglip", _VISION_ENCODERS
        ),
        "vision_tile_size": _read_int(data, "vision_tile_size", 384),
        "vision_max_tiles": _read_int(data, "vision_max_tiles", 9),
        "vision_dim": _read_int(data, "vision_dim", 1152),
        "vision_dropout": _read_float(data, "vision_dropout", 0.0),
    }
