from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import cast, get_args

from baby_whale_v4.config._parse import (
    ConfigJsonValue,
    ConfigPayload,
    default_layer_schedule,
    parse_config_payload,
)
from baby_whale_v4.typing import (
    AttentionKind,
    Backend,
    ConfigHash,
    Precision,
    QuantMode,
    VisionEncoderKind,
    ensure_in,
)

_BACKENDS: tuple[Backend, ...] = get_args(Backend)
_PRECISIONS: tuple[Precision, ...] = get_args(Precision)
_ATTENTION: tuple[AttentionKind, ...] = get_args(AttentionKind)
_QUANT: tuple[QuantMode, ...] = get_args(QuantMode)
_VISION_ENCODERS: tuple[VisionEncoderKind, ...] = get_args(VisionEncoderKind)
_VISION_HASH_FIELDS = (
    "enable_vision",
    "vision_encoder_kind",
    "vision_tile_size",
    "vision_max_tiles",
    "vision_dim",
    "vision_dropout",
)


@dataclass
class BabyWhaleV4Config:
    name: str = "baby-whale-v4-30m"
    backend: Backend = "mlx"
    precision: Precision = "fp32"
    attention_impl: AttentionKind = "sliding_mqa"

    vocab_size: int = 32768
    context_length: int = 1024
    n_layer: int = 8
    n_embd: int = 384
    n_head: int = 6
    n_kv_head: int = 1
    sliding_window: int = 256
    rope_fraction: float = 0.25

    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    rms_norm_eps: float = 1e-5
    tie_weights: bool = True

    hc_mult: int = 1
    mtp_heads: int = 0

    n_expert: int = 8
    n_shared_expert: int = 1
    experts_per_token: int = 2
    n_hash_layers: int = 1
    moe_intermediate_size: int = 512
    swiglu_clamp: float = 30.0
    aux_free_bias_rate: float = 0.0

    layer_schedule: tuple[AttentionKind, ...] = field(default_factory=default_layer_schedule)
    hca_compress_rate: int = 16
    hca_block_size: int = 16
    csa_compress_rate: int = 4
    csa_block_size: int = 4
    csa_block_stride: int = 2
    csa_index_topk: int = 8
    kv_lora_rank: int = 64

    quant_mode: QuantMode = "none"
    activation_checkpoint: bool = False

    # Vision (DeepSeek-VL2 recipe, Step 8). Off by default; when disabled the
    # text-only path is bit-identical and these fields are excluded from
    # config_hash so pre-vision checkpoints keep their hash.
    enable_vision: bool = False
    vision_encoder_kind: VisionEncoderKind = "siglip"
    vision_tile_size: int = 384
    vision_max_tiles: int = 9
    vision_dim: int = 1152
    vision_dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.layer_schedule, list):
            self.layer_schedule = tuple(self.layer_schedule)
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def rotary_dim(self) -> int:
        dim = int(self.head_dim * self.rope_fraction)
        return dim - (dim % 2)

    @property
    def effective_layer_schedule(self) -> tuple[AttentionKind, ...]:
        if len(self.layer_schedule) == self.n_layer:
            return self.layer_schedule
        if self.layer_schedule == default_layer_schedule():
            return (self.attention_impl,) * self.n_layer
        raise ValueError(
            f"layer_schedule has {len(self.layer_schedule)} entries but n_layer={self.n_layer}"
        )

    def validate(self) -> None:
        ensure_in("backend", self.backend, _BACKENDS)
        ensure_in("precision", self.precision, _PRECISIONS)
        ensure_in("attention_impl", self.attention_impl, _ATTENTION)
        ensure_in("quant_mode", self.quant_mode, _QUANT)
        for kind in self.effective_layer_schedule:
            ensure_in("layer_schedule entry", kind, _ATTENTION)
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head for grouped/MQA expansion")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if self.sliding_window > self.context_length:
            raise ValueError("sliding_window cannot exceed context_length")
        if not (0.0 < self.rope_fraction <= 1.0):
            raise ValueError("rope_fraction must be in (0, 1]")
        if self.rotary_dim == 0:
            raise ValueError("rope_fraction is too small; rotary_dim would be zero")
        if self.hc_mult < 1:
            raise ValueError("hc_mult must be >= 1")
        if self.hc_mult > 4:
            raise ValueError("hc_mult > 4 is not supported in this educational stack")
        if self.mtp_heads < 0:
            raise ValueError("mtp_heads must be >= 0")
        if self.mtp_heads > 4:
            raise ValueError("mtp_heads > 4 is not supported in this educational stack")
        if self.n_expert <= 0:
            raise ValueError("n_expert must be positive")
        if self.n_shared_expert != 1:
            raise ValueError("only one shared expert is implemented in this educational slice")
        if not (1 <= self.experts_per_token <= self.n_expert):
            raise ValueError("experts_per_token must be in [1, n_expert]")
        if self.n_hash_layers < 0 or self.n_hash_layers > self.n_layer:
            raise ValueError("n_hash_layers must be in [0, n_layer]")
        if self.moe_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size must be positive")
        if self.aux_free_bias_rate < 0:
            raise ValueError("aux_free_bias_rate must be >= 0")
        if self.aux_free_bias_rate > 1.0:
            raise ValueError("aux_free_bias_rate must be <= 1.0; pick a small value (e.g. 1e-3)")
        if self.hca_compress_rate <= 0 or self.hca_block_size <= 0:
            raise ValueError("hca_compress_rate and hca_block_size must be positive")
        if self.csa_compress_rate <= 0 or self.csa_block_size <= 0:
            raise ValueError("csa_compress_rate and csa_block_size must be positive")
        if self.csa_block_stride <= 0 or self.csa_block_stride > self.csa_block_size:
            raise ValueError("csa_block_stride must be in [1, csa_block_size]")
        if self.csa_index_topk <= 0:
            raise ValueError("csa_index_topk must be positive")
        if self.kv_lora_rank <= 0:
            raise ValueError("kv_lora_rank must be positive")
        if "mla" in self.effective_layer_schedule and self.kv_lora_rank > self.n_embd:
            raise ValueError("kv_lora_rank cannot exceed n_embd for MLA layers")
        if (self.quant_mode == "fp4-expert" or self.quant_mode.startswith("fp4-native")) and (
            self.backend != "mlx"
        ):
            raise ValueError(
                f"{self.quant_mode} requires backend='mlx'; choose a supported MLX runtime separately"
            )
        if type(self.activation_checkpoint) is not bool:
            raise TypeError("activation_checkpoint must be a boolean")
        if type(self.enable_vision) is not bool:
            raise TypeError("enable_vision must be a boolean")
        ensure_in("vision_encoder_kind", self.vision_encoder_kind, _VISION_ENCODERS)
        if self.enable_vision:
            if self.vision_dim <= 0:
                raise ValueError("vision_dim must be positive when enable_vision is set")
            if self.vision_max_tiles < 1:
                raise ValueError("vision_max_tiles must be >= 1 when enable_vision is set")
            if self.vision_tile_size <= 0:
                raise ValueError("vision_tile_size must be positive when enable_vision is set")
            if not (0.0 <= self.vision_dropout < 1.0):
                raise ValueError("vision_dropout must be in [0, 1)")

    def to_dict(self) -> ConfigPayload:
        # Manual walk preserves tuple-typed `layer_schedule` (dataclasses.asdict
        # would coerce tuples to lists and break the config_hash signature).
        return cast(
            ConfigPayload,
            {f.name: getattr(self, f.name) for f in dataclasses.fields(self)},
        )

    def config_hash(self) -> ConfigHash:
        payload = dict(self.to_dict())
        if not self.enable_vision:
            # Text-only checkpoints predate the vision fields; excluding them when
            # disabled keeps the hash byte-identical so those checkpoints load.
            for key in _VISION_HASH_FIELDS:
                payload.pop(key, None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return ConfigHash(hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16])

    @classmethod
    def from_dict(cls, data: object) -> BabyWhaleV4Config:
        return cls(**parse_config_payload(data))

    @classmethod
    def tiny(cls, vocab_size: int = 128, context_length: int = 32) -> BabyWhaleV4Config:
        return cls(
            name="baby-whale-v4-tiny",
            vocab_size=vocab_size,
            context_length=context_length,
            n_layer=2,
            n_embd=64,
            n_head=4,
            n_kv_head=1,
            sliding_window=min(16, context_length),
            rope_fraction=0.5,
            n_expert=4,
            experts_per_token=2,
            n_hash_layers=1,
            moe_intermediate_size=128,
        )

    @classmethod
    def hybrid_tiny(
        cls,
        vocab_size: int = 128,
        context_length: int = 64,
        hc_mult: int = 1,
        mtp_heads: int = 0,
    ) -> BabyWhaleV4Config:
        schedule: tuple[AttentionKind, ...] = ("sliding_mqa", "hca", "csa", "sliding_mqa")
        return cls(
            name="baby-whale-v4-hybrid-tiny",
            vocab_size=vocab_size,
            context_length=context_length,
            n_layer=4,
            n_embd=64,
            n_head=4,
            n_kv_head=1,
            sliding_window=8,
            rope_fraction=0.5,
            n_expert=4,
            experts_per_token=2,
            n_hash_layers=1,
            moe_intermediate_size=64,
            layer_schedule=schedule,
            hca_compress_rate=4,
            hca_block_size=4,
            csa_compress_rate=2,
            csa_block_size=4,
            csa_block_stride=2,
            csa_index_topk=4,
            hc_mult=hc_mult,
            mtp_heads=mtp_heads,
        )

    @classmethod
    def mla_tiny(
        cls,
        vocab_size: int = 128,
        context_length: int = 64,
        kv_lora_rank: int = 16,
    ) -> BabyWhaleV4Config:
        schedule: tuple[AttentionKind, ...] = ("mla", "sliding_mqa", "mla", "sliding_mqa")
        return cls(
            name="baby-whale-v4-mla-tiny",
            vocab_size=vocab_size,
            context_length=context_length,
            n_layer=4,
            n_embd=64,
            n_head=4,
            n_kv_head=1,
            sliding_window=8,
            rope_fraction=0.5,
            n_expert=4,
            experts_per_token=2,
            n_hash_layers=1,
            moe_intermediate_size=64,
            layer_schedule=schedule,
            kv_lora_rank=kv_lora_rank,
        )


def config_for_inference(cfg: BabyWhaleV4Config) -> BabyWhaleV4Config:
    """Return ``cfg`` with ``activation_checkpoint=False``.

    Activation checkpointing rebuilds the forward graph from saved inputs and
    is incompatible with the mutable KV cache used during decode/rollout.
    Loading a training checkpoint for inference/RL must therefore override
    this single field. Going through ``dataclasses.replace`` (instead of a
    dict round-trip) keeps validation, type-checking, and frozen-ness intact.
    """
    return dataclasses.replace(cfg, activation_checkpoint=False)


__all__ = [
    "BabyWhaleV4Config",
    "ConfigJsonValue",
    "ConfigPayload",
    "config_for_inference",
    "parse_config_payload",
]
