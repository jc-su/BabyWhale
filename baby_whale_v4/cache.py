from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

type KVPair = tuple[mx.array, mx.array]


class KVCache(Protocol):
    """Structural interface the model reads/writes during attention.

    Both the dense :class:`DynamicKVCache` and the paged
    ``baby_whale_v4.inference.paged_kv.PagedKVCache`` satisfy it, so the model
    forward is agnostic to KV storage. MLA layers additionally use the latent
    methods; the paged cache raises there (it has no latent path).
    """

    def sequence_length(self, layer_idx: int) -> int: ...

    def append(
        self, layer_idx: int, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array]: ...

    def max_sequence_length(self) -> int: ...

    def latent_length(self, layer_idx: int) -> int: ...

    def append_latent(self, layer_idx: int, latent: mx.array) -> mx.array: ...


@dataclass(frozen=True)
class KVCacheStats:
    n_layer: int
    present_layers: int
    max_sequence_length: int
    bytes: int

    def __post_init__(self) -> None:
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.present_layers < 0 or self.present_layers > self.n_layer:
            raise ValueError("present_layers must be in [0, n_layer]")
        if self.max_sequence_length < 0:
            raise ValueError("max_sequence_length must be non-negative")
        if self.bytes < 0:
            raise ValueError("bytes must be non-negative")


@dataclass
class DynamicKVCache:
    keys: list[mx.array | None]
    values: list[mx.array | None]
    latents: list[mx.array | None] | None = None

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values):
            raise ValueError("keys and values must have the same number of layers")
        if self.latents is None:
            self.latents = [None] * len(self.keys)
        elif len(self.latents) != len(self.keys):
            raise ValueError("latents must have the same number of layers as keys")

    @classmethod
    def empty(cls, n_layer: int) -> DynamicKVCache:
        return cls(
            keys=[None] * n_layer,
            values=[None] * n_layer,
            latents=[None] * n_layer,
        )

    def sequence_length(self, layer_idx: int) -> int:
        key = self.keys[layer_idx]
        if key is not None:
            return int(key.shape[2])
        latent = (self.latents or [None])[layer_idx]
        if latent is None:
            return 0
        return int(latent.shape[1])

    def latent_length(self, layer_idx: int) -> int:
        latent = (self.latents or [None])[layer_idx]
        return 0 if latent is None else int(latent.shape[1])

    def append_latent(self, layer_idx: int, latent: mx.array) -> mx.array:
        if latent.ndim != 3:
            raise ValueError("MLA latent tensors must have shape [B, T, R]")
        latents = self.latents
        if latents is None:
            raise RuntimeError("DynamicKVCache.latents was not initialized")
        old = latents[layer_idx]
        new = latent if old is None else mx.concatenate([old, latent], axis=1)
        latents[layer_idx] = new
        return new

    def max_sequence_length(self) -> int:
        return max((self.sequence_length(i) for i in range(len(self.keys))), default=0)

    def clone(self) -> DynamicKVCache:
        latents = self.latents or [None] * len(self.keys)
        return DynamicKVCache(
            keys=[None if key is None else mx.array(key) for key in self.keys],
            values=[None if value is None else mx.array(value) for value in self.values],
            latents=[None if latent is None else mx.array(latent) for latent in latents],
        )

    def stats(self) -> KVCacheStats:
        total_bytes = 0
        present = 0
        latents = self.latents or [None] * len(self.keys)
        for key, value, latent in zip(self.keys, self.values, latents, strict=True):
            if key is None and value is None and latent is None:
                continue
            if (key is None or value is None) and (key is not None or value is not None):
                raise ValueError("cache has incomplete key/value pair")
            if latent is not None and latent.ndim != 3:
                raise ValueError("cache latent tensors must have shape [B, T, R]")
            present += 1
            if key is not None and value is not None:
                total_bytes += _array_nbytes(key) + _array_nbytes(value)
            if latent is not None:
                total_bytes += _array_nbytes(latent)
        return KVCacheStats(
            n_layer=len(self.keys),
            present_layers=present,
            max_sequence_length=self.max_sequence_length(),
            bytes=total_bytes,
        )

    def append(self, layer_idx: int, key: mx.array, value: mx.array) -> KVPair:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("KV tensors must have shape [B, H, T, D]")
        if key.shape != value.shape:
            raise ValueError("key and value tensors must have identical shapes")
        old_key = self.keys[layer_idx]
        old_value = self.values[layer_idx]
        if old_key is None:
            if old_value is not None:
                raise ValueError("cache layer has value without key")
            new_key = key
            new_value = value
        else:
            if old_value is None:
                raise ValueError("cache layer has key without value")
            new_key = mx.concatenate([old_key, key], axis=2)
            new_value = mx.concatenate([old_value, value], axis=2)

        self.keys[layer_idx] = new_key
        self.values[layer_idx] = new_value
        return new_key, new_value


def _array_nbytes(x: mx.array) -> int:
    return int(x.size) * _dtype_nbytes(str(x.dtype))


def _dtype_nbytes(dtype: str) -> int:
    if "float64" in dtype or "int64" in dtype:
        return 8
    if "float32" in dtype or "int32" in dtype:
        return 4
    if "bfloat16" in dtype or "float16" in dtype or "int16" in dtype:
        return 2
    if "int8" in dtype or "uint8" in dtype or "bool" in dtype:
        return 1
    raise ValueError(f"unsupported dtype size for {dtype}")
