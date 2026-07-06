import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np

from baby_whale_v4.cache import DynamicKVCache


@dataclass(frozen=True)
class KVOffloadReport:
    path: Path
    manifest_path: Path
    n_layer: int
    present_layers: int
    sequence_length: int
    bytes_written: int

    def __post_init__(self) -> None:
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.present_layers < 0 or self.present_layers > self.n_layer:
            raise ValueError("present_layers must be in [0, n_layer]")
        if self.sequence_length < 0:
            raise ValueError("sequence_length must be non-negative")
        if self.bytes_written <= 0:
            raise ValueError("bytes_written must be positive")


def save_kv_cache_npz(cache: DynamicKVCache, path: Path | str) -> KVOffloadReport:
    if not isinstance(cache, DynamicKVCache):
        raise TypeError("cache must be a DynamicKVCache")
    out = Path(path)
    # Mirror np.savez_compressed's ".npz" auto-append so out.stat() and the
    # manifest reference the file numpy actually writes (same foot-gun as
    # data/packing.py).
    if out.suffix != ".npz":
        out = out.with_name(out.name + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    present_kv: list[int] = []
    present_latent: list[int] = []
    latents = cache.latents or [None] * len(cache.keys)
    for idx, (key, value, latent) in enumerate(zip(cache.keys, cache.values, latents, strict=True)):
        if key is not None or value is not None:
            if key is None or value is None:
                raise ValueError(f"cache layer {idx} has incomplete key/value pair")
            arrays[f"k_{idx}"] = _to_numpy_float32(key, ndim=4)
            arrays[f"v_{idx}"] = _to_numpy_float32(value, ndim=4)
            present_kv.append(idx)
        if latent is not None:
            arrays[f"latent_{idx}"] = _to_numpy_float32(latent, ndim=3)
            present_latent.append(idx)
    if not arrays:
        raise ValueError("cannot offload an empty KV cache")
    np.savez_compressed(out, **cast(Any, arrays))
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest = {
        "format": "baby_whale_v4_kv_cache_npz_v2",
        "n_layer": len(cache.keys),
        "present_kv": present_kv,
        "present_latent": present_latent,
        "sequence_length": cache.max_sequence_length(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return KVOffloadReport(
        path=out,
        manifest_path=manifest_path,
        n_layer=len(cache.keys),
        present_layers=len(set(present_kv) | set(present_latent)),
        sequence_length=cache.max_sequence_length(),
        bytes_written=out.stat().st_size,
    )


def load_kv_cache_npz(path: Path | str, *, expected_n_layer: int) -> DynamicKVCache:
    if expected_n_layer <= 0:
        raise ValueError("expected_n_layer must be positive")
    src = Path(path)
    if src.suffix != ".npz":
        src = src.with_name(src.name + ".npz")
    manifest_path = src.with_suffix(src.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("KV cache manifest must be an object")
    if manifest.get("format") != "baby_whale_v4_kv_cache_npz_v2":
        raise ValueError(f"unsupported KV cache format {manifest.get('format')!r}")
    n_layer = manifest.get("n_layer")
    if n_layer != expected_n_layer:
        raise ValueError(
            f"KV cache layer count mismatch: got {n_layer}, expected {expected_n_layer}"
        )
    present_kv = _read_layer_list(manifest, "present_kv")
    present_latent = _read_layer_list(manifest, "present_latent")
    cache = DynamicKVCache.empty(expected_n_layer)
    with np.load(src) as data:
        for idx in present_kv:
            key_name = f"k_{idx}"
            value_name = f"v_{idx}"
            if key_name not in data or value_name not in data:
                raise ValueError(f"KV cache file missing layer {idx}")
            key = mx.array(data[key_name])
            value = mx.array(data[value_name])
            if key.shape != value.shape:
                raise ValueError(f"KV cache layer {idx} key/value shape mismatch")
            if key.ndim != 4:
                raise ValueError(f"KV cache layer {idx} key/value tensors must be rank-4")
            cache.keys[idx] = key
            cache.values[idx] = value
        latents = cache.latents
        if latents is None:
            raise RuntimeError("DynamicKVCache.latents was not initialized")
        for idx in present_latent:
            latent_name = f"latent_{idx}"
            if latent_name not in data:
                raise ValueError(f"KV cache file missing latent layer {idx}")
            latent = mx.array(data[latent_name])
            if latent.ndim != 3:
                raise ValueError(f"KV cache latent layer {idx} tensor must be rank-3")
            latents[idx] = latent
    return cache


def _read_layer_list(manifest: dict, key: str) -> list[int]:
    value = manifest.get(key)
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise TypeError(f"KV cache manifest {key} must be a list of layer integers")
    return cast(list[int], value)


def _to_numpy_float32(value: mx.array, *, ndim: int) -> np.ndarray:
    if not isinstance(value, mx.array):
        raise TypeError("KV cache tensors must be MLX arrays")
    if value.ndim != ndim:
        raise ValueError(f"KV cache tensors must be rank-{ndim}")
    return np.array(value.tolist(), dtype=np.float32)
