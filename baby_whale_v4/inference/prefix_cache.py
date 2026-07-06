import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import mlx.core as mx

from baby_whale_v4.cache import DynamicKVCache
from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.device import active_runtime
from baby_whale_v4.typing import ConfigHash, MLXRuntime, TokenizerHash


@dataclass(frozen=True)
class PrefixCacheKey:
    prefix_hash: str
    config_hash: ConfigHash
    tokenizer_hash: TokenizerHash
    backend: str
    runtime: MLXRuntime
    precision: str
    quant_mode: str
    layer_schedule_hash: str

    def __post_init__(self) -> None:
        fields = (
            self.prefix_hash,
            self.config_hash,
            self.tokenizer_hash,
            self.backend,
            self.runtime,
            self.precision,
            self.quant_mode,
            self.layer_schedule_hash,
        )
        if any(not field for field in fields):
            raise ValueError("prefix cache key fields must be non-empty")

    @classmethod
    def build(
        cls,
        *,
        prefix_ids: list[int],
        config: BabyWhaleV4Config,
        tokenizer_hash: TokenizerHash,
        runtime: MLXRuntime | None = None,
    ) -> PrefixCacheKey:
        h_prefix = hashlib.sha256(b",".join(str(t).encode() for t in prefix_ids)).hexdigest()[:24]
        h_schedule = hashlib.sha256(",".join(config.effective_layer_schedule).encode()).hexdigest()[
            :12
        ]
        return cls(
            prefix_hash=h_prefix,
            config_hash=config.config_hash(),
            tokenizer_hash=tokenizer_hash,
            backend=config.backend,
            runtime=active_runtime() if runtime is None else runtime,
            precision=config.precision,
            quant_mode=config.quant_mode,
            layer_schedule_hash=h_schedule,
        )


@dataclass
class _Entry:
    n_tokens: int
    cache: DynamicKVCache
    last_logits: mx.array


class PrefixCache:
    def __init__(self, capacity: int = 16):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._entries: OrderedDict[PrefixCacheKey, _Entry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: PrefixCacheKey) -> tuple[int, DynamicKVCache, mx.array] | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry.n_tokens, _clone_cache(entry.cache), mx.array(entry.last_logits)

    def put(
        self,
        key: PrefixCacheKey,
        cache: DynamicKVCache,
        n_tokens: int,
        last_logits: mx.array,
    ) -> None:
        self._entries[key] = _Entry(
            n_tokens=n_tokens,
            cache=_clone_cache(cache),
            last_logits=mx.array(last_logits),
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0


def _clone_cache(src: DynamicKVCache) -> DynamicKVCache:
    return src.clone()
