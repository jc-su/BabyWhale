"""vLLM-style PagedAttention KV storage (educational).

The PagedAttention contribution (Kwon et al., SOSP 2023) is twofold:

1. **Block-allocated KV storage.** A global pool of fixed-size KV blocks
   indexed by per-request *page tables*, replacing contiguous per-request
   tensors. This eliminates fragmentation when sequence lengths vary, lets
   multiple requests share a prefix's blocks, and bounds the per-token
   memory cost to block-aligned granularity.

2. **Paged-attention kernel.** A fused attention kernel that gathers K/V
   from the block pool through the page table without materializing the
   contiguous K/V tensor — that's where the production wall-clock win
   comes from on large models.

This module implements **(1) faithfully** and **(2) educationally**: we
gather blocks to a dense ``[B, H, T, D]`` tensor on demand and call our
existing attention path. That preserves the page-table indirection and
block-allocation semantics — the parts a student needs to see — without
requiring a custom Metal/CUDA kernel.

At our scale (ctx=384, single Mac, no memory pressure) this offers no
performance win. It exists for two reasons:

* **Pedagogy.** PagedAttention is the canonical example of "treat KV like
  virtual memory" — understanding it is required reading for anyone
  building production inference infra.
* **Future headroom.** When context grows (1M-token agent traces, long
  multi-turn conversations), block allocation makes the difference
  between O(N²)-wasteful padding and O(blocks_used) actual cost.

See ``refs/INFERENCE_OPTIMIZATIONS.md`` for the cost/benefit notes against
PagedAttention's other claims at our scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from baby_whale_v4.config import BabyWhaleV4Config


def _dtype_for_precision(precision: str) -> mx.Dtype:
    match precision:
        case "fp32":
            return mx.float32
        case "fp16":
            return mx.float16
        case "bf16":
            return mx.bfloat16
        case _:
            raise ValueError(f"unsupported precision {precision!r}")


_MLA_UNSUPPORTED = (
    "PagedKVCache has no MLA latent path; build the Engine with a DynamicKVCache "
    "(paged_pool=None) for models whose layer schedule contains 'mla'"
)


@dataclass(frozen=True)
class PagedKVConfig:
    """Static shape parameters for a :class:`PagedKVPool`.

    ``block_size`` is the number of tokens per block (vLLM defaults to 16);
    ``n_blocks`` is the pool capacity. A pool of ``n_blocks=64`` with
    ``block_size=16`` stores up to 1024 cached tokens across all requests.
    """

    n_layer: int
    n_heads: int
    head_dim: int
    block_size: int = 16
    n_blocks: int = 64
    dtype: mx.Dtype = mx.float32

    def __post_init__(self) -> None:
        for name, value in (
            ("n_layer", self.n_layer),
            ("n_heads", self.n_heads),
            ("head_dim", self.head_dim),
            ("block_size", self.block_size),
            ("n_blocks", self.n_blocks),
        ):
            if value <= 0:
                raise ValueError(f"PagedKVConfig.{name} must be positive")

    @classmethod
    def from_model_config(
        cls,
        config: BabyWhaleV4Config,
        *,
        block_size: int = 16,
        n_blocks: int = 64,
    ) -> PagedKVConfig:
        """Build a pool config matching ``config``.

        The attention cache stores K/V at ``n_kv_head`` heads — GQA/MQA
        expansion happens *after* the cache read — so the pool's head count is
        ``config.n_kv_head``, not ``config.n_head``. dtype follows the model
        precision so scattered K/V match the pre-zeroed pool blocks.
        """
        return cls(
            n_layer=config.n_layer,
            n_heads=config.n_kv_head,
            head_dim=config.head_dim,
            block_size=block_size,
            n_blocks=n_blocks,
            dtype=_dtype_for_precision(config.precision),
        )


class PageTable:
    """Per-request block map plus each layer's written token count.

    ``blocks`` is the logical→physical block map (list index = logical block
    number, value = physical block index in the pool). It is **shared across
    all layers**: in the vLLM model, token position ``t`` lives at the same
    ``(block, offset)`` in every layer, and each layer keeps its own physical
    KV in the pool. ``layer_lengths`` records how many tokens each layer has
    written, so the lock-step layers grow the shared block map exactly once per
    new logical block (the first layer to reach it allocates; the rest reuse).
    """

    def __init__(self) -> None:
        self.blocks: list[int] = []
        self.layer_lengths: dict[int, int] = {}

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def layer_length(self, layer_idx: int) -> int:
        """Tokens written into ``layer_idx`` for this request."""
        return self.layer_lengths.get(layer_idx, 0)

    @property
    def length(self) -> int:
        """Max tokens written across layers.

        Layers advance in lock-step during a full forward, so after each
        forward every layer shares this value; it is also the number of logical
        tokens the shared block map covers.
        """
        return max(self.layer_lengths.values(), default=0)

    def last_block_offset(self, block_size: int) -> int:
        """Token offset within the *last* block (0 means the block is empty)."""
        if self.length == 0:
            return 0
        return self.length % block_size


class PagedKVPool:
    """Global pool of KV blocks shared across requests.

    Layout per layer:
        keys  [n_blocks, n_heads, block_size, head_dim]
        values[n_blocks, n_heads, block_size, head_dim]

    A request's K/V for a given token at position ``t`` lives at
    ``keys[blocks[t // block_size], :, t % block_size, :]``.
    """

    def __init__(self, config: PagedKVConfig) -> None:
        self.config = config
        # Pre-allocated K/V blocks (zero-initialized). One [B, H, T, D]
        # tensor per layer — the first dim is *block index*, not batch.
        self._keys: list[mx.array] = [
            mx.zeros(
                (config.n_blocks, config.n_heads, config.block_size, config.head_dim),
                dtype=config.dtype,
            )
            for _ in range(config.n_layer)
        ]
        self._values: list[mx.array] = [
            mx.zeros(
                (config.n_blocks, config.n_heads, config.block_size, config.head_dim),
                dtype=config.dtype,
            )
            for _ in range(config.n_layer)
        ]
        # Free list: indices of unused blocks.
        self._free: list[int] = list(range(config.n_blocks))
        self._allocated: set[int] = set()

    @property
    def n_free(self) -> int:
        return len(self._free)

    @property
    def n_allocated(self) -> int:
        return len(self._allocated)

    def allocate(self) -> int:
        """Pop a free block index. Raises if the pool is exhausted."""
        if not self._free:
            raise RuntimeError(
                f"PagedKVPool exhausted (capacity={self.config.n_blocks}); "
                "either raise n_blocks or free pages from finished requests"
            )
        idx = self._free.pop()
        self._allocated.add(idx)
        return idx

    def free(self, block_idx: int) -> None:
        """Return a block to the free list. Idempotent for already-freed indices."""
        if block_idx not in self._allocated:
            return
        self._allocated.discard(block_idx)
        self._free.append(block_idx)

    def free_table(self, table: PageTable) -> None:
        """Free every block this page table held, and zero its bookkeeping."""
        for b in table.blocks:
            self.free(b)
        table.blocks = []
        table.layer_lengths = {}

    # ---- KV write -------------------------------------------------------

    def append_tokens(
        self,
        layer_idx: int,
        table: PageTable,
        keys: mx.array,
        values: mx.array,
    ) -> None:
        """Write ``keys``/``values`` for ``T`` new tokens to ``table``.

        Shapes: ``keys`` and ``values`` are ``[1, H, T, D]``. New blocks are
        allocated as needed to fit ``T`` tokens after ``table.length``.
        """
        if keys.shape != values.shape:
            raise ValueError("keys and values must share shape")
        if keys.ndim != 4 or keys.shape[0] != 1:
            raise ValueError("paged append expects [1, H, T, D] tensors")
        H, T, D = keys.shape[1], keys.shape[2], keys.shape[3]
        cfg = self.config
        if cfg.n_heads != H or cfg.head_dim != D:
            raise ValueError(
                f"K/V shape mismatch vs config: got [1,{H},{T},{D}], "
                f"expected H={cfg.n_heads}, D={cfg.head_dim}"
            )
        # Walk ``T`` new tokens through the table, allocating blocks as we
        # cross block boundaries. The block map is shared across layers, so a
        # block allocated while writing layer 0 is reused by layers 1..n at the
        # same logical position — only the physical KV in ``_keys[layer_idx]``
        # differs per layer.
        start = table.layer_lengths.get(layer_idx, 0)
        for t in range(T):
            pos = start + t
            block_pos = pos // cfg.block_size
            offset = pos % cfg.block_size
            if block_pos >= len(table.blocks):
                table.blocks.append(self.allocate())
            block_idx = table.blocks[block_pos]
            # Slice update: keys[block_idx, :, offset, :] = keys_in[0, :, t, :]
            self._keys[layer_idx] = _scatter_token(
                self._keys[layer_idx], block_idx, offset, keys[0, :, t, :]
            )
            self._values[layer_idx] = _scatter_token(
                self._values[layer_idx], block_idx, offset, values[0, :, t, :]
            )
        table.layer_lengths[layer_idx] = start + T

    # ---- KV read --------------------------------------------------------

    def gather(self, layer_idx: int, table: PageTable) -> tuple[mx.array, mx.array]:
        """Materialize the contiguous K/V for ``table`` at this layer.

        Returns ``(keys, values)`` of shape ``[1, H, length, D]`` —
        identical to what :class:`DynamicKVCache` would store inline.
        Production paged-attention fuses this gather into the attention
        kernel; we do it as a dense materialization step for clarity.
        """
        cfg = self.config
        length = table.layer_length(layer_idx)
        if length == 0:
            empty = mx.zeros((1, cfg.n_heads, 0, cfg.head_dim), dtype=cfg.dtype)
            return empty, empty
        block_arrs_k: list[mx.array] = []
        block_arrs_v: list[mx.array] = []
        full_blocks = length // cfg.block_size
        leftover = length - full_blocks * cfg.block_size
        for i in range(full_blocks):
            b = table.blocks[i]
            block_arrs_k.append(self._keys[layer_idx][b : b + 1])  # [1, H, BS, D]
            block_arrs_v.append(self._values[layer_idx][b : b + 1])
        if leftover > 0:
            b = table.blocks[full_blocks]
            block_arrs_k.append(self._keys[layer_idx][b : b + 1, :, :leftover, :])
            block_arrs_v.append(self._values[layer_idx][b : b + 1, :, :leftover, :])
        # Concatenate along the token dim. Each block contributes [1, H, T_block, D].
        keys = mx.concatenate(block_arrs_k, axis=2) if len(block_arrs_k) > 1 else block_arrs_k[0]
        values = mx.concatenate(block_arrs_v, axis=2) if len(block_arrs_v) > 1 else block_arrs_v[0]
        return keys, values


def _scatter_token(pool: mx.array, block_idx: int, offset: int, token_kv: mx.array) -> mx.array:
    """Functional in-place update: ``pool[block_idx, :, offset, :] = token_kv``.

    MLX arrays are functional; we use ``mx.put`` semantics via explicit
    indexed assignment by constructing the updated tensor. ``token_kv``
    must be shape ``[H, D]``.
    """
    if token_kv.ndim != 2:
        raise ValueError("token_kv must be 2D [H, D]")
    # `pool[block_idx, :, offset, :] = token_kv`
    # MLX supports indexed update via __setitem__ on arrays returned by
    # `mx.array(...)`; for a defensive impl we go through `mx.array` to
    # ensure we own a mutable copy.
    pool[block_idx, :, offset, :] = token_kv
    return pool


@dataclass
class PagedKVCache:
    """Adapter that presents :class:`PagedKVPool` storage with the same
    ``append`` / ``gather`` shape contract used by the dense
    :class:`baby_whale_v4.cache.DynamicKVCache`.

    Each layer's KV lives in the shared :class:`PagedKVPool`, addressed by
    this request's :class:`PageTable`. The pool is shared across requests
    so multiple PagedKVCache instances can coexist and (in a future
    extension) share blocks for common prefixes.
    """

    pool: PagedKVPool
    table: PageTable = field(default_factory=PageTable)

    @property
    def length(self) -> int:
        return self.table.length

    def sequence_length(self, layer_idx: int) -> int:
        """Tokens cached for ``layer_idx`` — the model reads this to place the
        next tokens' positions. Matches :meth:`DynamicKVCache.sequence_length`.
        """
        return self.table.layer_length(layer_idx)

    def max_sequence_length(self) -> int:
        """Longest per-layer history; the model uses it for the context bound."""
        return self.table.length

    def append(self, layer_idx: int, key: mx.array, value: mx.array) -> tuple[mx.array, mx.array]:
        """Write the new K/V into pool blocks and return the full gathered K/V.

        Mirrors :meth:`DynamicKVCache.append` semantics: callers expect the
        cumulative K/V across the request's history back.
        """
        self.pool.append_tokens(layer_idx, self.table, key, value)
        return self.pool.gather(layer_idx, self.table)

    def gather(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        return self.pool.gather(layer_idx, self.table)

    def latent_length(self, layer_idx: int) -> int:
        raise NotImplementedError(_MLA_UNSUPPORTED)

    def append_latent(self, layer_idx: int, latent: mx.array) -> mx.array:
        raise NotImplementedError(_MLA_UNSUPPORTED)

    def free(self) -> None:
        """Release this request's blocks back to the pool."""
        self.pool.free_table(self.table)
