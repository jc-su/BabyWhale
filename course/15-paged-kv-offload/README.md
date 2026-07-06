# 15 · Paged KV & offload

**Prereqs:** [14 · KV cache](../14-kv-cache/) · **Unlocks:**
[17 · Continuous batching](../17-continuous-batching/).

## 1 · The wall

A single contiguous KV cache per request is wasteful: you must reserve the *maximum*
length up front, and different-length requests fragment memory — so you can't fit as
many concurrent requests as you should, and very long contexts overflow.

## 2 · The idea

Borrow the OS trick: **paging**. Store KV in fixed-size **blocks** and keep a per-request
**block table** mapping logical positions to physical blocks (this is the vLLM idea).
Allocation becomes block-granular — no per-request over-reservation, little fragmentation.
Cold blocks can be **offloaded** to CPU memory and reloaded on demand for very long or
paused contexts.

## 🧩 From theory to code

Not an equation — a *data structure*. It's the OS's virtual-memory trick applied to the
KV cache:

| The idea (OS paging) | The code (`inference/paged_kv.py`) | Why this |
|----------------------|-------------------------------------|----------|
| store KV in fixed-size blocks | `PagedKVCache` block pool | allocate in block units, not one contiguous span |
| map logical position → physical block | a per-request block table | different-length requests share the pool without fragmenting it |
| allocate / free blocks on demand | pool alloc + release | fit more concurrent requests in the same memory |
| move cold blocks to CPU | `kv_offload` | survive very long or paused contexts |

Why blocks (vLLM's idea)? a contiguous cache must reserve the *max* length up front; blocks
let it grow lazily and pack many requests tightly.

## 3 · In the code

- `baby_whale_v4/inference/paged_kv.py` — `class PagedKVCache` (block map shared across
  layers + per-layer lengths; `from_model_config`).
- KV **offload** moves blocks to/from CPU (`tests/test_paged_engine_offload.py`).

## 4 · The payoff, measured

Paging lets more requests share the same memory budget. Inspect block **occupancy** vs a
contiguous cache for a mix of request lengths — fragmentation is the metric.

## 5 · Break it & reflect

- **Reflect (🔬 systems):** block size 16 vs 256 — estimate the average wasted
  (internal-fragmentation) memory per request and the block-table overhead. Sweet spot?

- Shrink the block size — less waste per request, more block-table overhead. The tradeoff.

**Next:** [16 · Speculative decoding](../16-speculative-decoding/) — decode several tokens per step.
