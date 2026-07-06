# 14 · KV cache — goes to work

**Prereqs:** [02 · Attention basics](../02-attention-basics/) · **Unlocks:**
[15 · Paged KV](../15-paged-kv-offload/), [16 · Speculative](../16-speculative-decoding/),
[17 · Batching](../17-continuous-batching/).

## 1 · The wall

To generate token 100, a naive model re-runs attention over tokens 1–99. To generate
token 101, it redoes 1–100. Decoding an `n`-token reply is O(n²) work — most of it
recomputing Keys and Values it already computed.

## 2 · The idea

Compute each token's **K and V once, cache them**, and have every new token attend to
the cache. Decode drops to O(n). The cache is defined as a **Protocol** so storage can
vary (dynamic, paged) without changing attention. **Chunked prefill** feeds a long
prompt through in blocks; a **prefix cache** reuses the KV of a shared prompt prefix
across requests.

## 🧩 From theory to code

| The math | The code (`cache.py`) | Why this |
|----------|------------------------|----------|
| `K_cache ← [K_cache ; k_new]` | `DynamicKVCache.append` (concatenate on time) | keep the past, add the new key |
| attend `q_t` to all of `K_cache` | ordinary attention over cached K/V | O(t) per step, not an O(t²) recompute |
| MLA layers cache a latent instead | `append_latent` (Module 03) | store `r` numbers/token, not full K/V |

Why does this make decode O(n)? each new token does one attention over the cache and one
append — no earlier token's K/V is ever recomputed.

## 3 · In the code

- `baby_whale_v4/cache.py` — `KVCache` (the Protocol) and `DynamicKVCache` (append K/V,
  `sequence_length`); MLA layers instead store latents (`append_latent`, Module 03).
- `baby_whale_v4/inference/prefix_cache.py` — reuse a shared prompt prefix.

## 4 · The payoff, measured

```bash
uv run python course/14-kv-cache/ablation.py
```

```bash
uv run baby-whale-v4 bench-compare --help     # cached vs re-compute tokens/sec
```

Cached decode is the difference between "a few tokens/sec" and "usable".

## 5 · Break it & reflect

- **Reflect (🔬 systems):** caching makes decode O(n) compute — but the cache grows O(n)
  in memory. For a 100k-token context, which is now the wall? (→ Module 15)

- Generate with a tiny vs huge prompt — where does prefill vs decode dominate the time?

**Next:** [15 · Paged KV & offload](../15-paged-kv-offload/) — managing that cache's memory.
