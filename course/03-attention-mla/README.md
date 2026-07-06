# 03 · MLA — the KV-cache breakthrough ⭐

> This is the fully-built exemplar module. Its shape — a connection map, five beats, a
> systems lens, three tracks, and a milestone tie-in — is the template every other
> module follows.

**Prereqs:** [02 · Attention basics](../02-attention-basics/README.md) (KV, RoPE) ·
**Unlocks:** [14 · KV cache](../14-kv-cache/README.md) (why the cache is small) → milestone
[**E · It serves**](../MILESTONES.md).

## 1 · The wall

At **generation** time, a transformer keeps a **KV cache**: for every past token,
in every layer, it stores the Keys and Values so it doesn't recompute them. That
cache is the memory bottleneck of inference. With ordinary multi-head attention
(MHA), it's `n_head × head_dim × 2` numbers **per token, per layer** — and it grows
linearly with context length. Long contexts and big batches hit a wall of pure KV
memory long before they run out of compute.

Multi-Query Attention (MQA) shrinks it by sharing *one* KV head across all query
heads — much smaller cache, but attention quality suffers because every head now
reads the same K/V.

**Can we get MHA-quality attention at MQA-size cache?**

## 2 · The idea

**Multi-head Latent Attention (MLA)**, from the DeepSeek-V2/V3 papers, says: don't
cache K and V at all. Cache a single small **latent** `c_kv` per token — a low-rank
compression — and *reconstruct* the per-head K and V from it on the fly during
attention.

```
       cache full K,V          cache a latent, reconstruct per head
MHA:   [head_dim × n_head × 2]  MLA:  c = x·W_down   (dim r, small — THIS is cached)
                                       K,V = c·W_up_{k,v}   (rebuilt per head at use)
```

Because the reconstruction is per-head, you keep MHA-style head diversity; because
only `c_kv` (dim `kv_lora_rank`) is cached, the memory is MQA-ish. Best of both.

📄 DeepSeek-V2 (2024), §"Multi-Head Latent Attention".

## 🧩 From theory to code

The whole method is three moves: compress to a latent, cache *that*, reconstruct per head.
Here's the math mapped to the exact operations in `attention.py`'s `MLAAttention`:

| The math | The code | Why this |
|----------|----------|----------|
| `c = x · W_DKV` (latent, dim `r` ≪ d) | the down-projection to `c_kv` | K/V across heads are correlated, so a low-rank `c` keeps most of the signal in `r` numbers |
| cache `c` | `cache.append_latent(c_kv)` | store `r` per token, not `n_head · head_dim · 2` |
| `K, V = split(c · W_UKV)` | one up-projection `kv_b_proj` | rebuild all heads' K/V from the latent *at use* → MHA-style head diversity |
| `softmax(q·Kᵀ / √d) · V` | ordinary attention | unchanged — MLA only changed *what gets cached* |

So "8× smaller cache at MHA quality" isn't magic: it's a rank-`r` bottleneck on K/V that
you pay a matmul to reconstruct. Push `kv_lora_rank` down and you trade reconstruction
fidelity for memory — the exact knob beat 5 asks you to turn.

## 3 · In the code

- **`baby_whale_v4/attention.py:268` — `class MLAAttention`.** The docstring spells
  out the compression: the input becomes one low-rank latent `c_kv` of dimension
  `kv_lora_rank`.
- The **cache stores latents, not K/V**: `cache.append_latent(...)` and
  `cache.latent_length(...)` (contrast with the K/V `append`/`sequence_length`
  used by `SlidingMQAAttention`). That's the entire memory win, made concrete.
- The size knob is **`kv_lora_rank`** in `baby_whale_v4/config/__init__.py`
  (default 64) — the latent dimension.

Open `attention.py` at `MLAAttention` and read the forward alongside beat 2: down-
project to `c_kv`, append it to the cache, up-project to per-head K/V, attend.

## 4 · The payoff, measured

```bash
uv run python course/03-attention-mla/ablation.py
```

```
KV cache, bytes/token/layer (bf16):
  MHA (n_kv_head=8)     2048    1.0x smaller than MHA
  MQA (n_kv_head=1)      256    8.0x smaller than MHA
  MLA (latent=128)       256    8.0x smaller than MHA

MLA caches ~MQA-size but reconstructs per-head K/V -> MHA-quality attention.
```

The number is the point: **8× less KV memory than MHA, at the same size as MQA** —
but with per-head reconstruction, so quality tracks MHA, not MQA.

## 🔬 Systems lens

KV bytes are one axis. Put a whole model's numbers on the table:

```bash
uv run python -c "from course.systems import print_systems; from course.presets import load_preset; print_systems(load_preset('plus-mla'))"
```

`course/systems.py` is the recurring "by the numbers" beat — params, weight memory
(bf16 vs fp4), MoE sparsity, attention MACs — so every module ends on a real cost.

## 5 · Break it & reflect

- Halve `kv_lora_rank` (edit the config, re-run the ablation) → cache halves again.
  How low can it go before quality would suffer? (That's a real research question —
  measure perplexity with the eval in Module 19.)
- **Reflect (systems):** a 32-layer model, 8 KV heads × 128 head-dim, bf16, at 8k
  context — how many **GB** of KV cache for *one* sequence under MHA? Under MLA with
  `kv_lora_rank=512`? That ratio is how many more users fit on the same GPU.
- Compare against the `plus-mla` vs `gpt-minimal` presets (`course/presets.py`):
  train both (Module 09) and see whether the latent bottleneck costs any loss.

---

## 🔨 Build track — implement it yourself

The core of MLA is two matmuls: compress, then reconstruct. Fill in
`mla_roundtrip` and let the repo's own reference grade you:

```bash
uv run python course/03-attention-mla/lab_mla.py     # NotImplementedError until you fill it
# ... edit lab_mla.py ...
uv run python course/03-attention-mla/lab_mla.py     # PASS ✅
```

The grader (`course/labs.py:grade_mla_roundtrip`) checks your latent is the right
(small) shape *and* that both matmuls match the reference exactly. You cannot pass
by guessing.

## 🚀 Extend track

- Add a **decoupled RoPE** path (real MLA applies rotary position to only part of
  the query/key) and measure the quality delta.
- Wire your `mla_roundtrip` intuition to the real `MLAAttention` and confirm the
  cached-latent decode is bit-identical to a full-K/V recompute (a parity test in
  the spirit of `tests/test_mla.py`).

**Next:** [04 · Compressed attention](../04-attention-compressed/README.md) — HCA/CSA, for
*long-range* reach rather than cache size.
