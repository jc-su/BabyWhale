# 02 · Attention basics

**Prereqs:** [01 · Backbone](../01-backbone/README.md) · **Unlocks:** [03 · MLA](../03-attention-mla/README.md),
[04 · Compressed attention](../04-attention-compressed/README.md).

## 1 · The wall

A word's meaning depends on other words ("bank" of a river vs money). A per-token
MLP can't mix information *across* positions. You need a way for each token to look
at the others — but looking at *all* of them is O(n²), and a language model must not
peek at the future.

## 2 · The idea

**Scaled dot-product attention**: each token emits a query; it scores every key,
softmaxes, and reads a weighted sum of values. Three refinements this repo uses:

- **Causal mask** — a token attends only to itself and the past.
- **Sliding window** — attend only to the last `W` tokens: cheap, and most
  dependencies are local anyway.
- **MQA (multi-query)** — all query heads share *one* key/value head: far smaller KV
  cache (you'll cash this in at Module 14).
- **Partial RoPE** — rotate part of q/k by position so attention is
  position-aware without learned position embeddings.

## 🧩 From theory to code

| The math | The code (`attention.py`, `layers.py`) | Why this |
|----------|----------------------------------------|----------|
| $q, k, v = xW_q,\, xW_k,\, xW_v$ | the q/k/v projections | per-token query / key / value |
| rotate $q, k$ by position | `PartialRotaryEmbedding` | inject *relative* position into the dot-product |
| $s = qk^\top / \sqrt{d}$ | `(q @ k.T) / √head_dim` | similarity, scaled so softmax stays sharp |
| keep $j \le i$ and $j > i - W$ | causal + sliding mask | no future; only the last $W$ tokens |
| $a = \operatorname{softmax}(s)\,v$ | softmax then `@ v` | a weighted read of the values |

Why MQA (one shared K/V head)? queries still differ per head, but sharing K/V shrinks the
cache ~`n_head`× — the memory win you bank in Module 14.

## 3 · In the code

The whole operation (`attention.py`, `SlidingMQAAttention`):

```python
q = self._split_heads(self.q_proj(x), self.n_head)
k = self._split_heads(self.k_proj(x), self.n_kv_head)   # MQA: fewer KV heads than Q heads
q, k = self.rope(q, k, positions)                        # rotary positions
scores = (q @ k_e.swapaxes(-2, -1)) / math.sqrt(self.head_dim)
causal = key_pos <= q_pos                                # no future
local = key_pos >= (q_pos - self.sliding_window + 1)     # only the last W
weights = self.attn_dropout(_masked_softmax(scores, causal & local))
```


- `baby_whale_v4/attention.py` — `class SlidingMQAAttention` (`_attend`: build the
  causal + sliding mask, softmax, weight the values).
- `baby_whale_v4/layers.py` — `class PartialRotaryEmbedding` (the RoPE rotation;
  `rope_fraction` controls how much of the head is rotated).

## 4 · The payoff, measured

```bash
uv run python course/02-attention-basics/ablation.py
```

Sliding attention turns O(n²) into O(n·W). With `sliding_window=W`, each token does
at most `W` comparisons regardless of sequence length — check `sliding_window` in a
preset and reason about the cost.

## 5 · Break it & reflect

- **Reflect (🔬 systems):** attention memory is O(n²). Double the sequence length — how
  much more? Now cap it with a window `W`: what does the scaling become?

- Widen/shrink `sliding_window` and re-run the journey (Module 00) — quality vs cost.
- Set `rope_fraction=0` — does position information disappear?

## 🔨 Build — implement it yourself

Two labs; each docstring walks math → code → *why*:

```bash
uv run python course/02-attention-basics/lab_rope.py        # rotate-half positions
uv run python course/02-attention-basics/lab_attention.py   # softmax(q·kᵀ/√d)·v
```

(The RoPE lab uses the *split-half* rotate convention `[-x2, x1]`; `baby_whale_v4`'s
`rotate_half` uses the *interleaved* even/odd variant — both are valid RoPE.)

**Next:** [03 · MLA ⭐](../03-attention-mla/README.md) — the cache-memory breakthrough.
