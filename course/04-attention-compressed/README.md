# 04 · Compressed attention (HCA / CSA)

**Prereqs:** [02 · Attention basics](../02-attention-basics/) · **Unlocks:**
[10 · Mid-training](../10-midtraining/) (long context).

## 1 · The wall

Sliding-window attention (Module 02) is cheap but **short-sighted**: a token can't
see anything older than its window. Full attention sees everything but costs O(n²).
For long-context retrieval — "what was the name defined 5,000 tokens ago?" — you need
long reach *without* quadratic cost.

## 2 · The idea

Compress the far past so it stays affordable to attend to:

- **HCA (Hierarchical Compressed Attention)** — attend to recent tokens in full (the sliding
  window) **plus** a mean-pooled *summary* of each distant block.
- **CSA (Compressed Selective Attention)** — NSA-style: the same local window, but for the far
  past build overlapping compressed blocks and **select the top-k** most relevant (via a
  learned indexer) rather than using all of them.

A schedule mixes layer kinds (`sliding_mqa`, `mla`, `hca`, `csa`) so the model gets
both cheap local layers and a few long-reach layers.

## 🧩 From theory to code

| The math | The code (`attention.py`) | Why this |
|----------|---------------------------|----------|
| local: raw keys within the sliding window | the `raw_allowed` mask | recent tokens attended in full |
| distant (HCA): `k̄_b = mean(block b)` | `_block_mean_pool` + `comp_allowed` | far past → one summary key per block |
| distant (CSA): top-k of learned block scores | `_overlap_mean_pool` + `indexer` + `argsort` | pick the few most relevant *compressed* far blocks |

So both keep a full-detail **local** window and add **compressed** far-past keys — HCA every
block summary, CSA only the top-k it selects. Why a mean pool? it's a lossy but O(1) summary
of a block — enough to *route* attention to the right region of a long context, which a
sliding window can't reach at all.

## 3 · In the code

- `baby_whale_v4/attention.py` — `class HCAAttention` (block mean-pool) and
  `class CSAAttention` (overlap pool + top-k index).
- `baby_whale_v4/config/__init__.py` — `layer_schedule` chooses the kind per layer;
  `hca_block_size`, `csa_block_size`, `csa_block_stride`, `csa_index_topk` tune them.

## 4 · The payoff, measured

The needle-retrieval eval (Module 19) is exactly the probe: can the model recall a
fact placed far back? Sliding-only fails once the needle is past its window; the
compressed layers can still reach it.

```bash
# see Module 19; compares reach across schedules
```

## 5 · Break it & reflect

- **Reflect (🧠 theory):** block-mean-pooling forgets *which* token in a block mattered.
  When does that lossy summary break retrieval — and why does CSA's top-k selection help?

- Use the `plus-compressed` vs `gpt-minimal` presets (`course/presets.py`) and run the
  needle eval — does reach improve?

**Next:** [05 · Mixture of Experts](../05-moe/) — more capacity without more FLOPs.
