# 16 · Speculative decoding

**Prereqs:** [07 · MTP](../07-mtp/README.md), [14 · KV cache](../14-kv-cache/README.md) · **Unlocks:**
faster serving.

## 1 · The wall

Decoding is **memory-bound**: each step is one forward pass that emits *one* token, and
the GPU spends most of its time moving weights, not computing. You're paying for a full
forward to get a single token.

## 2 · The idea

**Draft, then verify.** Cheaply guess the next few tokens (here, using the **MTP heads**
from Module 07), then run **one** forward pass that verifies all the guesses in parallel,
and accept the longest correct prefix. When guesses are good you emit several tokens per
forward; when they're wrong you fall back to one. Crucially it's **bit-identical to
greedy** — speed with no quality change.

## 🧩 From theory to code

| The math | The code (`model.py` `spec_decode`) | Why this |
|----------|--------------------------------------|----------|
| draft `d₁..dₖ` from MTP heads | the draft step | cheap guesses of the next k tokens |
| verify: one forward → greedy `g₁..gₖ` | the parallel verify pass | one forward checks all k at once |
| accept the longest prefix with `dᵢ = gᵢ` | the accept rule | emit multiple tokens, bit-identical to greedy |

Why is one forward enough to verify k tokens? attention over the drafted block is a single
batched forward — so verification costs one pass regardless of k. That's the speedup.

## 3 · In the code

- `baby_whale_v4/model.py` — `spec_decode(...)` returns a `SpecDecodeResult` whose
  `acceptance_rate = drafts_accepted / drafts_proposed`.
- Requires `mtp_heads > 0` (the `plus-mtp` / `full` presets).

## 4 · The payoff, measured

```bash
uv run python course/16-speculative-decoding/ablation.py
```

Two numbers: **tokens/sec** (up) and **acceptance rate** (how often drafts are right).

```bash
uv run baby-whale-v4 bench-compare --help     # greedy vs speculative
```

## 5 · Break it & reflect

- **Reflect (🧠 + 🔬):** if each drafted token is accepted with probability p and you
  draft k, what's the expected number emitted per verify pass? Why is it still
  bit-identical to greedy?

- Set `mtp_heads=0` — speculation is unavailable. With more heads, acceptance usually
  falls per-head but tokens/step can rise. Find the sweet spot.

## 🔨 Build — implement speculative acceptance yourself

```bash
uv run python course/16-speculative-decoding/lab_spec_accept.py
```

The lab shows *why* stopping at the first mismatch makes speculation lossless.

**Next:** [17 · Continuous batching](../17-continuous-batching/README.md) — serve many requests at once.
