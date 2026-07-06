# 09 · Pre-training — learns to read

**Prereqs:** [01 · Backbone](../01-backbone/README.md),
[08 · Tokenizer & data](../08-tokenizer-and-data/README.md) · **Unlocks:**
[10 · Mid-training](../10-midtraining/README.md).

## 1 · The wall

A freshly-initialized model outputs noise. Everything it will ever know about language
has to come from somewhere. That somewhere is **pre-training**: predict the next token,
billions of times, over a large corpus.

## 2 · The idea

Minimize next-token **cross-entropy** with **AdamW**, but at scale that means getting
the boring things right: **gradient accumulation** (token-weighted, so short and long
sequences count fairly), a **learning-rate schedule** (warmup + decay), **throughput**
(tokens/sec), and — because runs are long — **checkpointing with exact resume** and
**corruption detection** so a crash doesn't cost you the run.

## 🧩 From theory to code

| The math | The code | Why this |
|----------|----------|----------|
| `L = −mean log p(next token given context)` | `cross_entropy_ignore` | next-token likelihood, ignoring pad/prompt |
| accumulate grads over microbatches, weighted by tokens | grad accumulation | fair averaging across ragged lengths |
| `θ ← θ − lr·m̂/(√v̂+ε) − lr·λ·θ` | `AdamW.step` | adaptive step + *decoupled* weight decay |

Why weight accumulation by tokens, not batches? a microbatch with more real tokens should
count more in the mean loss — otherwise short sequences get over-weighted.

## 3 · In the code

- `baby_whale_v4/training/pretrain.py` — the loop (accumulation, schedule, metrics).
- `baby_whale_v4/training/mlx_optim.py` — `class AdamW` (`step(params, grads)`).
- `baby_whale_v4/training/checkpoint.py` — `.bw4` save/load with a config hash and
  tamper checks (`tests/test_checkpoint.py`).

## 4 · The payoff, measured

The real bounded run in this repo: **5.56M params, 15.2M tokens**, train loss
**8.12 → 4.46**, held-out eval **4.89**, ~10.5K tok/s. Reproduce the shape locally:

```bash
uv run python course/00-the-map/journey.py 200     # a mini pre-train
uv run baby-whale-v4 pretrain --help               # the real thing
```

## 5 · Break it & reflect

- **Reflect (🎓 + 🔬):** why weight gradient accumulation by *tokens*, not sequences?
  And roughly how many tokens must pass to update each parameter enough to learn?

- Kill a run mid-way and resume from the checkpoint — confirm optimizer state restores.
- Raise the learning rate until it diverges. Where's the edge?

**Next:** [10 · Mid-training](../10-midtraining/README.md) — from broad to specialized.
