# 12 · DPO — learns preferences

**Prereqs:** [11 · SFT](../11-sft/) · **Unlocks:** [13 · RL](../13-rl-grpo/).

## 1 · The wall

SFT (Module 11) teaches the model to *imitate* good answers. But "good" is comparative —
given two plausible answers, which is *better*? Classic RLHF answers this by training a
separate reward model and running PPO: powerful, but heavy and finicky.

## 2 · The idea

**Direct Preference Optimization** skips the reward model. Given `(prompt, chosen,
rejected)` triples, push the policy to raise the log-prob of `chosen` and lower
`rejected` — *relative to a frozen reference model* (so it doesn't drift far from the
SFT model). One clean loss, no RL loop. Because the reference is **frozen**, its
log-ratios are constant across steps and are **precomputed once** — a free ~2× on the
reference forwards.

## 🧩 From theory to code

| The math | The code (`training/dpo.py`) | Why this |
|----------|------------------------------|----------|
| `Δ = (log π_c − log π_r) − (log ref_c − log ref_r)` | policy-vs-reference log-ratio | prefer chosen over rejected, relative to the frozen ref |
| `L = −log σ(β·Δ)` | `-log_sigmoid(β·Δ)` | grow the preference margin |
| ref log-ratios precomputed once | `_precompute_ref_logratios` | the ref is frozen → constant across steps |

Why anchor to a reference at all? without it the policy wanders off the SFT distribution
and loses fluency while chasing the preference signal.

## 3 · In the code

- `baby_whale_v4/training/dpo.py` — `dpo_loss` / `_dpo_pair_loss` (the β-scaled
  log-ratio of policy-vs-reference), and `_precompute_ref_logratios` (the caching).

## 4 · The payoff, measured

Track the **chosen-minus-rejected reward margin** rising over training, or run
`uv run baby-whale-v4` eval for DPO. The equivalence of the cached and recomputed paths
is proven in `tests/test_dpo_cache.py`.

## 5 · Break it & reflect

- **Reflect (🧠 theory):** DPO is the RLHF objective in closed form. What does the frozen
  reference prevent, and what happens to fluency as β → ∞?

- Raise `beta` (stronger pull from the reference) — does the model stay fluent or
  collapse toward the reference?

## 🔨 Build — implement the DPO loss yourself

```bash
uv run python course/12-dpo/lab_dpo.py
```

One line, graded against the objective in `baby_whale_v4/training/dpo.py`.

**Next:** [13 · RL with verifiable rewards](../13-rl-grpo/) — when the environment grades you.
