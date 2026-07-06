# 13 · RL with verifiable rewards — learns to reason

**Prereqs:** [12 · DPO](../12-dpo/README.md) · **Unlocks:** reasoning / agent RL.

## 1 · The wall

Preference data (Module 12) needs humans to label what's better. But for **code and
math**, the world already grades you: does the program pass its tests? does the answer
equal the key? That's a reward signal with no human in the loop — if you can turn it
into a training update.

## 2 · The idea

**GRPO (Group Relative Policy Optimization)**: for a prompt, sample a **group** of
rollouts, score each with a **verifier** (run the code in a sandbox, check the tests),
then set each rollout's advantage by **normalizing rewards within the group** — no value
network needed. **RLOO** is the leave-one-out baseline variant. The policy is nudged
toward the above-average rollouts. This is the loop behind modern "reasoning" models,
in miniature.

## 🧩 From theory to code

| The math | The code (`training/grpo.py`, `rloo.py`, `rl/`) | Why this |
|----------|-------------------------------------------------|----------|
| $r_i = \operatorname{verify}(\text{rollout}_i)$ | sandbox reward (`rl/`) | the environment grades — do the tests pass? |
| GRPO $A_i = \dfrac{r_i - \bar r}{\operatorname{std}(r)}$ · RLOO $A_i = r_i - \operatorname{mean}(r_{j\ne i})$ | `grpo.py` / `rloo.py:_leave_one_out_advantage` | score each rollout against its group — no value net |
| maximize $\sum_i A_i \log\pi(\text{rollout}_i)$ | policy-gradient step | push probability toward above-average rollouts |

Why the group baseline? the group mean *is* the value estimate — GRPO also divides by the
group std, RLOO leaves each sample out — so neither needs a separate learned value network.

## 3 · In the code

- `baby_whale_v4/training/rloo.py` and `baby_whale_v4/rl/` — the GRPO/RLOO update.
- The **code sandbox** reward + end-to-end loop are exercised by
  `tests/test_code_agent.py` (`TestCodeGRPOEndToEnd`, `TestCodeRewardEndToEnd`).

## 4 · The payoff, measured

The verifiable metric *is* the reward: **pass@1** on held-out problems, rising as RL
proceeds.

```bash
uv run baby-whale-v4 grpo --help        # the RL loop
# eval-code gives the pass@1 number (Module 19)
```

## 5 · Break it & reflect

- **Reflect (🧠 theory):** why does normalizing rewards *within a group* remove the need
  for a separate value network? What does the group baseline estimate?

- Shrink the group size to 1 — the within-group baseline vanishes and variance explodes.
  Why does a *group* matter?

## 🔨 Build — implement the group advantage yourself

```bash
uv run python course/13-rl-grpo/lab_grpo.py
```

Normalize rewards within the group; the grader checks zero-mean and the formula.

**Next:** [14 · KV cache](../14-kv-cache/README.md) — now make the trained model fast to run.
