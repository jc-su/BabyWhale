# 05 · Mixture of Experts

**Prereqs:** [01 · Backbone](../01-backbone/README.md) · **Unlocks:** [18 · Quantization](../18-quantization/README.md).

## 1 · The wall

A dense feed-forward layer makes *every* token pay for *every* parameter. Want a
smarter model? Add parameters — but now every token costs more FLOPs. Capacity and
compute are chained together.

## 2 · The idea

**Sparse MoE** breaks the chain: have `N` expert MLPs, but **route each token to only
top-k** of them. You get N× the parameters at ~k× the FLOPs. Two hard parts this repo
shows honestly:

- **Routing stability** — early in training a learned router is random. Bootstrap with
  **hash routing** (deterministic token→expert) for the first layers, then hand off to
  a learned top-k router.
- **Load balancing** — naive routing collapses onto a few experts. The classic fix is
  an auxiliary loss, but that *fights* the language-model loss. DeepSeek-V3's
  **aux-loss-free** trick nudges routing with a per-expert **bias** updated from load —
  no extra gradient term.

## 🧩 From theory to code

| The math | The code (`moe.py`) | Why this |
|----------|---------------------|----------|
| `s = √softplus(x·Wᵣ)` | the router (`_learned_routes`) | a per-expert score for this token |
| top-k of `s + bias` | top-k select | sparse: run only the k best; the bias shifts *selection* only |
| `gₖ = sₖ / Σ sₖ`;  `y = Σₖ gₖ · Expertₖ(x)` | gated combine (raw `s`, unbiased) | a score-weighted sum of the chosen experts |
| `biasₑ ∓= rate` when `loadₑ ≷ mean` | `_maybe_update_bias` (no grad) | a fixed step toward balance — over-used down, under-used up |

(A always-on **shared expert** also runs on every token, alongside the routed ones.) Why a
bias, not an auxiliary loss? it steers *selection* without a gradient term that competes with
the language-model loss — DeepSeek-V3's aux-loss-free balancing.

## 3 · In the code

- `baby_whale_v4/moe.py` — `class SparseMoE`; the aux-loss-free per-expert bias
  (`router_bias`, `aux_free_bias_rate`) is a non-array leaf so it stays out of the
  gradient tree.
- Config: `n_expert`, `experts_per_token`, `n_shared_expert`, `n_hash_layers`,
  `aux_free_bias_rate`.

## 4 · The payoff, measured

```bash
uv run python course/05-moe/ablation.py
```

Turn balancing on/off with the `plus-moe-balanced` vs `plus-compressed` presets and
compare **expert utilization** — a balanced model uses all experts (high entropy), an
unbalanced one collapses onto a few.

## 5 · Break it & reflect

- **Reflect (🧠 + 🔬):** at k=2 of N=8 experts, how many more parameters than a dense
  FFN, at what extra FLOP cost? And *why* does an auxiliary load-balance loss pull
  against the language-model loss?

- Set `aux_free_bias_rate=0` and watch experts collapse.
- Set `experts_per_token = n_expert` — you've rebuilt a (slow) dense layer. Why?

## 🔨 Build — implement it yourself

```bash
uv run python course/05-moe/lab_moe_route.py   # top-k routing
uv run python course/05-moe/lab_swiglu.py      # the SwiGLU expert (graded vs the real one)
```

The routing lab derives softmax → top-k → renormalize — the *standard* formulation;
`baby_whale_v4` gates with √softplus (same top-k idea, different squashing).

**Next:** [06 · HyperConnect](../06-hyperconnect/README.md) — is `x + f(x)` really the best residual?
