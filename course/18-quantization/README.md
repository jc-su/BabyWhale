# 18 · Quantization — gets compressed

**Prereqs:** [01 · Backbone](../01-backbone/README.md) · **Unlocks:** on-device deployment.

## 1 · The wall

Weights in fp32/bf16 are big and slow to move. On-device inference is bottlenecked by
memory bandwidth — but naive low-bit quantization wrecks quality, and not every layer
tolerates it equally.

## 2 · The idea

Store and multiply weights in **4-bit (FP4/NVFP4)** where it's safe, and be **explicit**
about where it isn't. A **placement policy** decides per layer: quantize the MoE experts
(most of the params) but keep sensitive layers (like `lm_head`) higher precision. Every
FP4 path is a **fail-fast gate** — no silent fallback that quietly costs quality.

## 🧩 From theory to code

| The math | The code (`quantization/`, `mlx_fp4`) | Why this |
|----------|----------------------------------------|----------|
| $s = \max\lvert w\rvert / q_{\max}$ | calibration | fit the weights into the 4-bit range |
| $w_q = \operatorname{round}(w / s)$ (4-bit) | quantize | store 4-bit codes + one scale per group |
| $\hat w = w_q\, s$ at matmul | quantized matmul | move ~4× fewer bytes for the same product |

Why a per-placement policy? sensitive layers (the `lm_head`, the router) keep high precision
via `WhaleLinear`'s `placement`, while the bulk (the experts) goes to 4-bit.

## 3 · In the code

- `baby_whale_v4/quantization/` and `baby_whale_v4/mlx_fp4/` — the FP4 matmul + policy.
- `baby_whale_v4/layers.py` — `WhaleLinear` carries a `placement` so the policy applies;
  config `quant_mode` selects the scheme.

## 4 · The payoff, measured

```bash
uv run python course/18-quantization/ablation.py
```

Memory and speed vs quality: compare a `quant_mode="none"` model against a quantized one
on size and tokens/sec (`bench-compare`), and check the quality cost with an eval
(Module 19).

## 5 · Break it & reflect

- **Reflect (🧠 + 🔬):** FP4 is 4× smaller — why not quantize `lm_head` and the router
  too? What makes a layer quantization-sensitive?

- Quantize `lm_head` too — watch quality drop. Why is the output projection sensitive?

## 🔨 Build — implement quantize / dequantize yourself

```bash
uv run python course/18-quantization/lab_quantize.py
```

The symmetric absmax round-trip (the repo ships group-affine / NVFP4 — same idea).

**Next:** [19 · Evaluation](../19-evaluation/README.md) — how do you *know* any of this worked?
