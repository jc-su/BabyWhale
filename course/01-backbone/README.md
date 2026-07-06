# 01 · Backbone

The skeleton every other module hangs on.

**Prereqs:** [00 · The map](../00-the-map/README.md) · **Unlocks:** the whole stack.

## 1 · The wall

You have token ids. You want next-token probabilities. Between them you need a
*differentiable function* deep enough to learn language but stable enough to train.
Stack naive layers and gradients explode or vanish; the model won't learn.

## 2 · The idea

A transformer decoder: **embed** tokens into vectors, run them through **N identical
blocks**, **norm**, then project to vocabulary logits. Each block has two sublayers —
attention (mix across tokens) and an MLP/MoE (think per token) — each wrapped in a
**residual connection** (`x + f(norm(x))`). The residual stream is the highway
information flows along; **RMSNorm** keeps activations well-scaled so gradients stay
healthy. Training signal: **cross-entropy** between shifted logits and targets.

## 🧩 From theory to code

Tokens in, logits out — as residual updates on one shared stream:

| The math | The code (`model.py`) | Why this |
|----------|------------------------|----------|
| $h = \operatorname{Embed}(\text{ids})$ | `tok_emb(input_ids)` | token ids → vectors |
| $h \mathrel{+}= \operatorname{Attn}(\operatorname{RMSNorm}(h))$ | block sublayer 1 | pre-norm residual: mix across tokens, add back |
| $h \mathrel{+}= \operatorname{FFN}(\operatorname{RMSNorm}(h))$ | block sublayer 2 | per-token compute, add back |
| $\text{logits} = \operatorname{RMSNorm}(h)\, E^\top$ | `lm_head(norm(x))` | project to vocab (weight-tied to `Embed`) |

Why *pre*-norm (norm inside the residual, not around it)? it keeps the residual stream
un-normalized, so gradients flow cleanly through all `n_layer` blocks.

## 3 · In the code

The block forward — pre-norm, sublayer, residual add-back (`model.py`, `BabyWhaleV4Block`):

```python
x = self.hc.consume(h, layer_idx=self.layer_idx, sublayer_idx=0)
x = self.ln_1(x)                                    # pre-norm
delta_a = self.attn(x, cache=cache, positions=positions, key_mask=key_mask)
h = self.hc.produce(h, delta_a, layer_idx=self.layer_idx, sublayer_idx=0)
# ... then the same pattern for the MoE sublayer (ln_2 -> moe -> produce) ...
```


- `baby_whale_v4/model.py` — `class BabyWhaleV4Block` (the two-sublayer residual block)
  and `class BabyWhaleV4Model` (embed → blocks → norm → `lm_head`).
- `baby_whale_v4/layers.py` — `class RMSNorm`, `class WhaleLinear`.
- `baby_whale_v4/model.py` — `cross_entropy_ignore` (the loss, with padding ignore).

Read `BabyWhaleV4Model.__call__` top to bottom: it *is* the forward pass.

## 4 · The payoff, measured

```bash
uv run python course/00-the-map/journey.py
```

That builds this backbone and trains it — loss falls from ~6 to ~2. You can also ask
a model how big it is: `BabyWhaleV4Model(cfg).num_parameters()`.

## 5 · Break it & reflect

- **Reflect (🧠 theory):** mentally delete the residual and stack 12 layers — why do
  gradients vanish, and why does `x + f(x)` restore a clean gradient path?

- Set `n_layer=1` in a preset and re-run the journey — how much worse?
- Remove the residual (mentally): why would deep stacks stop training?

## 🔨 Build — implement it yourself

```bash
uv run python course/01-backbone/lab_rmsnorm.py            # RMSNorm
uv run python course/01-backbone/lab_transformer_layer.py  # assemble a real transformer layer
```

Each docstring walks math → code → *why*; the layer lab is graded against the **real**
`baby_whale_v4` modules. See the [Build track](../BUILD.md) for the full order.

**Next:** [02 · Attention basics](../02-attention-basics/README.md) — how a token looks at other tokens.
