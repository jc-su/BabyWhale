# 20 · Vision (VL2) — the frontier

**Prereqs:** [05 · MoE](../05-moe/README.md), [17 · Continuous batching](../17-continuous-batching/README.md)
· **Unlocks:** multimodal models.

## 1 · The wall

Everything so far is text-only. To answer questions about an image, the model needs to
*see* it — but bolting on vision must not disturb the text model you spent the whole
course building (existing checkpoints must still load, text behavior must be identical).

## 2 · The idea

The DeepSeek-VL2 recipe: **tile** a high-res image into a grid (plus a thumbnail),
**encode** each tile (a SigLIP-style encoder), **project** the tile features into the LLM's
embedding space with an **MLP connector**, and **prepend** those image tokens to the text
stream. Everything is gated by `enable_vision`, off by default — and the vision config
fields are **excluded from the config hash when off**, so pre-vision checkpoints load
unchanged.

## 🧩 From theory to code

Not an equation — a *pipeline*. Pixels become tokens the existing transformer already knows
how to handle:

| The pipeline stage | The code (`vision/`, `model.py`) | Why this |
|--------------------|----------------------------------|----------|
| split a high-res image into tiles + a thumbnail | `vision/tiling.py` `plan_tiles` | see detail without one huge fixed resolution |
| encode each tile to feature vectors | a SigLIP-style encoder (a weights port) | turn pixels into vectors the LLM can read |
| project features into the token space | `vision/connector.py` `VisionMLPConnector` | `vision_dim → n_embd`, a shared space for image + text |
| prepend image tokens to the text stream | `model.py` `_prepend_vision` | the LLM attends to them as a prefix |

Why prepend rather than cross-attend? it reuses the *exact* same transformer + KV cache —
images become "just more tokens," so nothing else in the stack has to change (and text-only
stays bit-identical when `enable_vision=False`).

## 3 · In the code

The integration point (`model.py`, `_prepend_vision`) — image features become a token prefix:

```python
vis = connector(image_features)                     # [B, n_vis, n_embd] — now token-shaped
combined = mx.concatenate([vis, x], axis=1)         # image tokens FIRST, then text
pad_ids = mx.zeros((input_ids.shape[0], vis.shape[1]), dtype=input_ids.dtype)
block_ids = mx.concatenate([pad_ids, input_ids], axis=1)   # placeholder ids for MoE routing
```

And the connector itself (`vision/connector.py`) is a two-layer `WhaleLinear` MLP:

```python
hidden = nn.gelu(self.fc1(features))    # vision_dim -> n_embd
return self.fc2(hidden)                 # n_embd -> n_embd
```


- `baby_whale_v4/vision/tiling.py` — `plan_tiles` (grid that minimizes padding + thumbnail).
- `baby_whale_v4/vision/connector.py` — `VisionMLPConnector` (`vision_dim → n_embd`).
- `baby_whale_v4/model.py` — `_prepend_vision` puts image tokens first (placeholder ids
  keep MoE routing happy); text-only path is bit-identical.

## 4 · The payoff, measured

`tests/test_vision_config.py` proves a **real pre-vision checkpoint still loads** (hash
stability), and `tests/test_vision_integration.py` shows image features prepend to the
sequence while `image_features=None` is a perfect no-op.

## 5 · Break it & reflect

- **Reflect (🧠 + 🔬):** why prepend image tokens rather than cross-attend to them? What
  does each choice cost in sequence length and attention compute?

- Flip `enable_vision` and diff the `config_hash` — why must it *not* change when off?
- Change the image's aspect ratio and watch `plan_tiles` pick a different grid.

**Next:** [21 · Capstone](../21-capstone/README.md) — take *your* model through the whole pipeline.
