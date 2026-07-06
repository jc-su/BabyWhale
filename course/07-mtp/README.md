# 07 · Multi-token prediction

**Prereqs:** [01 · Backbone](../01-backbone/README.md) · **Unlocks:**
[16 · Speculative decoding](../16-speculative-decoding/README.md).

## 1 · The wall

A standard LM predicts exactly one token — the next one — from each position. But the
hidden state at position `t` knows a lot about `t+2`, `t+3` too. Throwing that away
wastes both training signal and, later, decoding speed.

## 2 · The idea

Add small **MTP heads** that predict `t+2`, `t+3`, … from the *same* hidden state the
main head uses. Two payoffs:

- **Training** — extra prediction targets are a denser learning signal (the MTP loss is
  a weighted add-on to the main loss).
- **Inference** — those heads become a cheap **draft model** for speculative decoding
  (Module 16): guess several tokens, verify in one pass.

## 🧩 From theory to code

| The math | The code (`mtp.py`, `model.py`) | Why this |
|----------|---------------------------------|----------|
| `logitsₖ = MTPₖ(h)` | `mtp["head_k"](x)` | from the *same* hidden state, predict a token beyond the next (head 0 → t+2, head 1 → t+3) |
| `L += wₘ · Σₖ CE(logitsₖ, targets shifted by k)` | the MTP loss term | a denser training signal per position |

Why predict from the hidden state (not the token id)? the hidden carries the rich context
the main LM head uses, so the extra heads are good enough to *draft* for speculation
(Module 16) — the Medusa/EAGLE recipe.

## 3 · In the code

- `baby_whale_v4/mtp.py` — `class MTPHead` (projects the last hidden state to a token
  distribution at `t+k`).
- `baby_whale_v4/model.py` — `self.mtp` heads; the forward returns `mtp_logits`, the
  loss adds `mtp_loss_weight × Σ mtp_losses`. Config: `mtp_heads`.

## 4 · The payoff, measured

`mtp_heads=2` (the `plus-mtp` / `full` presets) unlocks `model.spec_decode(...)` —
whose `SpecDecodeResult.acceptance_rate` you'll measure in Module 16.

## 5 · Break it & reflect

- **Reflect (🧠 theory):** why is predicting t+2 a useful *training* signal (a denser
  gradient), not just an inference-time trick?

- Set `mtp_heads=0`: speculative decoding is no longer available. Why?
- Raise `mtp_loss_weight` — does forcing multi-token prediction help or hurt main loss?

## 🔨 Build — implement the MTP head yourself

```bash
uv run python course/07-mtp/lab_mtp.py
```

Graded against the real `MTPHead`. See the [Build track](../BUILD.md).

**Next:** [08 · Tokenizer & data](../08-tokenizer-and-data/README.md) — what the model actually eats.
