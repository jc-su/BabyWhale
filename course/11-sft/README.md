# 11 · SFT — learns to behave

**Prereqs:** [09 · Pre-training](../09-pretraining/README.md) · **Unlocks:** [12 · DPO](../12-dpo/README.md).

## 1 · The wall

A pre-trained base model *continues text*. Ask it a question and it might reply with
more questions — it has never been shown that a prompt should be *answered*, or what a
chat turn looks like.

## 2 · The idea

**Supervised fine-tuning**: train on curated `(instruction, response)` pairs formatted
with a **chat template** (special tokens marking system/user/assistant turns). Crucially,
compute the loss **only on the response tokens** — you want the model to learn to
*produce* answers, not to re-predict the user's prompt.

## 🧩 From theory to code

| The math | The code (`training/sft.py`) | Why this |
|----------|------------------------------|----------|
| render turns with role tokens | `data/chat.py` `render_chat_prompt` | model + server agree on one chat format |
| `L = CE(logits, targets)`, prompt positions masked | response-only loss mask | learn to *produce* answers, not re-predict prompts |

Why mask the prompt? capacity should go to responses; re-predicting the user's prompt is
wasted signal that dilutes instruction-following.

## 3 · In the code

The masking is done in the *data*, not the loss (`data/chat.py`) — prompt positions get
target `-1`, which `cross_entropy_ignore` skips:

```python
ids, mask = format_chat(ex.messages, self.tokenizer)   # mask=1 on response tokens only
x = mx.array(ids[:-1], dtype=mx.int32)
y = mx.array(ids[1:], dtype=mx.int32)
mask_y = mx.array(mask[1:], dtype=mx.int32)
y = mx.where(mask_y == 0, mx.full(y.shape, -1, dtype=mx.int32), y)   # prompt -> ignore
```

The SFT loop itself (`training/sft.py`) is then just supervised training on `(x, y)`.


- `baby_whale_v4/training/sft.py` — the SFT loop and response-only loss masking.
- `baby_whale_v4/data/chat.py` — `render_chat_prompt` (turns messages into the templated
  token stream the model and server agree on).

## 4 · The payoff, measured

Prompt the base model vs the SFT'd model with the same instruction (via `generate` /
the chat server, Module 17) — the SFT'd one answers in-format instead of rambling.

```bash
uv run baby-whale-v4 sft --help
```

## 5 · Break it & reflect

- **Reflect (🎓 alignment):** if you *don't* mask the prompt from the loss, what does the
  model waste capacity learning — and why does that hurt instruction following?

- Turn *off* prompt masking (loss on all tokens) — the model wastes capacity learning
  to parrot prompts. Why does that hurt?

**Next:** [12 · DPO](../12-dpo/README.md) — from imitation to preference.
