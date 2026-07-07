# 19 · Evaluation — is judged

**Prereqs:** none · **Unlocks:** every [milestone](../MILESTONES.md).

## 1 · The wall

Training loss going down is necessary but not sufficient. A low loss can still mean a
model that can't follow an instruction, retrieve a fact, or write correct code. Without
task metrics you're flying blind — and you can't tell if a change *helped*.

## 2 · The idea

Measure what you actually care about, per capability:

- **Bits-per-byte** — a tokenizer-independent language-modeling score.
- **pass@1** — does generated code pass held-out tests? (run in a sandbox)
- **IFEval** — did it obey the instruction's constraints?
- **Needle retrieval** — can it recover a fact planted far back? (the long-context probe)

Evaluation isn't a final step — it's the **measure beat** of every module in this course.

## 🧩 From theory to code

| The math | The code (`cli/eval.py`, `eval/`) | Why this |
|----------|-----------------------------------|----------|
| $\text{bpb} = \dfrac{\sum \text{CE (nats)}}{\ln 2 \cdot n_\text{bytes}}$ | `eval-bpb` | a *tokenizer-independent* language-model score |
| $\text{pass@1} = \text{solved} / \text{total}$ | `eval-code` (sandboxed) | did the generated code actually run? |
| $\text{acc} = \text{retrieved} / n$ | `eval/needle.py` | long-range retrieval (Module 04) |

Why bits-per-byte, not perplexity? perplexity depends on the tokenizer; normalizing by
*bytes* makes scores comparable across different tokenizers.

## 3 · In the code

Bits-per-byte (`cli/eval.py`) — nats → bits, normalized by *bytes* so tokenizers can't game it:

```python
mean_loss_nats = total_loss_nats / total_tokens_in_loss
bpb = (mean_loss_nats / math.log(2)) * tokens_per_byte
```

And the needle probe (`eval/needle.py`) — per-sample *random* answers force retrieval:

```python
x = rng.integers(low, vocab_size, size=(n_samples, seq_len), dtype=np.int32)  # filler
answers = rng.integers(low, vocab_size, size=(n_samples,), dtype=np.int32)   # varies!
x[i, pos] = marker_id; x[i, pos + 1] = int(answers[i])   # plant marker+answer mid-sequence
x[i, seq_len - 1] = marker_id                            # query marker at the end
preds = mx.argmax(final_logits, axis=-1)                 # can it recall the answer?
```


- `baby_whale_v4/cli/eval.py` — `eval-bpb`, `eval-code`, `eval-ifeval`, `eval-dpo`,
  `eval-rl-health`.
- `baby_whale_v4/eval/needle.py` — `evaluate_needle_retrieval` (per-sample varying
  answers, so it tests *retrieval*, not memorization).

## 4 · The payoff, measured

```bash
uv run baby-whale-v4 eval-bpb --help
uv run baby-whale-v4 eval-code --help      # sandboxed pass@1
uv run python -m unittest tests.test_needle_eval
```

`test_needle_eval` shows the eval is *sensitive*: a trained model scores far above an
untrained one.

## 5 · Break it & reflect

- **Reflect (🧠 theory):** why can training loss keep dropping while a task metric (pass@1)
  stays flat? (Goodhart, and the gap between likelihood and capability.)

- Move the needle deeper than the sliding window and re-measure — which attention
  schedule (Module 04) still finds it?

**Next:** [20 · Vision (VL2)](../20-vision-vl2/README.md) — give the model eyes.
