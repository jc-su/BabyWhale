# 10 · Mid-training — specializes

> The phase most tutorials skip entirely.

**Prereqs:** [09 · Pre-training](../09-pretraining/README.md) · **Unlocks:** [11 · SFT](../11-sft/README.md).

## 1 · The wall

Pre-training (Module 09) gives broad but shallow competence on a fixed context length.
You now want: **longer context** (it was trained short for speed), **higher-quality**
behavior (the web corpus was noisy), and **specific capabilities** (code, math). Jumping
straight to alignment wastes the chance to fix these cheaply first.

## 2 · The idea

**Mid-training** sits between pre-training and post-training:

- **Context extension** — continue training at longer sequence lengths (a curriculum).
- **Data annealing** — shift the mix toward curated, high-quality data as you go.
- **Capability injection** — over-sample code/math so the base model has the skills that
  later RL will sharpen.

It's still next-token prediction — just on a deliberately chosen diet.

## 🧩 From theory to code

Not an equation — a *recipe*. Each step is a data/config change on top of the same
next-token loop:

| The recipe step | The knob / code | Why this |
|-----------------|-----------------|----------|
| extend the context window | raise `context_length`, keep training | RoPE + long documents teach the model to *use* the new range |
| anneal the data mix | shift toward curated/high-quality data, late | spend the final, most-effective steps on the best signal |
| inject capabilities | over-sample code / math | give the base the skills RL (Module 13) will then sharpen |

Why a separate phase, not just more pre-training? these fixes are cheap to do *after* broad
pre-training and *before* alignment — and fixing context + quality first makes every
downstream stage better. Measured with the needle eval (Module 19).

## 3 · In the code

The honest punchline — mid-training **is** pre-training on a different diet, and the code
says so literally (`training/midtrain.py`):

```python
def midtrain(*, config, midtrain_config, train_dataset, out_dir, ...):
    return pretrain(                     # the SAME loop —
        config=config,
        pretrain_config=PretrainConfig(  # — different lr/steps/warmup...
            lr=midtrain_config.lr,
            max_steps=midtrain_config.max_steps,
            warmup_steps=midtrain_config.warmup_steps,
            ...
        ),
        train_dataset=train_dataset,     # ...and, crucially, DIFFERENT DATA
        ...
    )
```

Everything that makes it "mid-training" is the *inputs*: longer-context data, curated
mixes, code/math oversampling — not a new algorithm.


- `uv run baby-whale-v4 midtrain --help` — the mid-training entry point.
- The context curriculum lives in the training/config plumbing; the payoff is measured
  with the long-context probe in `baby_whale_v4/eval/needle.py`.

## 4 · The payoff, measured

Use the **needle-retrieval** eval (Module 19): a model that has had its context extended
and been annealed should retrieve a fact placed far back far better than the raw base.

## 5 · Break it & reflect

- **Reflect (🧠 theory):** a model trained at 512 context sometimes works at 2k and
  sometimes fails. What about RoPE and the training data decides which?

- Extend context but *don't* include long documents in the data — does reach improve?
  (Capability follows data, not just the config number.)

**Next:** [11 · SFT](../11-sft/README.md) — teach it to follow instructions.
