# 06 · HyperConnect (mHC)

**Prereqs:** [01 · Backbone](../01-backbone/README.md) · **Unlocks:** richer depth-mixing.

## 1 · The wall

Every transformer bolts sublayers onto a single residual stream with a fixed rule:
`x + f(x)`. That's one hard-wired highway. Is a plain add really the best way to route
information between depth, or could the model *learn* how to combine branches?

## 2 · The idea

**Multi-branch Hyper-Connections (mHC)** widen the residual stream into `hc_mult`
parallel copies and let the model **learn** how each sublayer *reads from* (consume)
and *writes back to* (produce) those branches — a learned generalization of the
residual connection. `hc_mult=1` recovers ordinary residuals; larger gives the model
more freedom to mix.

## 🧩 From theory to code

Not an equation — a *mechanism*. mHC turns the single residual highway into several the
model learns to route between:

| The mechanism | The code (`mhc.py`) | Why this |
|---------------|---------------------|----------|
| widen the residual to `hc_mult` parallel streams | `HyperConnect.expand` | give depth more than one highway |
| a sublayer reads a *learned* mix of the streams | `consume(...)` | let each layer choose what to read |
| mix the streams, then add the sublayer output with learned write weights | `produce(...)` | recombine streams (sinkhorn) and write the update back |
| collapse the streams back to one | `reduce(...)` | rejoin for the LM head |

Why learnable (vs a fixed `x + f(x)`)? `hc_mult=1` recovers the plain residual exactly;
`>1` lets the model mix information across depth in ways a single add cannot express.

## 3 · In the code

- `baby_whale_v4/mhc.py` — `class HyperConnect` (`expand`, `consume`, `produce`,
  `reduce`).
- `baby_whale_v4/model.py` — the block calls `hc.consume(...)` before each sublayer and
  `hc.produce(...)` after; config knob `hc_mult`.

## 4 · The payoff, measured

Compare the `full` preset (`hc_mult=2`) against `plus-mtp` (`hc_mult=1`) on the same
training run (Module 09) — does the learned multi-branch residual buy any loss?

## 5 · Break it & reflect

- **Reflect (🧠 theory):** what information routing can a learned `hc_mult=2` residual
  represent that a plain `x + f(x)` cannot?

- Set `hc_mult=1`: you're back to a vanilla residual. Confirm the model still trains.
- Trace one token's value through `expand → consume → produce → reduce`.

**Next:** [07 · Multi-token prediction](../07-mtp/README.md) — predict more than one token at once.
