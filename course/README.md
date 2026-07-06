# Baby Whale — the course

> **One tiny model's journey from random noise to a served, RL-tuned, evaluated system.**

Most LLM tutorials teach *one slice*: nanoGPT is pretraining, Spinning Up is RL, the
Annotated Transformer is architecture. This course walks a single model through the
**entire modern lifecycle** — the thing you almost never get end-to-end:

```
born → learns to read → specializes → learns to behave → learns to reason → gets compressed → goes to work → is judged
(arch)    (pretrain)      (mid-train)     (SFT / DPO)          (RL)              (quant)          (serving)      (eval)
```

The **same checkpoint travels through every module.** Each module ends on a *measured*
number — because this repo ships `bench` and 290+ tests, "what's the gain?" is a command,
not a claim.

## How each module works — three tracks

Every module has the same shape. Pick your depth:

| Track | You do | Graded by |
|-------|--------|-----------|
| 📖 **Read** | Follow *why → theory → code → measure* in the module `README.md` | your understanding |
| 🔨 **Build** | Implement the core function in `lab_*.py` (starts as `NotImplementedError`) | **the repo's own tests** — green = correct, you can't fake it |
| 🚀 **Extend** | Open-ended project (add a variant, beat the number) | a PR |

The **Build** track is the key idea: the library `baby_whale_v4/` is the reference
*solution*, and its test suite is a free **autograder**. You implement a piece yourself;
`python course/NN-slug/lab_*_test.py` checks your version against the real one.

The course **never copies code** — it *narrates and links* the real `file:line` and
*imports* the real modules, so it can never drift from what's tested.

## The five beats of every module

1. **The wall** — what hurts *without* this feature? (motivation, made concrete)
2. **The idea** — the theory + intuition + the paper it comes from
3. **In the code** — the real implementation, annotated, linked to `file:line`
4. **The payoff, measured** — run an ablation, watch the number move
5. **Break it** — change a knob, re-run, discover the tradeoff yourself

Where there's real math, a **🧩 From theory to code** table bridges the equation to the
exact code — term by term, with a *why* for each — and a **Build** lab (10 so far: RMSNorm,
cross-entropy, attention, RoPE, MLA, MoE routing, DPO, GRPO, speculative accept, KV-cache
append) has you write it. That pairing is how this course answers *"why this, and how does
the idea become code?"* — a **🧩 From theory to code** table sits in **every** module, its
left column adapting to the feature: an equation, a mechanism, a recipe, a data structure,
a protocol, or a pipeline.

## Three lenses, woven through every module

This is **not only an ML-*systems* course** (that's [TinyTorch](https://mlsysbook.ai/tinytorch/)),
nor only a *modeling* course (that's nanoGPT). A real LLM is all of these at once, so each
module is read through whichever lenses fit it:

- **🧠 Modeling & theory** — *why* the idea works: the objective, the inductive bias, the
  failure mode (attention, MoE routing, MTP, DPO, GRPO…).
- **🎓 Training & alignment** — how *behavior* is shaped: pre → mid → SFT → DPO → RL.
- **🔬 Systems** — the real *cost*: params, memory, FLOPs, cache bytes, sparsity, via
  `course/systems.py` (MLA, KV cache, paged KV, quantization, batching…).

Plus **🎯 milestones** ([`MILESTONES.md`](MILESTONES.md)): a **Build** lab proves you know a
*piece*; a milestone proves the whole *system* works on a real task. Two are auto-gated.

## Progressive motivation

You don't meet MLA as a fact — you **feel its absence first.** The `course/presets.py`
configs let you turn features off and hit the wall before turning them on:

```
gpt-minimal → +mla → +compressed → +moe-balanced → +mtp → full
```

Train `gpt-minimal`, watch the KV cache blow up at serving time, flip on `+mla`, watch
bytes/token drop. That's *why we need this*, answered viscerally.

## The syllabus

> ⭐ = fully built exemplar (all three tracks). Others ship the **Read** track; Build/Extend
> stubs follow the same template (see [`CONTRIBUTING-A-MODULE.md`](CONTRIBUTING-A-MODULE.md)).

### Part 0 · The map
- **[00 · The map](00-the-map/)** — the lifecycle, and a **5-minute** train→generate run

### Part 1 · The architecture (born)
- **[01 · Backbone](01-backbone/)** — embedding, RMSNorm, residual block, LM head, loss
- **[02 · Attention basics](02-attention-basics/)** — causal, sliding window, MQA, partial RoPE
- **[03 · MLA ⭐](03-attention-mla/)** — low-rank latent KV: the cache-memory breakthrough
- **[04 · Compressed attention](04-attention-compressed/)** — HCA / CSA for long-range reach
- **[05 · Mixture of Experts](05-moe/)** — sparse experts + aux-loss-free load balancing
- **[06 · HyperConnect](06-hyperconnect/)** — learned multi-branch residuals (mHC)
- **[07 · Multi-token prediction](07-mtp/)** — extra heads that pay off at decode time

### Part 2 · Data
- **[08 · Tokenizer & data](08-tokenizer-and-data/)** — byte-BPE (the heap-encode story), packing

### Part 3 · Pre-training (learns to read)
- **[09 · Pre-training](09-pretraining/)** — loop, optimizer, grad-accum, checkpoint/resume, throughput

### Part 4 · Mid-training (specializes)
- **[10 · Mid-training](10-midtraining/)** — context extension, curriculum, annealing; the needle eval

### Part 5–7 · Post-training (learns to behave, then to reason)
- **[11 · SFT](11-sft/)** — chat templating, instruction tuning
- **[12 · DPO](12-dpo/)** — preference optimization, reference caching
- **[13 · RL with verifiable rewards](13-rl-grpo/)** — GRPO / RLOO, the code sandbox

### Part 8 · Inference & serving (goes to work)
- **[14 · KV cache](14-kv-cache/)** — the decode speedup that makes generation practical
- **[15 · Paged KV & offload](15-paged-kv-offload/)** — memory management for long context
- **[16 · Speculative decoding](16-speculative-decoding/)** — MTP draft + verify, bit-identical
- **[17 · Continuous batching](17-continuous-batching/)** — the scheduler, cohorts, ragged, the server

### Part 9 · Efficiency (gets compressed)
- **[18 · Quantization](18-quantization/)** — FP4, placement policies

### Part 10 · Evaluation (is judged)
- **[19 · Evaluation](19-evaluation/)** — bits-per-byte, code pass@1, IFEval, needle retrieval

### Part 11 · Frontier
- **[20 · Vision (VL2)](20-vision-vl2/)** — tiling, connector, model integration
- **[21 · Capstone](21-capstone/)** — take *your* model through the whole pipeline

## Prerequisites & setup

- **Apple Silicon Mac** (MLX/Metal) and **Python 3.14+** — see the repo root `README.md`.
- `uv sync --dev`, then everything runs via `uv run baby-whale-v4 <command>`.
- Comfort with Python and basic neural-net ideas (backprop, attention). No prior LLM
  systems knowledge assumed — that's what this builds.

Start at **[00 · The map](00-the-map/)**.
