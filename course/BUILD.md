# Build the LLM — the build track

Reading the course teaches you *how it works*. This track makes you **build it** — like
TinyTorch, you implement each real component yourself, and it's **graded against the actual
`baby_whale_v4` module**, not a toy. Then you assemble the pieces into a working model and
prove it on a real task.

> **What you build (and what you don't).** `baby_whale_v4` sits on **MLX**, which already
> gives you tensors, autograd, and matmul — so, unlike TinyTorch, you don't rebuild the
> framework. You build the **LLM**: the architecture, the training objectives, and the
> inference machinery, using MLX primitives. That's the layer that makes an LLM an LLM.

## How a build lab works

Each lab is a `lab_*.py` with a `NotImplementedError` stub and a **from-theory-to-code**
derivation in its docstring (math → code → *why*). You fill it in; running it grades your
version against the real thing:

```bash
uv run python course/01-backbone/lab_rmsnorm.py    # fails until you implement it
# ...edit the file...
uv run python course/01-backbone/lab_rmsnorm.py    # PASS ✅ = you built the real component
```

The graders live in `course/labs.py`; the course's own test suite proves every reference
solution passes its grader, so the autograder itself is verified.

## The build order

Follow it top to bottom — each step leans on the ones above (you can't fake the composition
step without the components under it).

### Phase 1 · Build a transformer layer
| # | Build | Lab |
|---|-------|-----|
| 1 | **RMSNorm** | `01-backbone/lab_rmsnorm.py` |
| 2 | **RoPE** (rotary positions) | `02-attention-basics/lab_rope.py` |
| 3 | **Scaled dot-product attention** | `02-attention-basics/lab_attention.py` |
| 4 | **SwiGLU expert** (the FFN) | `05-moe/lab_swiglu.py` |
| 5 | **⭐ Assemble the transformer layer** (composes 1–4) | `01-backbone/lab_transformer_layer.py` |

### Phase 2 · Architecture upgrades
| # | Build | Lab |
|---|-------|-----|
| 6 | **MLA** low-rank KV | `03-attention-mla/lab_mla.py` |
| 7 | **Block mean-pool** (HCA's compression) | `04-attention-compressed/lab_block_pool.py` |
| 8 | **MoE top-k routing** | `05-moe/lab_moe_route.py` |
| 9 | **HyperConnect** learned stream mix | `06-hyperconnect/lab_hyperconnect.py` |
| 10 | **MTP head** | `07-mtp/lab_mtp.py` |

### Phase 3 · Data, training & alignment
| # | Build | Lab |
|---|-------|-----|
| 11 | **Byte-BPE encode** (must match the fast real one) | `08-tokenizer-and-data/lab_bpe.py` |
| 12 | **Cross-entropy** (the pre-training objective) | `09-pretraining/lab_cross_entropy.py` |
| 13 | **Document packing** (mid-training's data mechanic) | `10-midtraining/lab_packing.py` |
| 14 | **SFT response-only targets** | `11-sft/lab_sft_mask.py` |
| 15 | **DPO loss** | `12-dpo/lab_dpo.py` |
| 16 | **GRPO group advantage** | `13-rl-grpo/lab_grpo.py` |

### Phase 4 · Inference & serving
| # | Build | Lab |
|---|-------|-----|
| 17 | **KV-cache append** | `14-kv-cache/lab_kv_append.py` |
| 18 | **Paged-KV address translation** | `15-paged-kv-offload/lab_paged_location.py` |
| 19 | **Speculative acceptance** | `16-speculative-decoding/lab_spec_accept.py` |
| 20 | **Cohort grouping** (the batching rule) | `17-continuous-batching/lab_cohorts.py` |

### Phase 5 · Efficiency, evaluation & vision
| # | Build | Lab |
|---|-------|-----|
| 21 | **Quantize / dequantize** (absmax) | `18-quantization/lab_quantize.py` |
| 22 | **Bits-per-byte** | `19-evaluation/lab_bpb.py` |
| 23 | **Dynamic tiling grid** | `20-vision-vl2/lab_tiling.py` |

## Then: prove it works

Building a *piece* is one thing; the [**milestones**](MILESTONES.md) prove the *assembled
system* works on a real task — it learns, it remembers, it reasons. That's the payoff:
you took a model from components you wrote to a thing that actually does the job.

## Honest coverage

**Every content module (01–20) now has at least one graded build lab — 23 in total**, each
checked against the real module or the real formula. What "build" means varies honestly by
module: for the architecture you build the *actual component* (weight-shared against the
real one); for systems modules you build the *core rule* (the block-table translation, the
cohort key) rather than the whole threaded machinery. Deeper labs — CSA's top-k selection,
ragged masks, the full serving loop — are the natural next contributions: same mechanism,
a grader in `course/labs.py` compared to `baby_whale_v4`'s real code. See
[`CONTRIBUTING-A-MODULE.md`](CONTRIBUTING-A-MODULE.md).

**Start:** [00 · The map](00-the-map/README.md), then work the table above.
