# Why the code is built this way

The modules explain each *feature*. This page explains the *engineering* — the recurring
decisions you'll see on every page of the source, and why an educational codebase makes
them differently than a production one. Every claim below points at real code.

## 1 · Fail fast, never substitute silently

A wrong-but-running system teaches the wrong lesson. So this codebase refuses instead of
guessing, everywhere:

```python
# layers.py — the quantization dispatch ends in:
case _:
    assert_never(self.quant_mode)     # unknown mode = crash, not "fall back to fp32"
```

- Configs validate in `__post_init__` and **raise** on anything inconsistent
  (`config/__init__.py`) — you cannot construct a model from a lie.
- Checkpoints carry a **config hash** and reject tampered/truncated files
  (`training/checkpoint.py`, `tests/test_checkpoint.py`).
- Unsupported paths raise `NotImplementedError` with the reason (e.g. ragged decode on a
  non-sliding schedule) rather than degrading quietly.

**Why:** when you're learning, a loud early failure *is* the documentation. Silent
fallbacks are how production systems accumulate mystery; a course can't afford mysteries.

## 2 · The interface is the diagram

Boundaries are typed so you can learn the *shape* of a subsystem without reading its body:

```python
# cache.py — attention doesn't care where KV lives:
class KVCache(Protocol):
    def append(self, layer_idx, key, value) -> KVPair: ...
    def sequence_length(self, layer_idx) -> int: ...
```

`DynamicKVCache` (Module 14) and the paged pool (Module 15) both satisfy it — which is
exactly the lesson: *paging changes storage, not attention*. The same pattern appears as
`TypedDict` payloads at the checkpoint/server edges and PEP 695 generics throughout.

**Why:** a `Protocol` is a diagram that type-checks. When the interface is honest, every
implementation swap (dynamic→paged, none→fp4) becomes a one-concept diff.

## 3 · Transparent beats clever

Attention is written as **explicit projections, masks, and softmax** — not a fused kernel:

```python
# attention.py — you can point at the causal and window constraints:
causal = key_pos <= q_pos
local = key_pos >= (q_pos - self.sliding_window + 1)
```

Likewise: cohort batching (same-length groups) came before ragged batching; the paged pool
is pure MLX ops, not a paged-attention kernel. The repo deliberately stops **at the
algorithm level** — where the idea lives — and leaves kernel fusion to production stacks.

**Why:** the performance *ideas* (windowing, latents, paging, batching) are learnable from
readable code; the last 10× of kernel engineering is not, and it buries the idea.

## 4 · Make illegal states unrepresentable — or untouchable

When something must not be trained, it's kept **out of the parameter tree by construction**:

```python
# moe.py — the aux-loss-free balancing bias:
@dataclass
class _RouterBalancerState:          # a plain dataclass, not an mx.array attribute —
    values: list[float]              # MLX's tree_flatten treats it as an opaque leaf,
                                     # so the optimizer and weight decay CAN'T touch it.
```

Same instinct elsewhere: the serving loop is the **only** thread that touches the model
(`serving.py`, Module 17) — thread-safety by architecture, not by locks sprinkled later.

**Why:** "the optimizer can't reach it" is a stronger guarantee than "we remember not to
update it" — and the technique itself (shaping data so tools can't misuse it) is a
transferable engineering lesson.

## 5 · Claims are tested equalities, not adjectives

Where this course says "bit-identical" or "numerically identical", there is a test
asserting the equality:

| Claim | The test |
|-------|----------|
| speculative decode ≡ greedy | `tests/test_inference_optimizations.py` |
| batched/ragged decode ≡ per-request | `tests/test_batched_scheduler.py` |
| DPO ref-caching ≡ recompute | `tests/test_dpo_cache.py` |
| text-only unchanged when vision is off | `tests/test_vision_integration.py` |
| fast BPE ≡ slow reference encoder | `tests/test_bpe_tokenizer.py` |
| docs' pasted code ≡ real source symbols | `tests/test_course_snippets.py` |

**Why:** optimizations that "should be equivalent" quietly stop being equivalent. Testing
the equivalence turns a claim into a property the CI enforces — including the claims made
by this course about its own snippets.

## 6 · Mac-first MLX, Python 3.14 — on purpose

- **MLX / Apple Silicon** because *laptop-runnable is learnable*: every ablation and lab
  in this course finishes in seconds on the machine you already have. (One real-world
  scar it will teach you: the first Metal dispatch on a background thread can hang, so
  the server warms the model on the main thread before spawning its loop —
  `serving.py:_warmup`.)
- **Python 3.14** as a deliberate showcase: PEP 758 (`except A, B:`), `type` statements,
  PEP 695 generics. The repo treats the modern language as part of the curriculum.

## 7 · Docs that cannot lie

Educational writing rots when the code moves. This course's defenses, in order of force:

1. Labs are **graded against the real modules** (`course/labs.py`) — a renamed API breaks
   the lab, not the learner's trust.
2. Pasted snippets are **drift-guarded** — `tests/test_course_snippets.py` fails CI if a
   shown identifier disappears from the source.
3. Measured beats are **runnable** — every "payoff" number comes from a command
   (`ablation.py`, `bench`, or a test), never from prose.

**Why:** the previous defense — "never paste code" — kept docs honest by keeping them
vague. Pasting + guarding keeps them honest *and* concrete.

---

*These aren't rules to admire — they're defaults to steal. If you adopt one thing, adopt
#5: any time you say "equivalent", write the test that makes it true.*
