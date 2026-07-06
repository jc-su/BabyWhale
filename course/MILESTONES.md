# Milestones — prove the system works

TinyTorch validates learning with *real tasks* (solve XOR, classify CIFAR) rather
than isolated unit tests: if the whole task works, everything underneath must have
composed correctly. These are baby_whale's version — each runs the actual model
end-to-end. Passing a milestone can't be faked by matching a reference: the **📖 Read**
and **🔨 Build** tracks prove you understand a *piece*; a milestone proves the *system*.

| # | Milestone | Proves the legs | How to verify | Auto-gated |
|---|-----------|-----------------|---------------|:---:|
| **A** | **It learns** | backbone + data + pre-training (01–09) | `uv run python course/00-the-map/journey.py` — loss falls | ✅ `tests/test_course.py` |
| **B** | **It remembers** | compressed attention + mid-training (04, 10) | needle retrieval beats chance after training | ✅ `tests/test_course.py` |
| **C** | **It follows** | SFT (11) | `uv run baby-whale-v4 sft …`, then prompt it — it answers in chat format | run it |
| **D** | **It reasons** | RL with verifiable rewards (13) | `uv run baby-whale-v4 grpo …`, then `eval-code` — pass@1 rises | run it |
| **E** | **It serves** | KV cache → batching (14–17) | `uv run python -m unittest tests.test_serving` — concurrent, token-identical | ✅ repo suite |

## Run the fast gates

```bash
uv run python -c "from course.milestones import FAST_MILESTONES; [print((r:=m()).passed, r.name, '—', r.evidence) for m in FAST_MILESTONES]"
```

Milestones A and B train real (tiny) models in seconds and are checked on every test
run. C–D–E need full SFT / RL / serving runs, so they're verified with the CLI and the
repo's own integration tests rather than re-run inline.

## Why this matters

A green **Build** lab tells you your `mla_roundtrip` matches the reference. A green
**milestone** tells you that MLA, MoE, RoPE, the training loop, the optimizer, the
tokenizer, and the eval *all work together well enough to do a real task*. That's the
difference between knowing the parts and having built the machine.
