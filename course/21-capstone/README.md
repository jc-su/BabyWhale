# 21 · Capstone — the whole life of one model

You've studied every leg in isolation. Now walk **one model** through all of them and
watch it change at each stage. This is the thing almost no course lets you do: take a
single set of weights from noise to a served, reasoning, evaluated system — and *measure*
it the whole way.

## The journey

| Stage | Do | Watch it change |
|-------|----|-----------------|
| **Born** | Pick a preset — climb `gpt-minimal → full` (`course/presets.py`) | more capable architecture, more params |
| **Reads** | Pre-train (Module 09) | training loss falls; **bits-per-byte** drops (Module 19) |
| **Specializes** | Mid-train: extend context (Module 10) | **needle retrieval** accuracy rises |
| **Behaves** | SFT (Module 11) | answers instructions in-format instead of rambling |
| **Prefers** | DPO (Module 12) | chosen-vs-rejected margin grows |
| **Reasons** | GRPO on code (Module 13) | **pass@1** climbs |
| **Compresses** | Quantize (Module 18) | size ↓, tokens/sec ↑, quality ~held |
| **Works** | Serve it (Modules 14–17) | streams tokens to many clients at once |
| **Judged** | Eval at every step (Module 19) | you have a number for each transition |

The point is the **continuity**: it's the same checkpoint the whole way, and every arrow
is a measurement you can reproduce.

## Three ways to run it

- **📖 Read** — follow the table, running each stage's `--help` command and eval.
- **🔨 Build** — before each stage, do that module's `lab_*.py` so you've implemented the
  core of what you're about to run.
- **🚀 Extend** — change one thing (an attention schedule, the expert count, the RL group
  size) and re-run the whole thread. Did the end-to-end system get better? That's a real
  experiment — and a real PR.

## What "done" looks like

You can point at a single model and say: *I chose its architecture, taught it to read,
extended its memory, aligned its behavior, gave it a verifiable skill, shrank it, served
it, and measured every step.* That's the whole modern LLM lifecycle — and you drove it.

## Where to go next

- Contribute a module or a lab — see [`../CONTRIBUTING-A-MODULE.md`](../CONTRIBUTING-A-MODULE.md).
- Push a frontier: finish the vision encoder + VL training (Module 20), or extend ragged
  batching to the compressed-attention layers (Module 17). The open edges are documented
  in `refs/IMPLEMENTATION_STATUS.md`.

**Back to the [start](../README.md).**
