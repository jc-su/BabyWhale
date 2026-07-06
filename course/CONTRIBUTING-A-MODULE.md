# Writing a module

Every module is a folder `NN-slug/` with a `README.md` and (optionally) runnable
`lab_*.py` / `ablation.py` scripts. Module 03 (MLA) is the reference — copy its shape.

## The `README.md` — five beats, always in this order

1. **The wall** — the concrete pain *without* this feature. Make the reader want it.
2. **The idea** — theory + intuition + the paper. A tiny diagram beats a paragraph.
3. **In the code** — the *real* implementation, linked as `baby_whale_v4/file.py:LINE`.
   **Never paste library code** — narrate it and link. The repo is the single source
   of truth; docs point at it so they can't lie.
4. **The payoff, measured** — a command that prints a number (`ablation.py`, `bench`,
   or a test). No beat-4 claim without a runnable measurement.
5. **Break it & reflect** — one knob to turn, plus a *quantitative* systems question
   (memory / FLOPs / complexity), in the spirit of TinyTorch's reflection prompts.

Open with a **Prereqs · Unlocks** connection-map line so dependencies are explicit.
Where the module has a real systems cost, add a **🔬 Systems lens** callout built on
`course/systems.py`. If the module completes a lifecycle leg, tie it to a
[milestone](MILESTONES.md). End with a one-line **Next:** link.

**Answering "why this, and how does the idea become code?"** Every content module has a
**🧩 From theory to code** table between beats 2 and 3, mapping each piece of the idea to
the exact operation in the code, with a *why* per row. The left column **adapts to what the
feature is**: an **equation** for math (MLA, DPO), but a **mechanism** (HyperConnect), a
**recipe** (mid-training), a **data structure** (paged-KV), a **protocol** (batching), or a
**pipeline** (vision) otherwise. A systems concept earns the same term-by-term bridge as an
equation — that's the point, not forced math. Where it fits, a **Build** lab whose docstring
repeats the derivation has the learner write it.

## The three tracks

- **📖 Read** — the five beats above. Every module has this.
- **🔨 Build** *(optional but encouraged)* — a `lab_*.py` where the learner implements
  one core function that starts as `raise NotImplementedError`. Grade it with a
  function in `course/labs.py` that checks against a reference — **the repo's tests
  are the autograder.** The grader lives in `course/labs.py` (importable + tested);
  the learner stub lives in `NN-slug/lab_*.py` and bootstraps the repo root onto
  `sys.path` so `from course.labs import ...` works when run directly.
- **🚀 Extend** — 1–3 open-ended prompts that could become real PRs.

## Conventions

- **Importable logic** (things the test suite green-gates: presets, ablations,
  graders, the journey) lives in `course/*.py` — *not* in the `NN-slug/` folders,
  whose digit prefixes aren't valid Python module names.
- The `NN-slug/` scripts are thin runnable wrappers over that logic.
- Add the module to `course/README.md`'s syllabus and to `mkdocs.yml`'s nav.
- Anything runnable must pass `ruff` + `ty` + the course tests (`tests/test_course.py`).

## Build the docs site

```bash
uv run --with mkdocs-material mkdocs serve   # live preview at :8000
```
