"""Reproduce every measured number in the paper's Figure 5 / Tables 2-3.

    uv run python paper/experiments/run_experiments.py

Runs three experiments on the local machine (Apple Silicon, MLX) and writes
paper/experiments/results.json plus pgfplots-ready coordinate blocks:

  A. Preset ladder (Table 2) — train each course preset (gpt-minimal -> full)
     for the same steps on the same corpus (the course's own Markdown,
     byte-level); report main-loss curves, params, tokens/sec.
  B. Needle-vs-depth cliff — EXPLORATORY, NOT REPORTED in the paper: with
     fresh-sampled training batches at these tiny scales the retrieval task
     does not train (final CE ~ ln(vocab) for both depths), so no cliff can be
     honestly shown; kept here as a negative result and a starting point.
  C. Inference microbenchmarks (Section 5) — cohort-batched decode throughput
     vs. group size (and cached/uncached decode, which at this toy scale is
     kernel-launch bound and therefore also not reported).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

from baby_whale_v4 import BabyWhaleV4Model  # noqa: E402
from baby_whale_v4.inference.batched import decode_group_batched  # noqa: E402
from baby_whale_v4.inference.engine import Engine, GenerationOptions  # noqa: E402
from baby_whale_v4.typing import RequestId  # noqa: E402
from baby_whale_v4.training.mlx_optim import AdamW  # noqa: E402
from course.presets import LADDER, load_preset  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results.json"


# ---------------------------------------------------------------- corpus ----
def corpus_ids() -> list[int]:
    """Byte ids (0-255) of the course's own Markdown — self-contained data."""
    text = "\n\n".join(
        p.read_text() for p in sorted((ROOT / "course").rglob("*.md"))
    )
    return list(text.encode("utf-8"))


def windows(ids: list[int], seq: int, batch: int, step: int, seed: int) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed * 100003 + step)
    starts = rng.integers(0, len(ids) - seq - 1, size=(batch,))
    x = np.stack([ids[s : s + seq] for s in starts])
    y = np.stack([ids[s + 1 : s + seq + 1] for s in starts])
    return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ---------------------------------------------------- A. preset ladder ------
def run_ladder(steps: int = 240, batch: int = 8, seq: int = 96) -> dict:
    ids = corpus_ids()
    results = {}
    for name in LADDER:
        mx.random.seed(0)
        cfg = load_preset(name)
        model = BabyWhaleV4Model(cfg)
        n_params = model.num_parameters()

        def loss_fn(m, x, y):
            out = m(x, targets=y)
            assert out.main_loss is not None
            return out.main_loss  # next-token CE only: comparable across presets

        lag = nn.value_and_grad(model, loss_fn)
        opt = AdamW(learning_rate=3e-3)
        curve = []
        t0 = time.perf_counter()
        for step in range(steps):
            x, y = windows(ids, seq, batch, step, seed=1)
            loss, grads = lag(model, x, y)
            model.update(opt.step(model.parameters(), grads))
            mx.eval(model.parameters())
            if step % 20 == 0 or step == steps - 1:
                curve.append((step + 1, round(float(loss), 3)))
        dt = time.perf_counter() - t0
        toks = steps * batch * seq
        results[name] = {
            "params": n_params,
            "curve": curve,
            "final_loss": curve[-1][1],
            "tok_per_s": round(toks / dt),
            "seconds": round(dt, 1),
        }
        print(f"[ladder] {name}: {n_params:,} params, loss->{curve[-1][1]}, "
              f"{results[name]['tok_per_s']:,} tok/s ({dt:.0f}s)", flush=True)
    return results


# ------------------------------------------- B. needle vs. schedule ---------
def needle_batch(n: int, seq: int, vocab: int, dist: int | None, seed: int) -> tuple[mx.array, mx.array]:
    """Marker+answer planted `dist` tokens before the final query marker
    (random dist in [2, seq-3] when None)."""
    rng = np.random.default_rng(seed)
    low, marker = 3, 1
    x = rng.integers(low, vocab, size=(n, seq), dtype=np.int32)
    answers = rng.integers(low, vocab, size=(n,), dtype=np.int32)
    for i in range(n):
        d = int(rng.integers(2, seq - 3)) if dist is None else dist
        pos = seq - 1 - d
        x[i, pos] = marker
        x[i, pos + 1] = int(answers[i])
        x[i, seq - 1] = marker
    return mx.array(x), mx.array(answers)


def run_needle(steps: int = 600, seq: int = 48, vocab: int = 24) -> dict:
    """Retrieval reach of sliding-window attention scales with DEPTH: each layer
    relays information one window further, so a d-layer window-W model reaches
    ~d*W. We train 2- and 4-layer window-16 models and measure the cliff."""
    window = 16
    depths = {"2-layer": 2, "4-layer": 4}
    dists = [4, 8, 12, 16, 20, 24, 28, 36, 44]
    out: dict = {"window": window, "dists": dists, "acc": {}}
    for label, n_layer in depths.items():
        mx.random.seed(0)
        cfg = dataclasses.replace(
            load_preset("gpt-minimal"),
            vocab_size=vocab,
            n_layer=n_layer,
            layer_schedule=tuple(["sliding_mqa"] * n_layer),
        )
        model = BabyWhaleV4Model(cfg)

        def loss_fn(m, x, ans):
            return mx.mean(nn.losses.cross_entropy(m(x).logits[:, -1, :], ans))

        lag = nn.value_and_grad(model, loss_fn)
        opt = AdamW(learning_rate=3e-3)
        final_loss = 0.0
        for step in range(steps):
            x, ans = needle_batch(24, seq, vocab, dist=None, seed=step)
            loss, grads = lag(model, x, ans)
            model.update(opt.step(model.parameters(), grads))
            mx.eval(model.parameters())
            final_loss = float(loss)
        print(f"[needle] {label} train CE at end: {final_loss:.2f}", flush=True)
        model.eval()
        accs = []
        for d in dists:
            x, ans = needle_batch(128, seq, vocab, dist=d, seed=10_000 + d)
            preds = mx.argmax(model(x).logits[:, -1, :], axis=-1)
            accs.append(round(float(mx.mean(mx.equal(preds, ans).astype(mx.float32))), 3))
        out["acc"][label] = accs
        print(f"[needle] {label} (reach~{n_layer * window}): dist {dists} -> {accs}", flush=True)
    return out


# ------------------------------------------- C. inference microbench --------
def run_bench(gen: int = 64, prompt_len: int = 16, group: int = 4) -> dict:
    mx.random.seed(0)
    cfg = load_preset("gpt-minimal")
    model = BabyWhaleV4Model(cfg)
    model.eval()
    engine = Engine(model=model, config=cfg, tokenizer_hash="bench")
    prompt = list(range(3, 3 + prompt_len))
    opts = GenerationOptions(max_new_tokens=gen, mode="greedy")
    # warm Metal
    engine.generate(prompt, GenerationOptions(max_new_tokens=4, mode="greedy"))

    # cached decode
    t0 = time.perf_counter()
    engine.generate(prompt, opts)
    cached_s = time.perf_counter() - t0

    # uncached: re-run the full prefix each step
    seq = mx.array([prompt], dtype=mx.int32)
    t0 = time.perf_counter()
    for _ in range(gen):
        logits = model(seq).logits[:, -1, :]
        nxt = mx.argmax(logits, axis=-1).reshape(1, 1)
        mx.eval(nxt)
        seq = mx.concatenate([seq, nxt], axis=1)
    uncached_s = time.perf_counter() - t0

    # cohort batching: aggregate decode throughput vs. group size, end-to-end
    scaling = {}
    for size in (1, 2, 4, 8):
        prompts = [list(prompt) for _ in range(size)]
        t0 = time.perf_counter()
        if size == 1:
            engine.generate(prompts[0], opts)
        else:
            states = []
            for i, p in enumerate(prompts):
                state = engine.new_request(RequestId(f"b{size}-{i}"), p, opts)
                while state.remaining_prefill > 0:
                    engine.prefill_chunk(state, chunk_size=cfg.context_length)
                states.append(state)
            while not all(s.finished for s in states):
                decode_group_batched(model, states)
        dt = time.perf_counter() - t0
        scaling[size] = round(size * gen / dt)

    res = {
        "gen_tokens": gen,
        "cached_tok_s": round(gen / cached_s),
        "uncached_tok_s": round(gen / uncached_s),
        "batch_tok_s_by_group": scaling,
    }
    print(f"[bench] {res}", flush=True)
    return res


if __name__ == "__main__":
    t0 = time.perf_counter()
    results = {"ladder": run_ladder(), "needle": run_needle(), "bench": run_bench()}
    results["total_seconds"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT} in {results['total_seconds']}s", flush=True)
