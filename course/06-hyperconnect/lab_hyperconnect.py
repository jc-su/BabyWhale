"""LAB 06 (build) — implement HyperConnect's `consume`, then run me:

    uv run python course/06-hyperconnect/lab_hyperconnect.py

Graded against the REAL `baby_whale_v4.mhc.HyperConnect.consume`.

From theory to code
-------------------
  theory : a sublayer reads a *learned* weighted mix of the parallel residual streams.
  math   : w = softmax(logits) over the hc_mult streams;  out = Σ_k w_k · h[:, :, k, :]
  code   : weights = softmax(input_logits, axis=-1)
           return einsum("btkd,k->btd", h, weights)

Why learned weights (vs a plain mean)? each layer chooses *which* streams to read — the
generalization of the fixed residual that Module 06 is about.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def hc_consume(h, input_logits):
    """``h``: [B, T, hc_mult, D] parallel streams. ``input_logits``: [hc_mult].

    Return [B, T, D] — the softmax-weighted mix of the streams.
    """
    raise NotImplementedError("softmax the logits, then einsum the streams")


if __name__ == "__main__":
    from course.labs import grade_hc_consume

    grade_hc_consume(hc_consume)
    print("PASS ✅  — you implemented HyperConnect's learned stream mix.")
