"""LAB 01 — implement RMSNorm, then run me:

    uv run python course/01-backbone/lab_rmsnorm.py

From theory to code
-------------------
  theory : normalize each vector by its root-mean-square, then rescale per feature.
  math   : y = x / sqrt(mean(x**2) + eps) * weight
  code   : ms = mean(x*x, last axis, keepdims=True);  y = x * rsqrt(ms + eps) * weight

Why this (and not LayerNorm)? LayerNorm also subtracts the mean; transformers don't
need that centering, so RMSNorm drops it — fewer ops, same quality.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def rms_norm(x, weight, eps):
    """Normalize ``x`` [.., d] by its per-row RMS, then multiply by ``weight`` [d].

    Return ``x * rsqrt(mean(x**2, axis=-1) + eps) * weight``. See the derivation above.
    """
    raise NotImplementedError("implement RMSNorm — one line from the math")


if __name__ == "__main__":
    from course.labs import grade_rms_norm

    grade_rms_norm(rms_norm)
    print("PASS ✅  — you implemented RMSNorm.")
