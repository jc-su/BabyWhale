"""LAB 18 (build) — implement symmetric absmax quantize/dequantize, then run me:

    uv run python course/18-quantization/lab_quantize.py

The core low-bit round-trip. (baby_whale_v4 ships group-affine / NVFP4; this is the
simplest real variant — same idea, one scale for the whole tensor.)

From theory to code
-------------------
  theory : squeeze weights onto a small integer grid, keep a scale to undo it.
  math   : qmax  = 2^(bits-1) - 1
           scale = max|w| / qmax
           w_q   = clip(round(w / scale), -qmax, qmax)      # the stored 4-bit codes
           ŵ     = w_q * scale                              # dequantized at matmul
  code   : follow the four lines above.

Why does it work? the round-trip error is at most half a step (scale/2), so with enough
levels the model barely notices — 4x less memory to move for ~the same product.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def quantize_dequantize(w, bits):
    """``w``: weight array. ``bits``: e.g. 4. Return the dequantized ``ŵ`` (same shape)."""
    raise NotImplementedError("scale by max|w|/qmax, round to the grid, scale back")


if __name__ == "__main__":
    from course.labs import grade_quantize_dequantize

    grade_quantize_dequantize(quantize_dequantize)
    print("PASS ✅  — you implemented symmetric absmax quantization.")
