"""LAB 02 — implement scaled dot-product attention, then run me:

    uv run python course/02-attention-basics/lab_attention.py

From theory to code
-------------------
  theory : each query reads a softmax-weighted mix of the values, weighted by how
           similar its query is to each key.
  math   : A = softmax( q·kᵀ / √d  +  mask ) · v        (mask: 0 = allowed, -inf = forbidden)
  code   : scores  = (q @ k.T) / d**0.5
           scores  = where(mask, scores, -1e9)          # forbid the future
           weights = softmax(scores, axis=-1)
           return    weights @ v

Why /√d?  q·k grows ~√d as the head dimension grows, which pushes softmax into
saturation (near-zero gradients). Dividing by √d keeps scores well-conditioned.
Why mask *before* softmax? so forbidden positions get ~0 weight, not merely a small one.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def attention(q, k, v, mask):
    """q, k, v: [T, d]; ``mask``: [T, T] boolean (True = allowed). Return [T, d].

    Follow the four code lines in the derivation above.
    """
    raise NotImplementedError("implement scaled dot-product attention")


if __name__ == "__main__":
    from course.labs import grade_attention

    grade_attention(attention)
    print("PASS ✅  — you implemented scaled dot-product attention.")
