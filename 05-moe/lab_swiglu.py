"""LAB 05 (build) — implement the SwiGLU expert FFN, then run me:

    uv run python course/05-moe/lab_swiglu.py

Graded against the REAL `baby_whale_v4.layers.SwiGLUExpert` — build the actual component.

From theory to code
-------------------
  theory : a gated MLP — one branch gates the other, then project down.
  math   : y = W_down( silu(clip(W_gate·x)) ⊙ clip(W_up·x) )
  code   : gate = clip(w_gate(x), -c, c);  up = clip(w_up(x), -c, c)
           return w_down( (gate * sigmoid(gate)) * up )   # silu(g) = g * sigmoid(g)

Why gating (vs a plain MLP)? the multiplicative gate lets the network modulate the
up-projection per feature — empirically better than ReLU/GELU FFNs at the same size.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def swiglu(x, w_gate, w_up, w_down, clamp):
    """``x``: [.., d]. ``w_gate``/``w_up``/``w_down``: linear modules. ``clamp``: float.

    Return the expert output. Follow the two code lines above (silu = x·sigmoid(x)).
    """
    raise NotImplementedError("implement the gated MLP")


if __name__ == "__main__":
    from course.labs import grade_swiglu

    grade_swiglu(swiglu)
    print("PASS ✅  — you implemented the real SwiGLU expert.")
