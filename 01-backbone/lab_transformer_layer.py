"""LAB 01 (build) — ASSEMBLE a transformer layer from real components:

    uv run python course/01-backbone/lab_transformer_layer.py

This is the composition step — you wire the pieces into one working layer, and
it's graded against the REAL `baby_whale_v4` modules (RMSNorm, attention, SwiGLU),
not a toy. Green means you built a real transformer layer.

From theory to code
-------------------
  theory : pre-norm residual — normalize, transform, add back; twice per layer.
  math   : h   = x + Attn(RMSNorm₁(x))        # mix across tokens
           out = h + FFN(RMSNorm₂(h))          # think per token
  code   : h = x + attn(ln1(x));  return h + ffn(ln2(h))

Why pre-norm (norm *inside* the residual)? it keeps the residual stream un-normalized,
so gradients flow cleanly through depth. (This is the hc_mult=1 form; Module 06 upgrades
the plain `+` to a learned multi-branch residual.)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def transformer_layer(x, ln1, attn, ln2, ffn):
    """``x``: [B, T, d]. ``ln1``/``ln2``: RMSNorm; ``attn``/``ffn``: callable modules.

    Return the layer output (same shape as x). Two pre-norm residual steps.
    """
    raise NotImplementedError("assemble the two pre-norm residual steps")


if __name__ == "__main__":
    from course.labs import grade_transformer_layer

    grade_transformer_layer(transformer_layer)
    print("PASS ✅  — you assembled a real transformer layer.")
