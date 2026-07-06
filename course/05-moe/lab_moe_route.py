"""LAB 05 — implement top-k expert routing, then run me:

    uv run python course/05-moe/lab_moe_route.py

From theory to code
-------------------
  theory : send each token to its k best experts, weighted by a softmax over just those k.
  math   : idx  = top_k(logits)            # the k highest-scoring experts
           gates = softmax(logits[idx])     # softmax OVER THE k, so they sum to 1
  code   : order = argsort(-logits, axis=-1)[:, :k]
           top   = take_along_axis(logits, order, axis=-1)
           return order, softmax(top, axis=-1)

Why softmax over the top-k (not all E)? the gates weight the experts you actually run,
so they must sum to 1 over exactly those k — then the combine is a proper weighted average.
Why this at all? sparsity: N experts' worth of parameters at only k experts' worth of FLOPs.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def moe_route(logits, k):
    """``logits``: [T, E] router scores. Return ``(indices [T, k], gates [T, k])``.

    Follow the three code lines above.
    """
    raise NotImplementedError("implement top-k routing")


if __name__ == "__main__":
    from course.labs import grade_moe_route

    grade_moe_route(moe_route)
    print("PASS ✅  — you implemented top-k expert routing.")
