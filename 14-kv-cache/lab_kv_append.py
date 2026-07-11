"""LAB 14 — implement the KV-cache append, then run me:

    uv run python course/14-kv-cache/lab_kv_append.py

From theory to code
-------------------
  theory : never recompute the past — keep every past token's key/value and append the
           new one, so decoding token t is O(1) attention work, not O(t).
  math   : K_cache <- [K_cache ; k_new]        (concatenate along the time axis)
  code   : return concatenate([k_cache, k_new], axis=-2)   # [B, H, T, D] grows in T

Why axis=-2? the layout is [batch, heads, time, head_dim]; time is the axis that grows
by one each decode step. That single concatenate is the whole reason generation is fast.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def kv_append(k_cache, k_new):
    """``k_cache``: [B, H, T, D]; ``k_new``: [B, H, 1, D]. Return [B, H, T+1, D]."""
    raise NotImplementedError("implement the append — one concatenate")


if __name__ == "__main__":
    from course.labs import grade_kv_append

    grade_kv_append(kv_append)
    print("PASS ✅  — you implemented the KV-cache append.")
