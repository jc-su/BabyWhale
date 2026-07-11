"""LAB 03 — implement the MLA low-rank KV round-trip, then run me:

    uv run python course/03-attention-mla/lab_mla.py

PASS means your two matmuls match exactly what baby_whale_v4's MLA layer does
with its cached latent. See README.md, beat 3, for the theory.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def mla_roundtrip(kv, w_down, w_up):
    """Compress and reconstruct.

    Args:
        kv:      [T, d_kv]  — the thing we'd otherwise cache in full
        w_down:  [d_kv, r]  — projects down to a rank-r latent
        w_up:    [r, d_kv]  — reconstructs an approximation of kv

    Return ``(latent, reconstructed)`` where ``latent`` is [T, r] (the small
    thing the cache actually stores) and ``reconstructed`` is [T, d_kv].

    Two matmuls. That's it — that's the trick the whole MLA cache saving rests on.
    """
    raise NotImplementedError("implement me — delete this line and write the two matmuls")


if __name__ == "__main__":
    from course.labs import grade_mla_roundtrip

    grade_mla_roundtrip(mla_roundtrip)
    print("PASS ✅  — you implemented the core of MLA.")
