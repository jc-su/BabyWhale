"""LAB 02 — implement RoPE (rotate-half), then run me:

    uv run python course/02-attention-basics/lab_rope.py

PASS means your rotation matches the one baby_whale_v4 applies to q and k.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def apply_rope(x, cos, sin):
    """Rotate the feature pairs of ``x`` by the given ``cos``/``sin`` (same shape as x).

    Split x's last dim in half; the "rotate-half" of ``[x1, x2]`` is ``[-x2, x1]``.
    Return ``x * cos + rotate_half(x) * sin``. At position 0 (cos=1, sin=0) it's a no-op.
    """
    raise NotImplementedError("implement the rotate-half formula — see README beat 2")


if __name__ == "__main__":
    from course.labs import grade_rope

    grade_rope(apply_rope)
    print("PASS ✅  — you implemented rotary position embedding.")
