"""LAB 16 — implement speculative acceptance, then run me:

    uv run python course/16-speculative-decoding/lab_spec_accept.py

From theory to code
-------------------
  theory : keep drafted tokens only while they match greedy verification; stop at the
           first miss (everything after it was conditioned on a token you now reject).
  code   : accepted = 0
           for i in range(k):
               if draft[i] != verify[i]:
                   break
               accepted += 1
           return accepted

Why stop at the first mismatch? the accepted prefix must be *identical* to what plain
greedy decoding would have produced — that equality is exactly what makes speculation
a lossless speedup instead of an approximation.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def spec_accept(draft, verify):
    """``draft``, ``verify``: [k] token-id arrays. Return the count of accepted tokens."""
    raise NotImplementedError("implement the accept-longest-correct-prefix rule")


if __name__ == "__main__":
    from course.labs import grade_spec_accept

    grade_spec_accept(spec_accept)
    print("PASS ✅  — you implemented speculative acceptance.")
