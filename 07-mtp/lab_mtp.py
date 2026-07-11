"""LAB 07 (build) — implement the MTP head, then run me:

    uv run python course/07-mtp/lab_mtp.py

Graded against the REAL `baby_whale_v4.mtp.MTPHead`.

From theory to code
-------------------
  theory : project the *hidden state* (not the token id) to a future token's logits.
  math   : logits = Head( silu(Transform(h)) )
  code   : t = transform(h);  return head(t * sigmoid(t))

Why from the hidden state? it carries the rich context the main LM head uses, so the
head is a good enough *draft* for speculative decoding (Module 16) — the Medusa recipe.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def mtp_head(h, transform, head):
    """``h``: [.., d] hidden state. ``transform``/``head``: linear modules. Return logits."""
    raise NotImplementedError("implement head(silu(transform(h)))")


if __name__ == "__main__":
    from course.labs import grade_mtp_head

    grade_mtp_head(mtp_head)
    print("PASS ✅  — you implemented the real MTP head.")
