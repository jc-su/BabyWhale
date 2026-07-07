"""LAB 11 (build) — implement response-only SFT targets, then run me:

    uv run python course/11-sft/lab_sft_mask.py

Graded against the REAL `format_chat` output — your targets must mask exactly the
tokens the repo's chat template marks as non-response.

From theory to code
-------------------
  theory : train the model to PRODUCE responses, not to re-predict prompts — so the
           loss must only see response positions.
  math   : y_t = ids_{t+1} if mask_{t+1} = 1, else ignore_index
  code   : return [t if m == 1 else ignore_index
                   for t, m in zip(ids[1:], mask[1:])]

Why shift by one? next-token prediction: position t predicts token t+1, so both the
targets AND the mask move left by one. (`cross_entropy_ignore` then skips every
ignore_index position — see Module 09.)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def sft_targets(ids, mask, ignore_index):
    """``ids``/``mask``: same-length lists from ``format_chat`` (mask 1 = response token).

    Return the shifted target list with non-response positions set to ``ignore_index``.
    """
    raise NotImplementedError("shift by one, mask where the shifted mask is 0")


if __name__ == "__main__":
    from course.labs import grade_sft_targets

    grade_sft_targets(sft_targets)
    print("PASS ✅  — you implemented response-only SFT targets.")
