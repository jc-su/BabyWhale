"""LAB 09 — implement cross-entropy with ignore-index, then run me:

    uv run python course/09-pretraining/lab_cross_entropy.py

From theory to code
-------------------
  theory : make the true next token likely; skip ignored (padding/prompt) positions.
  math   : L = -mean over kept t of  log softmax(logits_t)[target_t]
  code   : log_probs = logits - logsumexp(logits, -1, keepdims=True)   # stable log-softmax
           picked    = take_along_axis(log_probs, targets[:, None], -1)[:, 0]
           keep      = (targets != ignore_index)
           return -sum(picked * keep) / max(sum(keep), 1)

Why log-softmax via logsumexp (not log(softmax))? it's numerically stable — no exp
overflow. Why the ignore mask? padded/prompt tokens shouldn't dilute the average loss.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def cross_entropy(logits, targets, ignore_index):
    """``logits``: [N, V]; ``targets``: [N] int; skip positions equal to ``ignore_index``.

    Return the scalar mean negative log-likelihood over the kept positions.
    """
    raise NotImplementedError("implement cross-entropy — follow the four code lines above")


if __name__ == "__main__":
    from course.labs import grade_cross_entropy

    grade_cross_entropy(cross_entropy)
    print("PASS ✅  — you implemented the pre-training objective.")
