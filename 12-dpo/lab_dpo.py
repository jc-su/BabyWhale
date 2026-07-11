"""LAB 12 — implement the DPO loss, then run me:

    uv run python course/12-dpo/lab_dpo.py

PASS means your loss matches the closed-form preference objective.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    """All args are per-example log-probs; the ``ref_*`` come from the frozen reference.

    Return the scalar loss::

        -mean(log sigmoid( beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)) ))

    That's the whole method — essentially one line. See README beat 2.
    """
    raise NotImplementedError("implement the DPO loss")


if __name__ == "__main__":
    from course.labs import grade_dpo

    grade_dpo(dpo_loss)
    print("PASS ✅  — you implemented Direct Preference Optimization.")
