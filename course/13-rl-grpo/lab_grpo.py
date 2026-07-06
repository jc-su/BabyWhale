"""LAB 13 — implement GRPO's group-relative advantage, then run me:

    uv run python course/13-rl-grpo/lab_grpo.py

PASS means "above the group average" becomes the learning signal — no value net.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def group_advantages(rewards):
    """``rewards``: shape ``[G]`` — rewards for a group of rollouts of one prompt.

    Return ``[G]`` advantages by normalizing within the group:
    ``(reward - group_mean) / (group_std + eps)`` with ``eps ~ 1e-8``. See README beat 2.
    """
    raise NotImplementedError("implement group normalization")


if __name__ == "__main__":
    from course.labs import grade_group_advantages

    grade_group_advantages(group_advantages)
    print("PASS ✅  — you implemented GRPO's group-relative advantage.")
