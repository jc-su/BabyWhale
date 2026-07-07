"""LAB 17 (build) — implement cohort grouping, then run me:

    uv run python course/17-continuous-batching/lab_cohorts.py

The exact grouping rule the scheduler ticks on: requests share ONE batched forward
only if their caches are the same length AND they sample the same way.

From theory to code
-------------------
  theory : a batched forward stacks caches -> every row must have the same length; and
           one sampling pass serves the whole batch -> same temperature/top-k/....
  code   : groups = {}
           for rid, length, sig in requests:
               groups.setdefault((length, sig), []).append(rid)
           return groups

Why both keys? mixed lengths break the stacked cache (that's what ragged batching
fixes, per-row masks and all); mixed sampling breaks the shared sampling pass.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def form_cohorts(requests):
    """``requests``: iterable of ``(request_id, cache_length, sampling_sig)``.

    Return ``{(length, sig): [request_id, ...]}`` — the cohorts.
    """
    raise NotImplementedError("group by the (length, sampling signature) key")


if __name__ == "__main__":
    from course.labs import grade_form_cohorts

    grade_form_cohorts(form_cohorts)
    print("PASS ✅  — you implemented the scheduler's cohort rule.")
