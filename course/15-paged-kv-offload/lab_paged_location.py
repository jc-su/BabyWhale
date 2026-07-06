"""LAB 15 (build) — implement paged-KV address translation, then run me:

    uv run python course/15-paged-kv-offload/lab_paged_location.py

This is the exact rule `PagedKVCache` uses: `keys[blocks[t // bs], :, t % bs, :]`.

From theory to code
-------------------
  theory : the block table maps a logical token position to a physical (block, offset),
           just like an OS page table maps virtual to physical memory.
  math   : block  = block_table[t // block_size]
           offset = t mod block_size
  code   : return blocks[t // block_size], t % block_size

Why paging? blocks let different-length requests share one pool with no per-request
over-reservation and little fragmentation (the vLLM idea).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def paged_location(blocks, t, block_size):
    """``blocks``: list mapping logical block -> physical block. ``t``: token position.

    Return ``(physical_block, offset_within_block)``.
    """
    raise NotImplementedError("translate token t through the block table")


if __name__ == "__main__":
    from course.labs import grade_paged_location

    grade_paged_location(paged_location)
    print("PASS ✅  — you implemented paged-KV address translation.")
