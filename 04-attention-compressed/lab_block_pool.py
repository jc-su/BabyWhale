"""LAB 04 (build) — implement HCA's block mean-pool, then run me:

    uv run python course/04-attention-compressed/lab_block_pool.py

Graded against the REAL `baby_whale_v4.attention._block_mean_pool`.

From theory to code
-------------------
  theory : summarize each full block of keys into ONE mean vector; a token too far
           back to see raw keys can still attend to these cheap summaries.
  math   : k̄_b = mean(k[b·B : (b+1)·B])  for each full block b;  drop the partial tail.
  code   : n_full = T // block_size
           keep   = n_full * block_size
           pooled = mean(x[:, :, :keep, :].reshape(B, H, n_full, block_size, D), axis=3)
           return pooled, n_full          # (zeros, 0) when T < block_size

Why drop the trailing partial block? those newest tokens are inside the sliding window,
so they're already attended in full — summarizing them would double-count.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def block_mean_pool(x, block_size):
    """``x``: [B, H, T, D] keys. Return ``(pooled [B, H, n_full, D], n_full)``."""
    raise NotImplementedError("reshape full blocks to [.., n_full, block_size, D] and mean axis 3")


if __name__ == "__main__":
    from course.labs import grade_block_pool

    grade_block_pool(block_mean_pool)
    print("PASS ✅  — you implemented HCA's block compression.")
