"""LAB 08 (build) — implement byte-BPE encoding, then run me:

    uv run python course/08-tokenizer-and-data/lab_bpe.py

Your simple version must produce tokens IDENTICAL to the repo's heap-based
`_bpe_encode` — the exact equivalence `tests/test_bpe_tokenizer.py` enforces on
the real tokenizer.

From theory to code
-------------------
  theory : merges were learned in priority order; encoding replays them — always
           merge the lowest-rank (earliest-learned) adjacent pair present.
  code   : tokens = list(ids)
           loop:
             scan adjacent pairs; find the one with the LOWEST rank in `ranks`
             if none: return tokens
             replace that pair with the merged id: _BPE_BASE_VOCAB + rank

This O(n·m) scan is the honest baseline; Module 08's story is that the repo's
heap + linked-list version computes the SAME answer in O(n·log n) — which is the
difference between packing 15M tokens in ~52s and hanging forever.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def bpe_encode(ids, ranks):
    """``ids``: list of byte values. ``ranks``: dict (tok_a, tok_b) -> merge rank.

    Return the fully-merged token list (merged id = _BPE_BASE_VOCAB + rank).
    """
    raise NotImplementedError("repeatedly merge the lowest-rank adjacent pair")


if __name__ == "__main__":
    from course.labs import grade_bpe_encode

    grade_bpe_encode(bpe_encode)
    print("PASS ✅  — your encoder matches the real _bpe_encode exactly.")
