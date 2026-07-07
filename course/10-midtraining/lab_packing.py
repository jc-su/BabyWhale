"""LAB 10 (build) — implement document packing, then run me:

    uv run python course/10-midtraining/lab_packing.py

Graded against the REAL `baby_whale_v4.data.dataset.PackedDataset` token stream.
Packing is mid-training's data mechanic: extending context is useless unless the
training windows actually contain long, contiguous documents.

From theory to code
-------------------
  theory : concatenate documents into one stream with boundary markers, then cut the
           stream into fixed windows so no compute is spent on padding.
  math   : stream = [BOS d₁ EOS][BOS d₂ EOS]...;  usable = (len-1) // block_size
  code   : flat = []
           for doc in docs:        # skip empty docs
               flat.append(bos_id); flat.extend(doc); flat.append(eos_id)
           usable = (len(flat) - 1) // block_size
           return flat[: usable * block_size + 1]     # +1: targets are shifted by one

Why the `+1`? a window of block_size inputs needs block_size targets — the stream keeps
one extra token so `y = stream[i+1 : i+1+block]` always exists.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def pack_documents(docs, block_size, bos_id, eos_id):
    """``docs``: list of token-id lists. Return the packed flat stream (a list of ints)."""
    raise NotImplementedError("concatenate with BOS/EOS, truncate to usable*block_size + 1")


if __name__ == "__main__":
    from course.labs import grade_pack_documents

    grade_pack_documents(pack_documents)
    print("PASS ✅  — you implemented document packing.")
