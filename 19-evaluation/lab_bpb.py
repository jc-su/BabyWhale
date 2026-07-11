"""LAB 19 (build) — implement bits-per-byte, then run me:

    uv run python course/19-evaluation/lab_bpb.py

The same formula `eval-bpb` uses (`cli/eval.py`).

From theory to code
-------------------
  theory : perplexity depends on the tokenizer (fewer, bigger tokens => lower loss per
           token). Normalizing by BYTES makes models with different tokenizers comparable.
  math   : bpb = (mean nats per token / ln 2) * (tokens per byte)
  code   : mean_loss_nats  = total_loss_nats / total_tokens
           tokens_per_byte = total_tokens / total_bytes
           return (mean_loss_nats / math.log(2)) * tokens_per_byte

Why is 8.0 the magic baseline? a model that knows nothing assigns 1/256 to every next
byte: loss = ln 256 nats = 8 bits. Below 8 bpb, the model is genuinely compressing text.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def bits_per_byte(total_loss_nats, total_tokens, total_bytes):
    """Return the bits-per-byte score for a summed next-token loss (in nats)."""
    raise NotImplementedError("nats/token -> bits/token -> bits/byte")


if __name__ == "__main__":
    from course.labs import grade_bpb

    grade_bpb(bits_per_byte)
    print("PASS ✅  — you implemented bits-per-byte.")
