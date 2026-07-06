"""Byte-BPE encode: output identical to the ordered-merge reference, and fast."""

from __future__ import annotations

import time
import unittest

from baby_whale_v4.data.tokenizer import _BPE_BASE_VOCAB, train_byte_bpe


def _reference_encode(text: str, merges: tuple[tuple[int, int], ...]) -> list[int]:
    """Original O(len x n_merges) algorithm: apply each learned merge globally, in order."""
    ids = list(text.encode("utf-8"))
    for idx, pair in enumerate(merges):
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(_BPE_BASE_VOCAB + idx)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids


class TestByteBPEEncode(unittest.TestCase):
    def _tok(self, vocab: int = 400):
        corpus = [
            "def foo(x): return x + 1",
            "class Bar(object): pass",
            "for i in range(n): total += arr[i]",
            "import os\nimport sys\n",
            "the quick brown fox " * 3,
        ]
        return train_byte_bpe(corpus, vocab_size=vocab)

    def test_encode_matches_ordered_merge_reference(self) -> None:
        tok = self._tok()
        for text in [
            "def foo(x): return x + 1",
            "class Baz: total = 0",
            "the quick brown fox jumps",
            "unseen text with symbols !@#",
            "",
            "a",
        ]:
            self.assertEqual(tok.encode(text), _reference_encode(text, tok.merges), text)
            self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_encode_fast_on_long_line(self) -> None:
        # The old encoder hung for minutes on long lines; the rank-based one is quick.
        tok = self._tok(vocab=600)
        long_line = "def compute(a, b): return a * b + a - b  " * 400  # ~16k chars
        start = time.time()
        enc = tok.encode(long_line)
        self.assertLess(time.time() - start, 3.0)
        self.assertEqual(tok.decode(enc), long_line)

    def test_special_wrapping(self) -> None:
        tok = self._tok()
        enc = tok.encode("hi", add_bos=True, add_eos=True)
        self.assertEqual(enc[0], tok.bos_id)
        self.assertEqual(enc[-1], tok.eos_id)


if __name__ == "__main__":
    unittest.main()
