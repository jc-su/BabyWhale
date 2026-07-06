"""Drift guard for the code shown in each module's "In the code" beat.

Every identifier the course *displays* must exist in the real source (so a rename
can't silently make the docs lie) and must actually appear in the module README
(so the map stays in sync with what's shown). This is what lets the course paste
real code without it drifting from the implementation.
"""

from __future__ import annotations

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# module slug -> (source file under baby_whale_v4/, identifiers shown in beat 3)
SHOWN: dict[str, tuple[str, tuple[str, ...]]] = {
    "01-backbone": ("model.py", ("hc.consume", "self.ln_1", "self.attn", "hc.produce")),
    "02-attention-basics": (
        "attention.py",
        ("q_proj", "k_proj", "self.rope", "swapaxes(-2, -1)", "_masked_softmax", "sliding_window"),
    ),
    "03-attention-mla": ("attention.py", ("kv_a_proj", "append_latent", "kv_b_proj")),
    "04-attention-compressed": (
        "attention.py",
        ("_block_mean_pool", "comp_allowed", "raw_allowed", "sliding_window"),
    ),
    "05-moe": ("moe.py", ("nn.softplus", "_bias_array", "mx.argsort", "take_along_axis")),
    "06-hyperconnect": (
        "mhc.py",
        ("input_logits", 'einsum("btkd,k->btd"', "sinkhorn", "write_logits"),
    ),
    "07-mtp": ("mtp.py", ("self.head", "nn.silu", "self.transform")),
}


class TestCourseSnippets(unittest.TestCase):
    def test_shown_code_matches_source(self) -> None:
        for module, (src, identifiers) in SHOWN.items():
            readme = (ROOT / "course" / module / "README.md").read_text()
            source = (ROOT / "baby_whale_v4" / src).read_text()
            for ident in identifiers:
                self.assertIn(
                    ident, source, f"{module}: `{ident}` shown but not in {src} (renamed?)"
                )
                self.assertIn(
                    ident, readme, f"{module}: `{ident}` mapped but not shown in the README"
                )


if __name__ == "__main__":
    unittest.main()
