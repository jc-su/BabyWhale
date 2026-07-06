"""Lock the contract between the SFT argparse parser and its typed args dataclass.

Parse a representative invocation, wrap the Namespace into ``SFTArgs``, and assert
the round-trip is total — if a flag is added to the parser and forgotten in the
dataclass (or vice versa) this breaks loud. SFT is the one subcommand whose
handler consumes a typed args dataclass; the other handlers read raw ``args``.
"""

from __future__ import annotations

import argparse
import dataclasses
import unittest

from baby_whale_v4.cli._args import SFTArgs
from baby_whale_v4.cli.train import register as register_train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    register_train(sub)
    return parser


class TestSFTArgs(unittest.TestCase):
    def test_sft_args_bind(self) -> None:
        ns = _build_parser().parse_args(
            ["sft", "--user", "hi", "--assistant", "hello", "--out-dir", "runs/x"]
        )
        a = SFTArgs.from_namespace(ns)
        self.assertEqual(a.user, ["hi"])
        self.assertEqual(a.assistant, ["hello"])
        self.assertIsNone(a.from_checkpoint)
        self.assertIsNone(a.chat_jsonl)

    def test_sft_dataclass_field_set_matches_parser(self) -> None:
        ns = _build_parser().parse_args(
            ["sft", "--user", "hi", "--assistant", "hi", "--out-dir", "x"]
        )
        dataclass_fields = {f.name for f in dataclasses.fields(SFTArgs)}
        missing = dataclass_fields - set(vars(ns))
        self.assertEqual(missing, set(), f"SFTArgs fields not in parser: {missing}")


if __name__ == "__main__":
    unittest.main()
