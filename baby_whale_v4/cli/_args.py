"""Typed argument dataclass for the SFT CLI.

``argparse.Namespace`` gives every ``args.X`` access an ``Any`` type — typos
silently become ``None`` and refactors don't get caught. The SFT handler packs
its flags into a frozen dataclass at entry so the body is typed end-to-end:

    a = SFTArgs.from_namespace(args)
    # then use a.user, a.tokenizer_path, ... — typed, with parser/dataclass
    # drift caught by tests/test_cli_args.py.

SFT is currently the one subcommand wired this way; the other heavyweight
handlers read raw ``args`` directly.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Self


def _from_namespace[T](cls: type[T], ns: argparse.Namespace) -> T:
    """Build a frozen-dataclass instance from an ``argparse.Namespace``.

    Only fields declared on ``cls`` are consumed; extras on the namespace are
    ignored. A missing field raises the constructor's ``TypeError`` — a guarantee
    that the parser and the dataclass agree on the flag surface.
    """
    fields = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in vars(ns).items() if k in fields})


@dataclass(frozen=True)
class SFTArgs:
    user: list[str]
    assistant: list[str]
    chat_jsonl: str | None
    problems_jsonl: str | None
    tokenizer_path: str | None
    from_checkpoint: str | None
    block_size: int
    lr: float
    batch_size: int
    max_steps: int
    seed: int
    out_dir: str
    func: object = None
    command: str | None = None

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> Self:
        return _from_namespace(cls, ns)
