"""Tests for ContextCurriculum types and curriculum-mode pretrain."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config
from baby_whale_v4.data import ByteTokenizer, pack_normalized_jsonl
from baby_whale_v4.data.dataset import TensorPairDataset
from baby_whale_v4.training import (
    ContextCurriculum,
    CurriculumPhase,
    PretrainConfig,
    pretrain_with_curriculum,
)


class TestCurriculumPhase(unittest.TestCase):
    def test_valid_phase(self) -> None:
        phase = CurriculumPhase(context_length=384, n_tokens=50_000_000)
        self.assertEqual(phase.context_length, 384)
        self.assertEqual(phase.n_tokens, 50_000_000)

    def test_rejects_non_positive_context_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_length"):
            CurriculumPhase(context_length=1, n_tokens=1000)
        with self.assertRaisesRegex(ValueError, "context_length"):
            CurriculumPhase(context_length=0, n_tokens=1000)

    def test_rejects_non_positive_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_tokens"):
            CurriculumPhase(context_length=128, n_tokens=0)
        with self.assertRaisesRegex(ValueError, "n_tokens"):
            CurriculumPhase(context_length=128, n_tokens=-1)

    def test_phase_is_frozen(self) -> None:
        phase = CurriculumPhase(context_length=128, n_tokens=1000)
        with self.assertRaises(FrozenInstanceError):
            phase.__setattr__("context_length", 256)


class TestContextCurriculum(unittest.TestCase):
    def test_parse_basic(self) -> None:
        c = ContextCurriculum.parse("128:1000,256:2000")
        self.assertEqual(len(c.phases), 2)
        self.assertEqual(c.phases[0], CurriculumPhase(128, 1000))
        self.assertEqual(c.phases[1], CurriculumPhase(256, 2000))
        self.assertEqual(c.max_context_length, 256)
        self.assertEqual(c.total_tokens, 3000)

    def test_parse_human_suffixes(self) -> None:
        c = ContextCurriculum.parse("384:50M,768:50M,1536:1B")
        self.assertEqual(c.phases[0].n_tokens, 50_000_000)
        self.assertEqual(c.phases[2].n_tokens, 1_000_000_000)
        self.assertEqual(c.max_context_length, 1536)

    def test_parse_handles_fractional_suffixes(self) -> None:
        c = ContextCurriculum.parse("128:1.5K,256:2.5K")
        self.assertEqual(c.phases[0].n_tokens, 1500)
        self.assertEqual(c.phases[1].n_tokens, 2500)

    def test_parse_rejects_missing_separator(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing ':'"):
            ContextCurriculum.parse("128/1000")

    def test_parse_rejects_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty schedule"):
            ContextCurriculum.parse("")

    def test_rejects_decreasing_context_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            ContextCurriculum(
                phases=(
                    CurriculumPhase(256, 1000),
                    CurriculumPhase(128, 1000),
                )
            )

    def test_allows_equal_context_length(self) -> None:
        # Two phases at the same length are legal (e.g., two LR-cooldown phases).
        c = ContextCurriculum(
            phases=(
                CurriculumPhase(128, 1000),
                CurriculumPhase(128, 500),
            )
        )
        self.assertEqual(c.max_context_length, 128)
        self.assertEqual(c.total_tokens, 1500)

    def test_rejects_empty_phases(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one phase"):
            ContextCurriculum(phases=())


class TestPretrainWithCurriculum(unittest.TestCase):
    def test_curriculum_pretrain_runs_two_phases(self) -> None:
        mx.random.seed(0)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            tok = ByteTokenizer()
            jsonl = tmp / "train.jsonl"
            jsonl.write_text(
                "\n".join(
                    json.dumps({"kind": "pretrain", "text": "abcdefghij" * 20}) for _ in range(40)
                )
                + "\n",
                encoding="utf-8",
            )

            curriculum = ContextCurriculum.parse("16:200,32:200")
            cfg = BabyWhaleV4Config.tiny(
                vocab_size=tok.vocab_size,
                context_length=curriculum.max_context_length,
            )

            def build_dataset(block_size: int) -> TensorPairDataset:
                return pack_normalized_jsonl(jsonl, tokenizer=tok, block_size=block_size)

            out = tmp / "run"
            pretrain_with_curriculum(
                config=cfg,
                pretrain_config=PretrainConfig(
                    lr=1e-3,
                    optimizer="adamw",
                    max_steps=200,
                    batch_size=2,
                    log_every=5,
                    seed=0,
                    device="mlx",
                ),
                curriculum=curriculum,
                build_dataset=build_dataset,
                out_dir=out,
            )

            # Final checkpoint and metrics should exist.
            self.assertTrue((out / "final.bw4").exists())
            metrics = [
                json.loads(line)
                for line in (out / "metrics.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertGreater(len(metrics), 0)
            # Both phases must appear in the metrics log.
            phases_seen = sorted({m["phase"] for m in metrics if "phase" in m})
            self.assertEqual(phases_seen, [0, 1])
            # Phase 0 logs should report context_length=16; phase 1 should report 32.
            for m in metrics:
                if m.get("phase") == 0:
                    self.assertEqual(m["phase_context_length"], 16)
                elif m.get("phase") == 1:
                    self.assertEqual(m["phase_context_length"], 32)

    def test_curriculum_requires_max_match_config(self) -> None:
        mx.random.seed(0)
        tok = ByteTokenizer()
        curriculum = ContextCurriculum.parse("16:100,32:100")
        # Build config at a wrong (smaller) context_length.
        cfg = BabyWhaleV4Config.tiny(vocab_size=tok.vocab_size, context_length=16)

        def build_dataset(block_size: int) -> TensorPairDataset:
            raise AssertionError("should not be called")

        with (
            tempfile.TemporaryDirectory() as d,
            self.assertRaisesRegex(ValueError, "max_context_length"),
        ):
            pretrain_with_curriculum(
                config=cfg,
                pretrain_config=PretrainConfig(
                    max_steps=1, batch_size=1, log_every=1, device="mlx"
                ),
                curriculum=curriculum,
                build_dataset=build_dataset,
                out_dir=Path(d),
            )


if __name__ == "__main__":
    unittest.main()
