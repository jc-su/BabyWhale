"""Checkpoint corruption / tamper detection for the .bw4 format."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from baby_whale_v4 import BabyWhaleV4Config, BabyWhaleV4Model
from baby_whale_v4.training.checkpoint import load_checkpoint, save_checkpoint
from baby_whale_v4.typing import ConfigHash

_CORRUPT_ERRORS = (pickle.UnpicklingError, EOFError, ValueError, TypeError)


def _model() -> tuple[BabyWhaleV4Model, BabyWhaleV4Config]:
    mx.random.seed(0)
    cfg = BabyWhaleV4Config.tiny(vocab_size=64, context_length=16)
    return BabyWhaleV4Model(cfg), cfg


def _save(tmp: str, model: BabyWhaleV4Model, cfg: BabyWhaleV4Config) -> Path:
    return save_checkpoint(
        Path(tmp) / "c.bw4", config=cfg, model=model, optimizer=None, scheduler=None, step=5, seed=0
    )


def _tamper(path: Path, mutate) -> None:
    payload = pickle.loads(path.read_bytes())
    mutate(payload)
    path.write_bytes(pickle.dumps(payload))


class TestCheckpointCorruption(unittest.TestCase):
    def test_roundtrip_ok(self) -> None:
        model, cfg = _model()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, model, cfg)
            ckpt = load_checkpoint(path, expected_config_hash=cfg.config_hash())
            self.assertEqual(ckpt.step, 5)
            self.assertEqual(ckpt.config_hash, cfg.config_hash())

    def test_truncated_file_raises(self) -> None:
        model, cfg = _model()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, model, cfg)
            data = path.read_bytes()
            path.write_bytes(data[: len(data) // 2])
            with self.assertRaises(_CORRUPT_ERRORS):
                load_checkpoint(path)

    def test_garbage_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.bw4"
            path.write_bytes(b"not a pickle at all")
            with self.assertRaises(_CORRUPT_ERRORS):
                load_checkpoint(path)

    def test_tampered_config_hash_raises(self) -> None:
        model, cfg = _model()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, model, cfg)
            _tamper(path, lambda p: p.__setitem__("config_hash", "deadbeefdeadbeef"))
            with self.assertRaisesRegex(ValueError, "config_hash does not match"):
                load_checkpoint(path)

    def test_missing_key_raises(self) -> None:
        model, cfg = _model()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, model, cfg)
            _tamper(path, lambda p: p.pop("step"))
            with self.assertRaisesRegex(ValueError, "missing checkpoint keys"):
                load_checkpoint(path)

    def test_unknown_key_raises(self) -> None:
        model, cfg = _model()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, model, cfg)
            _tamper(path, lambda p: p.__setitem__("sneaky", 1))
            with self.assertRaisesRegex(ValueError, "unknown checkpoint keys"):
                load_checkpoint(path)

    def test_bad_format_version_raises(self) -> None:
        model, cfg = _model()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, model, cfg)
            _tamper(path, lambda p: p.__setitem__("format_version", 99))
            with self.assertRaisesRegex(ValueError, "format_version"):
                load_checkpoint(path)

    def test_non_array_model_state_raises(self) -> None:
        model, cfg = _model()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, model, cfg)
            _tamper(path, lambda p: p.__setitem__("model_state", {"w": "not-an-array"}))
            with self.assertRaisesRegex(TypeError, "MLX array"):
                load_checkpoint(path)

    def test_expected_hash_mismatch_raises(self) -> None:
        model, cfg = _model()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, model, cfg)
            with self.assertRaisesRegex(ValueError, "config hash mismatch"):
                load_checkpoint(path, expected_config_hash=ConfigHash("0" * 16))


if __name__ == "__main__":
    unittest.main()
