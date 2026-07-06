"""Vision (Step 8) config fields: validation, hash stability, and back-compat."""

from __future__ import annotations

import dataclasses
import os
import unittest

from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.training.checkpoint import load_checkpoint


class TestVisionConfig(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        self.assertFalse(BabyWhaleV4Config.tiny().enable_vision)

    def test_hash_stable_when_disabled(self) -> None:
        # A config with the vision fields at defaults must hash identically to the
        # same config parsed from a payload that has no vision fields at all.
        cfg = BabyWhaleV4Config.tiny()
        old_payload = {k: v for k, v in cfg.to_dict().items() if "vision" not in k}
        reparsed = BabyWhaleV4Config.from_dict(old_payload)
        self.assertFalse(reparsed.enable_vision)
        self.assertEqual(reparsed.config_hash(), cfg.config_hash())

    def test_enabling_vision_changes_hash(self) -> None:
        base = BabyWhaleV4Config.tiny()
        self.assertNotEqual(
            base.config_hash(), dataclasses.replace(base, enable_vision=True).config_hash()
        )

    def test_validation_when_enabled(self) -> None:
        base = BabyWhaleV4Config.tiny()
        with self.assertRaisesRegex(ValueError, "vision_dim"):
            dataclasses.replace(base, enable_vision=True, vision_dim=0)
        with self.assertRaisesRegex(ValueError, "vision_max_tiles"):
            dataclasses.replace(base, enable_vision=True, vision_max_tiles=0)

    def test_not_validated_when_disabled(self) -> None:
        cfg = dataclasses.replace(BabyWhaleV4Config.tiny(), enable_vision=False, vision_dim=0)
        self.assertEqual(cfg.vision_dim, 0)

    def test_existing_checkpoint_still_loads(self) -> None:
        # Acceptance for the trap: a checkpoint saved before vision fields existed
        # must still load (its stored config_hash matches the recomputed one).
        path = "runs/code3/pretrain/final.bw4"
        if not os.path.exists(path):
            self.skipTest("no local pre-vision checkpoint available")
        ckpt = load_checkpoint(path)
        self.assertFalse(ckpt.config.enable_vision)


if __name__ == "__main__":
    unittest.main()
