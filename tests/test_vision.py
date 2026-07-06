"""Vision Step-8 components: dynamic tiling geometry + the MLP connector."""

from __future__ import annotations

import unittest

import mlx.core as mx

from baby_whale_v4.vision import VisionMLPConnector, plan_tiles


class TestTiling(unittest.TestCase):
    def test_square_uses_square_grid(self) -> None:
        plan = plan_tiles(100, 100, tile_size=100, max_tiles=4)
        self.assertEqual(plan.cols, plan.rows)

    def test_wide_image_uses_more_columns(self) -> None:
        plan = plan_tiles(200, 100, tile_size=100, max_tiles=4)
        self.assertEqual((plan.cols, plan.rows), (2, 1))

    def test_tall_image_uses_more_rows(self) -> None:
        plan = plan_tiles(100, 300, tile_size=100, max_tiles=6)
        self.assertGreaterEqual(plan.rows, plan.cols)

    def test_respects_max_tiles(self) -> None:
        plan = plan_tiles(1000, 1000, tile_size=100, max_tiles=4)
        self.assertLessEqual(plan.cols * plan.rows, 4)

    def test_includes_thumbnail(self) -> None:
        plan = plan_tiles(200, 100, tile_size=100, max_tiles=4)
        self.assertEqual(plan.n_tiles, plan.cols * plan.rows + 1)
        self.assertEqual(plan.resized_width, plan.cols * plan.tile_size)

    def test_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            plan_tiles(0, 100, tile_size=100, max_tiles=4)
        with self.assertRaises(ValueError):
            plan_tiles(100, 100, tile_size=100, max_tiles=0)


class TestConnector(unittest.TestCase):
    def test_projects_to_n_embd(self) -> None:
        conn = VisionMLPConnector(vision_dim=1152, n_embd=64)
        features = mx.zeros((1, 5, 1152))  # [B, tokens, vision_dim]
        out = conn(features)
        self.assertEqual(tuple(out.shape), (1, 5, 64))

    def test_rejects_wrong_feature_dim(self) -> None:
        conn = VisionMLPConnector(vision_dim=1152, n_embd=64)
        with self.assertRaisesRegex(ValueError, "expected last dim"):
            conn(mx.zeros((1, 5, 999)))

    def test_rejects_bad_dims(self) -> None:
        with self.assertRaises(ValueError):
            VisionMLPConnector(vision_dim=0, n_embd=64)


if __name__ == "__main__":
    unittest.main()
